#Game server to run two human players against each other 
#
# Copyright (c) 2024, Michael P. Jones (mpj@alum.mit.edu)
# SPDX-License-Identifier: MIT


import zmq
import threading
import numpy as np
import orbit_defender2d.utils.utils as U
import copy
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
from orbit_defender2d.king_of_the_hill import koth
from orbit_defender2d.king_of_the_hill import game_server as GS
from orbit_defender2d.king_of_the_hill.examples.server_utils import *
from geo_patrol_utils import log_game_final_to_csv

from time import sleep

########### Keep these up to date ############
CSV_FILE_PATH = './logs/server_game_logs.csv'
GAME_TYPE = 'human_vs_human'
CASE_NUM = 1
import game_parameters_default as GP
#############################################

# Game Parameters
GAME_PARAMS = koth.KOTHGameInputArgs(
    max_ring=GP.MAX_RING,
    min_ring=GP.MIN_RING,
    geo_ring=GP.GEO_RING,
    init_board_pattern_p1=GP.INIT_BOARD_PATTERN_P1,
    init_board_pattern_p2=GP.INIT_BOARD_PATTERN_P2,
    init_fuel=GP.INIT_FUEL,
    init_ammo=GP.INIT_AMMO,
    min_fuel=GP.MIN_FUEL,
    fuel_usage=GP.FUEL_USAGE,
    engage_probs=GP.ENGAGE_PROBS,
    illegal_action_score=GP.ILLEGAL_ACT_SCORE,
    in_goal_points=GP.IN_GOAL_POINTS,
    adj_goal_points=GP.ADJ_GOAL_POINTS,
    fuel_points_factor=GP.FUEL_POINTS_FACTOR,
    win_score=GP.WIN_SCORE,
    max_turns=GP.MAX_TURNS,
    fuel_points_factor_bludger=GP.FUEL_POINTS_FACTOR_BLUDGER,
    )

class ListenerClient(object):
    '''bundles REQ and SUB sockets in one object'''
    def __init__(self, router_addr, pub_addr, plr_alias, sub_topic=''):
        ''' Create req and sub socket, and a thread for subsciption handling
        Args:
            router_addr : str
                IP+port number for connection to server ROUTER
            pub_addr : str
                IP+port number for connection to server PUB
            plr_alias : str
                alias used for registered player in KOTH game
            sub_topic : str
                topic for SUB subscription

        Notes:
            Want to use threads, not multiple processes, because I wanted shared memory objects
        
        Refs:
            https://realpython.com/intro-to-python-threading/
            https://stackoverflow.com/questions/24843193/stopping-a-python-thread-running-an-infinite-loop
        '''

        super().__init__()

        ctx = zmq.Context()
        self.alias = plr_alias
        self.player_id = None
        self.game_state = None
        self.actions = None
        self.engagement_outcomes = None
        self.player_registry = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # establish REQ socket and connect to ROUTER
        self.req_socket = ctx.socket(zmq.REQ)
        self.req_socket.connect(router_addr)

        # establish SUB socket and connect to PUB
        self.sub_socket = ctx.socket(zmq.SUB)
        # must set a subscription, missing this step is a common mistake. 
        # https://zguide.zeromq.org/docs/chapter1/#Getting-the-Message-Out
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, sub_topic) 
        self.sub_socket.connect(pub_addr)

        # establish subscription thread
        # make daemon so it is killed when __main__ ends
        # sub_thread = threading.Thread(target=self.subscriber_func, daemon=True)
        sub_thread = threading.Thread(target=self.subscriber_func)
        sub_thread.start()

    def register_player_req(self):
        '''format player registration request, send req, recv response, and check'''

        # format registration request message
        req_msg = dict()
        req_msg['apiVersion'] = API_VER_NUM_2P
        req_msg['context'] = 'playerRegistration'
        req_msg['playerAlias'] = self.alias

        # send registration request
        self.req_socket.send_json(req_msg)
        rep_msg = self.req_socket.recv_json()

        # check registration successful
        assert rep_msg['apiVersion'] == API_VER_NUM_2P
        assert rep_msg['context'] == 'playerRegistration'
        assert rep_msg['data']['kind'] == 'playerRegistrationResponse'
        assert rep_msg['data']['playerAlias'] == self.alias
        assert rep_msg[GS.DATA][GS.PLAYER_ID] in [U.P1, U.P2]
        assert isinstance(rep_msg[GS.DATA][GS.PLAYER_UUID], str)
        assert 'error' not in rep_msg.keys()

        # record backend player id
        self.player_id = rep_msg[GS.DATA][GS.PLAYER_ID]
        self.player_uuid = rep_msg[GS.DATA][GS.PLAYER_UUID]
    
    def assert_consistent_registry(self, registry):
        '''check that registry has not changed unexpectedly'''
        reg_entry = [reg for reg in registry if reg[GS.PLAYER_ALIAS]==self.alias]
        assert len(reg_entry) == 1
        reg_entry = reg_entry[0]
        assert reg_entry[GS.PLAYER_ID] == self.player_id, "Expect ID {}, got {}".format(self.player_id, reg_entry[GS.PLAYER_ID])

    def game_reset_req(self):
        '''format game reset request, send request, recv response, and check'''

        # format game reset request message
        req_msg = dict()
        req_msg['apiVersion'] = API_VER_NUM_2P
        req_msg['context'] = 'gameReset'
        req_msg['playerAlias'] = self.alias
        req_msg['playerUUID'] = self.player_uuid

        # send game reset request
        self.req_socket.send_json(req_msg)
        rep_msg = self.req_socket.recv_json()

        # check reset waiting or advancing
        assert rep_msg['apiVersion'] == API_VER_NUM_2P
        assert rep_msg['context'] == 'gameReset'
        assert rep_msg['data']['kind'] in ['waitingResponse', 'advancingResponse']
        assert 'error' not in rep_msg.keys()

    def send_random_action_req(self, context):
        ''' format and send random-yet-legal action depending on context '''
        req_msg = dict()
        req_msg['apiVersion'] = API_VER_NUM_2P
        req_msg['playerAlias'] = self.alias
        req_msg['playerUUID'] = self.player_uuid

        if context == U.DRIFT:
            req_msg['context'] = 'driftPhase'

        else:
            # select random valid action formatted as client request dictionary
            plr_actions = []
            req_msg[GS.DATA] = dict()
            for tok in self.game_state[GS.TOKEN_STATES]:
                if koth.parse_token_id(tok[GS.PIECE_ID])[0] == self.player_id:
                    #act = tok[GS.LEGAL_ACTIONS][choice(len(tok[GS.LEGAL_ACTIONS]))]
                    act = tok[GS.LEGAL_ACTIONS][0]
                    act[GS.PIECE_ID] = tok[GS.PIECE_ID]
                    plr_actions.append(act)

            if context == U.MOVEMENT:
                req_msg[GS.CONTEXT] = GS.MOVE_PHASE
                req_msg[GS.DATA][GS.KIND] = GS.MOVE_PHASE_REQ
                req_msg[GS.DATA][GS.MOVEMENT_SELECTIONS] = plr_actions

            elif context == U.ENGAGEMENT:
                req_msg[GS.CONTEXT] = GS.ENGAGE_PHASE
                req_msg[GS.DATA][GS.KIND] = GS.ENGAGE_PHASE_REQ
                req_msg[GS.DATA][GS.ENGAGEMENT_SELECTIONS] = plr_actions

            else:
                raise ValueError

        # send game reset request
        self.req_socket.send_json(req_msg)
        rep_msg = self.req_socket.recv_json()

        # check reset waiting or advancing
        assert rep_msg[GS.API_VERSION] == API_VER_NUM_2P
        assert rep_msg[GS.CONTEXT] in [GS.DRIFT_PHASE, GS.MOVE_PHASE, GS.ENGAGE_PHASE]
        assert 'error' not in rep_msg.keys(), "error received: {}".format(rep_msg[GS.ERROR][GS.MESSAGE])
        assert rep_msg[GS.DATA][GS.KIND] in [GS.WAITING_RESP, GS.ADVANCING_RESP]
            

    def drift_phase_req(self):
        '''format drift request, send msg, recv response, and check'''

        # format drift request
        req_msg = dict()
        req_msg['apiVersion'] = API_VER_NUM_2P
        req_msg['context'] = 'driftPhase'
        req_msg['playerAlias'] = self.alias
        req_msg['playerUUID'] = self.player_uuid

        # send drift request
        self.req_socket.send_json(req_msg)
        rep_msg = self.req_socket.recv_json()

        # check reset waiting or advancing
        assert rep_msg['apiVersion'] == API_VER_NUM_2P
        assert rep_msg['context'] == 'driftPhase'
        assert rep_msg['data']['kind'] in ['waitingResponse', 'advancingResponse']
        assert 'error' not in rep_msg.keys()


    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def subscriber_func(self):
        '''wait for and process message published on PUB
        
        Refs:
            https://stackoverflow.com/questions/26012132/zero-mq-socket-recv-call-is-blocking
        '''

        while not self.stopped():

            try:

                # wait for published message
                msg = self.sub_socket.recv_json(flags=zmq.NOBLOCK)

                # check message content
                assert msg[GS.API_VERSION] == API_VER_NUM_2P, "expected {}, got {}".format(API_VER_NUM_2P, msg[GS.API_VERSION])
                assert GS.ERROR not in msg.keys()

                # if registry response, wait a little while for request socket in other thread to 
                # to have time to receive registry info and update client info
                if msg[GS.CONTEXT] == GS.PLAYER_REGISTRATION:
                    sleep(0.25)

                # verify registry and update game state (shared memory, therefore use a lock)
                with self._lock:
                    #self.assert_consistent_registry(msg[GS.DATA][GS.PLAYER_REGISTRY])
                    self.game_state = msg[GS.DATA][GS.GAME_STATE]
                    self.player_registry = msg[GS.DATA][GS.PLAYER_REGISTRY]
                    self.actions = msg[GS.DATA][GS.ACTION_SELECTIONS]
                    if msg[GS.DATA][GS.KIND] == GS.ENGAGE_PHASE_RESP:
                        self.engagement_outcomes = msg[GS.DATA][GS.RESOLUTION_SEQUENCE]
                    assert_valid_game_state(game_state=self.game_state)

                print('{} client received and processed message on SUB!'.format(self.alias))

            except zmq.Again as e:
                # no messages waiting to be processed
                pass


ROUTER_PORT_NUM = 5555
PUB_PORT_NUM = 5556

API_VER_NUM_2P = "v2022.07.26.0000.2p"

ECHO_REQ_MSG_0 = {'context': 'echo', 'data': {'key0': 'value0'}}

def start_server():
    # create game object
    game = koth.KOTHGame(**GAME_PARAMS._asdict())

    # create game server
    comm_configs = {
        GS.ROUTER_PORT: ROUTER_PORT_NUM,
        GS.PUB_PORT: PUB_PORT_NUM
    }
    game_server = GS.TwoPlayerGameServer(game=game, comm_configs=comm_configs)

    # start game server object
    game_server.start()

    # remove local game objects
    # these are forked in a new thread, don't 
    # trick yourself into thinking they are the same object
    del game

    return game_server

def create_listener_client():
    #Register a listening client to monitor the game
    listener_client = ListenerClient(
        router_addr="tcp://localhost:{}".format(ROUTER_PORT_NUM),
        pub_addr="tcp://localhost:{}".format(PUB_PORT_NUM),
        plr_alias='listener_client',)
    return listener_client

def run_listener(game_server, listener_client, render=True):   
    # Don't register the client as a player, just subscribe to the pub socket
    tmp_game_state = listener_client.game_state
    game_started = False
    game_finised = False

    while tmp_game_state is None:
        sleep(5)
        tmp_game_state = listener_client.game_state
        print("Waiting for a game to begin")
    
    #Get the player registry
    no_player_reg = True
    while no_player_reg:
        plr_reg = listener_client.player_registry
        if plr_reg is not None:
            no_player_reg = False
        else:
            print("Waiting for player registry...")
            sleep(1)
    
    game_started = True
    print("Game started")
    #start logfile
    p1_alias = plr_reg[0][GS.PLAYER_ID]+": "+plr_reg[0][GS.PLAYER_ALIAS]
    p2_alias = plr_reg[1][GS.PLAYER_ID]+": "+plr_reg[1][GS.PLAYER_ALIAS]
    print("Player 1: ", p1_alias)
    print("Player 2: ", p2_alias)
    logfile = koth.start_log_file('./logs/game_log_server', p1_alias=p1_alias, p2_alias=p2_alias)

    if render:
        #Create local penv game to render
        penv = PZE.parallel_env(game_params=GAME_PARAMS, training_randomize=False, plr_aliases=[p1_alias, p2_alias])
        penv.reset()
        penv.render(mode='human')

    turn_phase = tmp_game_state[GS.TURN_PHASE]

    while tmp_game_state[GS.GAME_DONE] is False:
        tmp_game_state = listener_client.game_state
        local_game = penv.kothgame
        local_game.game_state, local_game.token_catalog, local_game.n_tokens_alpha, local_game.n_tokens_beta = GS.arbitrary_game_state_from_server(GAME_PARAMS,tmp_game_state)
        penv.kothgame = local_game
        
        if listener_client.actions is not None:
            new_dict = {}
            for key, value in listener_client.actions.items():
                if len(value) == 3: #then it is an engagement 
                    new_dict[key] = U.EngagementTuple(action_type=value[0], target=value[1], prob=value[2])
                elif len(value) > 3: #then it is a engagement outcome
                    new_dict[key] = U.EngagementOutcomeTuple(action_type=value[0], attacker=value[1], target=value[2], guardian=value[3], prob=value[4], success=value[5])
                elif len(value) == 1: #movement 
                    new_dict[key] = U.MovementTuple(action_type=value[0])
            #penv.actions = new_dict
            actions_dict = new_dict
        
        if listener_client.engagement_outcomes is not None:
            engagement_outcomes = listener_client.engagement_outcomes
            eg_outs_tuple_list = GS.arbitrary_engagement_outcomes_from_server(game_params=GAME_PARAMS,engagement_outcomes=engagement_outcomes)[0]
            local_game.engagement_outcomes = eg_outs_tuple_list
            koth.print_engagement_outcomes(local_game.engagement_outcomes)
            penv.kothgame.engagement_outcomes = eg_outs_tuple_list
            penv.actions = GS.arbitrary_engagement_outcomes_from_server(game_params=GAME_PARAMS,engagement_outcomes=engagement_outcomes)[1]
        else:
            engagement_outcomes = None
            penv.kothgame.engagement_outcomes = None

        if tmp_game_state[GS.TURN_PHASE] != turn_phase:
            #Log the info from the prior turn phase
            local_game.game_state[U.TURN_PHASE] = turn_phase
            if turn_phase == U.DRIFT:
                local_game.game_state[U.TURN_COUNT] = tmp_game_state[GS.TURN_NUMBER] - 1
            koth.print_game_info(game=local_game)
            koth.print_actions(actions=actions_dict)
            koth.log_game_to_file(game=local_game, logfile=logfile,actions=actions_dict)
            actions_dict = None
            # if tmp_game_state[GS.TURN_PHASE] == U.DRIFT:
            #     if hasattr(listener_client, 'engagement_outcomes'):
            #         with open(logfile, 'a') as f:
            #             print_engagement_outcomes_list(listener_client.engagement_outcomes, file=f)
            #             f.close()
            turn_phase = tmp_game_state[GS.TURN_PHASE]
            local_game.game_state[U.TURN_PHASE] = turn_phase
            local_game.game_state[U.TURN_COUNT] = tmp_game_state[GS.TURN_NUMBER]
        
        if render:
            penv.render(mode='human')

        if tmp_game_state[GS.GAME_DONE] is True:
            koth.print_endgame_status(local_game)
            koth.log_game_to_file(local_game, logfile)
            log_game_final_to_csv(CASE_NUM,GAME_PARAMS,local_game,CSV_FILE_PATH,GAME_TYPE,p1_alias=p1_alias,p2_alias=p2_alias)
            break
        print("Waiting for game to finish")
        sleep(1)

    # Game is finished, print final info and get winner
    winner = None
    alpha_score =local_game.game_state[U.P1][U.SCORE]
    beta_score = local_game.game_state[U.P2][U.SCORE]
    if alpha_score > beta_score:
        winner = U.P1
    elif beta_score > alpha_score:
        winner = U.P2
    else:
        winner = 'draw'

    
    if render:
        penv.render(mode='human')
        sleep(1)
        penv.draw_win(winner)
        sleep(10)
        penv.close()

    game_finised = True
    print("Game finished")
    
    restart_server(game_server, listener_client)

def restart_server(game_server, listener_client):
    #Restart the game server and listener client
    #Stop the listener client if it exists
    listener_client.stop()
    del listener_client

    game_server.terminate()
    game_server.join()
    del game_server

    game_server = start_server()
    listener_client = create_listener_client()
    run_listener(game_server, listener_client)

if __name__ == "__main__":
    game_server = start_server()
    listener_client = create_listener_client()
    run_listener(game_server, listener_client)
