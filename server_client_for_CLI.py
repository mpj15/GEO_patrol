"""
Client for the server to be used with command line input.
This is so that a human can play against the AI using the pygame interface.

Will need to copy the player client class from server_koth_game_rollout. 
Use CLI input to get the player's move and format as a touple.
Then use the player client class send_action_request method to send the action to the server. Should stay formatted as dict. Don't need the PZE flat representation.

TODO: Add the renderer to this script so that this player can see the game seperately from the other player.

TODO: Make the player client class a separate file that can be imported by both the server and the CLI client.

"""

#imports
import numpy as np
import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
import game_parameters_case1 as DGP
import orbit_defender2d.king_of_the_hill.game_server as GS
from orbit_defender2d.king_of_the_hill import koth


import zmq
import threading
import concurrent.futures
from copy import deepcopy
from orbit_defender2d.king_of_the_hill import koth
from orbit_defender2d.king_of_the_hill.examples.server_utils import \
    assert_valid_game_state, print_game_info, print_engagement_outcomes_list
from numpy.random import choice, rand, shuffle
from time import sleep

ROUTER_PORT_NUM = 5555
PUB_PORT_NUM = 5556

API_VER_NUM_2P = "v2022.07.26.0000.2p"

ECHO_REQ_MSG_0 = {'context': 'echo', 'data': {'key0': 'value0'}}

# Game Parameters
GAME_PARAMS = koth.KOTHGameInputArgs(
    max_ring=DGP.MAX_RING,
    min_ring=DGP.MIN_RING,
    geo_ring=DGP.GEO_RING,
    init_board_pattern_p1=DGP.INIT_BOARD_PATTERN_P1,
    init_board_pattern_p2=DGP.INIT_BOARD_PATTERN_P2,
    init_fuel=DGP.INIT_FUEL,
    init_ammo=DGP.INIT_AMMO,
    min_fuel=DGP.MIN_FUEL,
    fuel_usage=DGP.FUEL_USAGE,
    engage_probs=DGP.ENGAGE_PROBS,
    illegal_action_score=DGP.ILLEGAL_ACT_SCORE,
    in_goal_points=DGP.IN_GOAL_POINTS,
    adj_goal_points=DGP.ADJ_GOAL_POINTS,
    fuel_points_factor=DGP.FUEL_POINTS_FACTOR,
    win_score=DGP.WIN_SCORE,
    max_turns=DGP.MAX_TURNS,
    fuel_points_factor_bludger=DGP.FUEL_POINTS_FACTOR_BLUDGER,
    )

class PlayerClient(object):
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
                    act = tok[GS.LEGAL_ACTIONS][choice(len(tok[GS.LEGAL_ACTIONS]))]
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
    
    def send_action_req(self, context, actions):
        ''' format and send actions to the game server
        Input actions are formatted as discrete actions from pettingzoo env. 
        That means they should be a kothgame action dictionary and should
        basically be a drop in for the plr_actions in the function above that
        I copied this one from'''
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
            # for tok in self.game_state[GS.TOKEN_STATES]:
            #     if koth.parse_token_id(tok[GS.PIECE_ID])[0] == self.player_id:
            #         act = tok[GS.LEGAL_ACTIONS][choice(len(tok[GS.LEGAL_ACTIONS]))]
            #         act[GS.PIECE_ID] = tok[GS.PIECE_ID]
            #         plr_actions.append(act)

            #the key is the token id, the value is the action touple
            #Take the dictionary of actions and convert it to a list of actions ordered by token id
            #This is because the game server expects the actions to be ordered by token id
            #plr_actions = [actions[tok[GS.PIECE_ID]] for tok in self.game_state[GS.TOKEN_STATES] if koth.parse_token_id(tok[GS.PIECE_ID])[0] == self.player_id]
            if context == U.MOVEMENT:
                plr_actions = [{GS.PIECE_ID: tok[GS.PIECE_ID], GS.ACTION_TYPE: actions[tok[GS.PIECE_ID]][0]} for tok in self.game_state[GS.TOKEN_STATES] if koth.parse_token_id(tok[GS.PIECE_ID])[0] == self.player_id]
                print(plr_actions)
                req_msg[GS.CONTEXT] = GS.MOVE_PHASE
                req_msg[GS.DATA][GS.KIND] = GS.MOVE_PHASE_REQ
                req_msg[GS.DATA][GS.MOVEMENT_SELECTIONS] = plr_actions

            elif context == U.ENGAGEMENT:
                plr_actions = [{GS.PIECE_ID: tok[GS.PIECE_ID], GS.ACTION_TYPE: actions[tok[GS.PIECE_ID]][0], GS.TARGET_ID:actions[tok[GS.PIECE_ID]][1]} for tok in self.game_state[GS.TOKEN_STATES] if koth.parse_token_id(tok[GS.PIECE_ID])[0] == self.player_id]
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
                    self.assert_consistent_registry(msg[GS.DATA][GS.PLAYER_REGISTRY])
                    self.game_state = msg[GS.DATA][GS.GAME_STATE]
                    if msg[GS.DATA][GS.KIND] == GS.ENGAGE_PHASE_RESP:
                        self.engagement_outcomes = msg[GS.DATA][GS.RESOLUTION_SEQUENCE]
                    assert_valid_game_state(game_state=self.game_state)

                print('{} client received and processed message'.format(self.alias))

            except zmq.Again as e:
                # no messages waiting to be processed
                pass

def run_CLI_client():
    #This client will NOT create the game server, the script that generates the AI agent will do that.
    #TODO: eventaully create a seperate script to make the game server, and then run two scripts, one for each agent that connects
    #NOTE: This script creates the 'beta' client and the other script will create the 'alpha' client.

    print("Creating player client...")
    alias_valid = False
    while not alias_valid:
        alias = input("Enter player alias: ")
        #Make sure that alias is a string <10 characters long and is not empty
        if isinstance(alias, str) and len(alias) <= 10 and len(alias) > 0:
            alias_valid = True
        else:
            print("Invalid alias. Alias must be a string <10 characters long and not empty.")

    # alpha_client = context.socket(zmq.REQ)
    # alpha_client.connect("tcp://localhost:{}".format(ROUTER_PORT_NUM))
    plr_client = PlayerClient(
        #router_addr="tcp://localhost:{}".format(ROUTER_PORT_NUM),
        #pub_addr="tcp://localhost:{}".format(PUB_PORT_NUM),
        router_addr="tcp://10.47.7.76:{}".format(ROUTER_PORT_NUM),
        pub_addr="tcp://10.47.7.76:{}".format(PUB_PORT_NUM),
        plr_alias=alias
    )

    # register clients as players in order, with random time between the two
    print("Registering client with alias {}...".format(plr_client.alias))
    plr_client.register_player_req()
    sleep(rand())

    #Send game reset request
    plr_client.game_reset_req()

    #Get game state from server
    cur_game_state = plr_client.game_state

    #Wait for the other player to connect, if necessary
    #If the other player is not connected, game state will be None
    while cur_game_state is None:
        print("Waiting on other player to connect")
        cur_game_state = plr_client.game_state
        sleep(1)
    
    print("\n<==== GAME INITILIZATION ====>")
    print_game_info(game_state=cur_game_state)

    #Create local kothgame object and sync with the game server
    # create and reset pettingzoo env
    penv = PZE.parallel_env(game_params=GAME_PARAMS)
    
    # Start the rendered pygame window
    penv.render(mode="human")

    print("Player Alias: {}".format(plr_client.alias))
    print("Player ID: {}".format(plr_client.player_id))
    print("Player UUID: {}".format(plr_client.player_uuid))


    local_game = koth.KOTHGame(**GAME_PARAMS._asdict()) 

    logfile = koth.start_log_file('./logs/game_log_server_client')

    while not cur_game_state[GS.GAME_DONE]:

        print("\n<==== Turn: {} | Phase: {} ====>".format(cur_game_state[GS.TURN_NUMBER], cur_game_state[GS.TURN_PHASE]))

        turnphase = cur_game_state[GS.TURN_PHASE]
        
        if cur_game_state[GS.TURN_PHASE] == U.DRIFT:
            #send drift phase action request from penv client
            plr_client.send_action_req(context=cur_game_state[GS.TURN_PHASE], actions=[])

        else: #Game state is not DRIFT. Need to get new actions and send to server
            #update the local_game with the new game state from the server
            local_game.game_state, local_game.token_catalog, local_game.n_tokens_alpha, local_game.n_tokens_beta = local_game.arbitrary_game_state_from_server(cur_game_state)

            #actions_dict = local_game.get_input_actions(plr_client.player_id)
            #Get the actions from the player
            acts_received = False
            with concurrent.futures.ThreadPoolExecutor() as executor:
                t = executor.submit(local_game.get_input_actions, plr_client.player_id)
   
                while not acts_received:
                    if t.done():
                        acts_received = True
                        actions_dict = t.result()
                    sleep(3)
                    penv.render(mode="human")

            #Send the actions to the game server
            plr_client.send_action_req(context=cur_game_state[GS.TURN_PHASE], actions=actions_dict)


        # wait for game state to advance
        while cur_game_state[GS.TURN_PHASE] == turnphase and not cur_game_state[GS.GAME_DONE]:
            sleep(1)
            print('waiting for turn phase {} to advance'.format(cur_game_state[GS.TURN_PHASE]))
            cur_game_state = plr_client.game_state

        #update the local_game with the new game state from the server and update the render
        local_game.game_state, local_game.token_catalog, local_game.n_tokens_alpha, local_game.n_tokens_beta = local_game.arbitrary_game_state_from_server(cur_game_state)
        penv.kothgame = local_game
        if actions_dict is not None:
            koth.print_actions(actions_dict)
            actions_dict = None
        koth.log_game_to_file(local_game, logfile=logfile, actions=actions_dict)
        penv.render(mode="human")

        #check if plr_client has attribute engagement_outcomes
        if plr_client.engagement_outcomes is not None:
            print_engagement_outcomes_list(plr_client.engagement_outcomes, file=logfile)

    print("Stopping {} ({}) client thread...".format(plr_client.alias, plr_client.player_id))
    plr_client.stop()

    winner = None
    alpha_score =local_game.game_state[U.P1][U.SCORE]
    beta_score = local_game.game_state[U.P2][U.SCORE]
    if alpha_score > beta_score:
        winner = U.P1
    elif beta_score > alpha_score:
        winner = U.P2
    else:
        winner = 'draw'

    #Show end of game info
    GS.print_endgame_status(cur_game_state)

    if cur_game_state[GS.TOKEN_STATES][0]['fuel'] <= DGP.MIN_FUEL:
        term_cond = "alpha out of fuel"
    elif cur_game_state[GS.TOKEN_STATES][1]['fuel'] <= DGP.MIN_FUEL:
        term_cond = "beta out of fuel"
    elif cur_game_state[GS.SCORE_ALPHA] >= DGP.WIN_SCORE[U.P1]:
        term_cond = "alpha reached Win Score"
    elif cur_game_state[GS.SCORE_BETA]  >= DGP.WIN_SCORE[U.P2]:
        term_cond = "beta reached Win Score"
    elif cur_game_state[GS.TURN_NUMBER]  >= DGP.MAX_TURNS:
        term_cond = "max turns reached" 
    else:
        term_cond = "unknown"
    print("Termination condition: {}".format(term_cond))

    penv.draw_win(winner)

    #Ask user to press spacebar to end the game and close the pygame window
    select_valid = 0
    while not select_valid:
        selection = input("Press spacebar and then return to end game: ")
        if selection == " ":
            select_valid = 1
        else:
            print("Invalid selection. Please press spacebar")

    penv.close()
    
    return



if __name__ == "__main__":
    run_CLI_client()