import os
import datetime
import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.game_server as GS
from orbit_defender2d.king_of_the_hill import koth
import zmq
import threading
from orbit_defender2d.king_of_the_hill.examples.server_utils import \
    assert_valid_game_state, print_game_info, print_engagement_outcomes_list
from numpy.random import choice, rand, shuffle
from time import sleep

ROUTER_PORT_NUM = 5555
PUB_PORT_NUM = 5556

API_VER_NUM_2P = "v2022.07.26.0000.2p"

ECHO_REQ_MSG_0 = {'context': 'echo', 'data': {'key0': 'value0'}}

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
        self.engagement_outcomes = None
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

def log_game_final_to_csv(case_num, game_params, game, file_path, game_type, p1_alias=None, p2_alias=None):
    '''log final game state and engagement outcomes to csv file
        Note: this assumes a kothgame object, not a game server game state'''
    
    #Create the row to write to the csv file, with the following columns:
    #- Case number
    #- type of game
    #- P1 Alias
    #- P2 Alias
    #- Player1 Score
    #- Player2 Score
    #- Number of Turns
    #- Score Difference
    #- Termination Condition1, P1 HVA Fuel
    #- Termination Condition2, P2 HVA Fuel
    #- Termination Condition3, P1 Score
    #- Termination Condition4, P2 Score
    #- Termination Condition5, Max Turns
    #- Date and Time

    if p1_alias is None:
        p1_alias = U.P1
    if p2_alias is None:
        p2_alias = U.P2

    winner = None
    alpha_score =game.game_state[U.P1][U.SCORE]
    beta_score = game.game_state[U.P2][U.SCORE]
    if alpha_score > beta_score:
        winner = U.P1
    elif beta_score > alpha_score:
        winner = U.P2
    else:
        winner = 'draw'

    num_turns = game.game_state[U.TURN_COUNT]

    score_diff = alpha_score - beta_score
    
    cur_game_state = game.game_state
    tc1 = 0
    tc2 = 0
    tc3 = 0
    tc4 = 0
    tc5 = 0
    if cur_game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel <= game_params.MIN_FUEL:
        tc1 = 1
    if cur_game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel <= game_params.MIN_FUEL:
        tc2 = 1
    if cur_game_state[U.P1][U.SCORE] >= game_params.WIN_SCORE[U.P1]:
        tc3 = 1
    if cur_game_state[U.P2][U.SCORE]  >= game_params.WIN_SCORE[U.P2]:
        tc4 = 1
    if cur_game_state[U.TURN_COUNT]  >= game_params.MAX_TURNS:
        tc5 = 1

    #Get date and time
    now = datetime.datetime.now()

    #creat the row, row_out
    row_out = [case_num, game_type, p1_alias, p2_alias, alpha_score, beta_score, num_turns, score_diff, tc1, tc2, tc3, tc4, tc5, now]

    #Check if the file exists, if not, create it and write the header row
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('Case Number, Player1 Score, Player2 Score, Number of Turns, Score Difference, Termination Condition1, Termination Condition2, Termination Condition3, Termination Condition4, Termination Condition5, Date and Time\n')
            #Then write row_out to the file
            f.write(','.join([str(x) for x in row_out]) + '\n')
            #Close the file
            f.close()
    else: #Just append row_out to the file
        with open(file_path, 'a') as f:
            f.write(','.join([str(x) for x in row_out]) + '\n')
            f.close()
    

def get_engagement_dict_from_list(engagement_list):
    """
    Turns a list of engagement tuples or engagement outcome tuples into a list of dicts with the key as the token name and the tuple as the value
    """
    engagement_dict = {}
    for eng in engagement_list:
        engagement_dict[eng.attacker] = eng
    return engagement_dict
