# Copyright (c) 2024, Michael P. Jones (mpj@alum.mit.edu)
# SPDX-License-Identifier: MIT
#
# Description: This script contains the functions that run the game in different modes.

import numpy as np
import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.game_server as GS
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
from orbit_defender2d.king_of_the_hill import koth
import torch
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
from time import sleep
import concurrent.futures
from orbit_defender2d.king_of_the_hill.examples.server_utils import print_game_info

from geo_patrol_utils import PlayerClient, get_engagement_dict_from_list, log_game_final_to_csv, ROUTER_PORT_NUM, PUB_PORT_NUM
import game_parameters_default as DGP
import game_parameters_case1 as GP_1
import game_parameters_case2 as GP_2
import game_parameters_case3 as GP_3
import game_parameters_case4 as GP_4
import game_parameters_case5 as GP_5

CSV_FILE_PATH = './logs/logfile.csv'

def ai_v_ai_game_mode(model_path_alpha, model_path_beta, case_num):
    if case_num == 1:
        GP = GP_1
    elif case_num == 2:
        GP = GP_2
    elif case_num == 3:
        GP = GP_3
    elif case_num == 4:
        GP = GP_4
    elif case_num == 5:
        GP = GP_5
    else:
        GP = DGP
    run_game_ai_vs_ai(model_path_alpha,model_path_beta, GP, case_num)

def human_v_ai_game_mode(model_path_alpha, model_path_beta, case_num):
    if case_num == 1:
        GP = GP_1
    elif case_num == 2:
        GP = GP_2
    elif case_num == 3:
        GP = GP_3
    elif case_num == 4:
        GP = GP_4
    elif case_num == 5:
        GP = GP_5
    else:
        GP = DGP
    if model_path_alpha == None and model_path_beta == None:
        print("Both model paths are None. Exiting.")
        return
    elif model_path_alpha != None and model_path_beta != None:
        print("Both model paths are not None. Exiting.")
        return
    if model_path_alpha == None:
           #The computer plays as beta, player 2, and the human plays as alpha, player 1
           #Alpha is offense and beta is defense, so human in this case is offense
        run_game_humanA_v_aiB(model_path_beta, GP, case_num)
    if model_path_beta == None:
        #The computer plays as alpha, player 1, and the human plays as beta, player 2
        #Alpha is offense and beta is defense, so human in this case is defense
        run_game_humanB_v_aiA(model_path_alpha, GP, case_num)

def run_server_client_game_mode(gs_host_addr,case_num):
    if case_num == 1:
        GP = GP_1
    elif case_num == 2:
        GP = GP_2
    elif case_num == 3:
        GP = GP_3
    elif case_num == 4:
        GP = GP_4
    elif case_num == 5:
        GP = GP_5
    else:
        GP = DGP
    run_server_client_game(gs_host_addr, GP, case_num)


def run_game_ai_vs_ai(model_path_alpha, model_path_beta, GP, case_num):
    game_type = "ai_v_ai"
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
    # create and reset pettingzoo env
    penv = PZE.parallel_env(game_params=GAME_PARAMS, training_randomize=False, plr_aliases=None)
    obs = penv.reset()
    # Update rendered pygame window
    #penv.render(mode="debug")
    
    
    model_alpha = torch.jit.load(model_path_alpha)
    model_beta = torch.jit.load(model_path_beta)
    model_alpha.eval()
    model_beta.eval()

    #start logfile
    logfile = koth.start_log_file('./logs/game_log_AIvsAI')

    # iterate through game with valid random actions
    while True:

        print("\n<==== Turn: {} | Phase: {} ====>".format(
            penv.kothgame.game_state[U.TURN_COUNT], 
            penv.kothgame.game_state[U.TURN_PHASE]))
        koth.print_scores(penv.kothgame)


        #Get actions from loaded policy to compare with actions from ray policy
        new_obs_tensor_alpha = torch.tensor(obs[U.P1]['observation'], dtype=torch.float32)
        new_obs_tensor_beta = torch.tensor(obs[U.P2]['observation'], dtype=torch.float32)

        #Get Action masks
        new_obs_tensor_am_alpha = torch.tensor(obs[U.P1]['action_mask'], dtype=torch.float32)
        new_obs_tensor_am_alpha = new_obs_tensor_am_alpha.reshape(-1,)
        new_obs_tensor_am_beta = torch.tensor(obs[U.P2]['action_mask'], dtype=torch.float32)
        new_obs_tensor_am_beta = new_obs_tensor_am_beta.reshape(-1,)
        
        #Concatenate the action mask and observation tensors
        new_obs_dict_alpha = {'obs':torch.cat((new_obs_tensor_am_alpha,new_obs_tensor_alpha),dim=0).unsqueeze(0)}
        new_obs_dict_beta = {'obs':torch.cat((new_obs_tensor_am_beta,new_obs_tensor_beta),dim=0).unsqueeze(0)}

        #Get the actions from the loaded policy
        acts_alpha = model_alpha(new_obs_dict_alpha, [torch.tensor([0.0], dtype=torch.float32)], torch.tensor([0], dtype=torch.int64))
        acts_beta = model_beta(new_obs_dict_beta, [torch.tensor([0.0], dtype=torch.float32)], torch.tensor([0], dtype=torch.int64))      
    
        #Format the acts_model as a gym spaces touple which is the same as the action space tuple defined in 
        # pettingzoo_env.py as self.per_player = spaces.Tuple(tuple([self.per_token for _ in range(self.n_tokens_per_player)])) 
        # where self.per_token = spaces.Discrete(38) and self.n_tokens_per_player = 11
        #TODO: Get rid of hard coded values 11 and 38. 
        acts_alpha_reshaped = acts_alpha[0].reshape(11,38)
        acts_alpha_list = torch.argmax(acts_alpha_reshaped, dim=1).tolist()

        acts_beta_reshaped = acts_beta[0].reshape(11,38)
        acts_beta_list = torch.argmax(acts_beta_reshaped, dim=1).tolist()

        acts_alpha_tuple = tuple(acts_alpha_list)
        acts_beta_tuple = tuple(acts_beta_list)

        #Decode the actions from the model into the action dicts that can be used by koth
        actions_alpha_dict = penv.decode_discrete_player_action(U.P1,acts_alpha_tuple)
        actions_beta_dict = penv.decode_discrete_player_action(U.P2,acts_beta_tuple)

        actions = {}
        actions.update(actions_alpha_dict)
        actions.update(actions_beta_dict)
        penv.actions = actions #Add actions to the penv sot that they can be rendered
        
        koth.print_actions(actions)
        koth.log_game_to_file(penv.kothgame, logfile=logfile, actions=actions)

        # Update rendered pygame window
        penv.render(mode="debug")
        #penv.screen_shot(file_name=filename)

        # encode actions into flat gym space
        encoded_actions = penv.encode_all_discrete_actions(actions=actions)

        # apply encoded actions
        obs, rewards, dones, info = penv.step(actions=encoded_actions)

        # assert zero-sum game
        assert np.isclose(rewards[U.P1], -rewards[U.P2])

        #If game_sate is "MOVEMENT" Then print the engagement outcomes from the prior ENGAGEMENT phase
        if penv.kothgame.game_state[U.TURN_PHASE] == U.MOVEMENT and penv.kothgame.game_state[U.TURN_COUNT] > 0:
            koth.print_engagement_outcomes(penv.kothgame.engagement_outcomes)

        # assert rewards only from final timestep
        if any([dones[d] for d in dones.keys()]):
            assert np.isclose(rewards[U.P1], 
                penv.kothgame.game_state[U.P1][U.SCORE] - penv.kothgame.game_state[U.P2][U.SCORE])
            break
        else:
            assert np.isclose(rewards[U.P1], 0.0)

    winner = None
    alpha_score = penv.kothgame.game_state[U.P1][U.SCORE]
    beta_score = penv.kothgame.game_state[U.P2][U.SCORE]
    if alpha_score > beta_score:
        winner = U.P1
    elif beta_score > alpha_score:
        winner = U.P2
    else:
        winner = 'draw'
    #t_now = datetime.datetime.now()
    #filename = "od2d_screen_shot_"+t_now.strftime("%Y%m%d_%H%M%S")+".png"
    
    #penv.screen_shot(file_name=filename)

    #Print final engagement outcomes
    koth.print_engagement_outcomes(penv.kothgame.engagement_outcomes)
    koth.log_game_to_file(penv.kothgame, logfile=logfile, actions=actions)
    log_game_final_to_csv(case_num, GAME_PARAMS,penv.kothgame, CSV_FILE_PATH, game_type, p1_alias=U.P1+":AI", p2_alias=U.P2+":AI", associated_logfile=logfile)

    cur_game_state = penv.kothgame.game_state
    if cur_game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel <= GAME_PARAMS.min_fuel:
        print(U.P1+" seeker out of fuel")
    if cur_game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel <= GAME_PARAMS.min_fuel:
        print(U.P2+" seeker out of fuel")
    if cur_game_state[U.P1][U.SCORE] >= GAME_PARAMS.win_score[U.P1]:
        print(U.P1+" reached Win Score")
    if cur_game_state[U.P2][U.SCORE]  >= GAME_PARAMS.win_score[U.P2]:
        print(U.P2+" reached Win Score")
    if cur_game_state[U.TURN_COUNT]  >= GAME_PARAMS.max_turns:
        print("max turns reached")
        
    print("\n====GAME FINISHED====\nWinner: {}\nScore: {}|{}\n=====================\n".format(winner, alpha_score, beta_score))

    penv.draw_win(winner)

    #Ask user to press spacebar to end the game and close the pygame window
    select_valid = 0
    while not select_valid:
        selection = input("Press spacebar and then return to end game: ")
        if selection == " ":
            select_valid = 1
        else:
            print("Invalid selection. Please press spacebar")
    return


def run_game_ai_vs_ai_no_render(model_path_alpha, model_path_beta, GP, case_num):
    
    game_type = "ai_v_ai"

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
    # create and reset pettingzoo env
    penv = PZE.parallel_env(game_params=GAME_PARAMS, training_randomize=False, plr_aliases=None)
    obs = penv.reset()
    
    model_alpha = torch.jit.load(model_path_alpha)
    model_beta = torch.jit.load(model_path_beta)
    model_alpha.eval()
    model_beta.eval()

    # iterate through game with valid random actions
    while True:

        #Get actions from loaded policy to compare with actions from ray policy
        new_obs_tensor_alpha = torch.tensor(obs[U.P1]['observation'], dtype=torch.float32)
        new_obs_tensor_beta = torch.tensor(obs[U.P2]['observation'], dtype=torch.float32)

        #Get Action masks
        new_obs_tensor_am_alpha = torch.tensor(obs[U.P1]['action_mask'], dtype=torch.float32)
        new_obs_tensor_am_alpha = new_obs_tensor_am_alpha.reshape(-1,)
        new_obs_tensor_am_beta = torch.tensor(obs[U.P2]['action_mask'], dtype=torch.float32)
        new_obs_tensor_am_beta = new_obs_tensor_am_beta.reshape(-1,)
        
        #Concatenate the action mask and observation tensors
        new_obs_dict_alpha = {'obs':torch.cat((new_obs_tensor_am_alpha,new_obs_tensor_alpha),dim=0).unsqueeze(0)}
        new_obs_dict_beta = {'obs':torch.cat((new_obs_tensor_am_beta,new_obs_tensor_beta),dim=0).unsqueeze(0)}

        #Get the actions from the loaded policy
        acts_alpha = model_alpha(new_obs_dict_alpha, [torch.tensor([0.0], dtype=torch.float32)], torch.tensor([0], dtype=torch.int64))
        acts_beta = model_beta(new_obs_dict_beta, [torch.tensor([0.0], dtype=torch.float32)], torch.tensor([0], dtype=torch.int64))      
    
        #Format the acts_model as a gym spaces touple which is the same as the action space tuple defined in 
        acts_alpha_reshaped = acts_alpha[0].reshape(11,38)
        acts_alpha_list = torch.argmax(acts_alpha_reshaped, dim=1).tolist()

        acts_beta_reshaped = acts_beta[0].reshape(11,38)
        acts_beta_list = torch.argmax(acts_beta_reshaped, dim=1).tolist()

        acts_alpha_tuple = tuple(acts_alpha_list)
        acts_beta_tuple = tuple(acts_beta_list)

        #Decode the actions from the model into the action dicts that can be used by koth
        actions_alpha_dict = penv.decode_discrete_player_action(U.P1,acts_alpha_tuple)
        actions_beta_dict = penv.decode_discrete_player_action(U.P2,acts_beta_tuple)

        actions = {}
        actions.update(actions_alpha_dict)
        actions.update(actions_beta_dict)
        penv.actions = actions #Add actions to the penv sot that they can be rendered

        # encode actions into flat gym space
        encoded_actions = penv.encode_all_discrete_actions(actions=actions)

        # apply encoded actions
        obs, rewards, dones, info = penv.step(actions=encoded_actions)

        # assert zero-sum game
        assert np.isclose(rewards[U.P1], -rewards[U.P2])

        # assert rewards only from final timestep
        if any([dones[d] for d in dones.keys()]):
            assert np.isclose(rewards[U.P1], 
                penv.kothgame.game_state[U.P1][U.SCORE] - penv.kothgame.game_state[U.P2][U.SCORE])
            break
        else:
            assert np.isclose(rewards[U.P1], 0.0)

    log_game_final_to_csv(case_num, GAME_PARAMS,penv.kothgame, './logs/sample_logfile.csv', game_type, p1_alias=U.P1+":AI", p2_alias=U.P2+":AI", associated_logfile=None)

    return

def run_game_humanB_v_aiA(model_path_alpha, GP, case_num):
    #Get the user's name:
    game_type = "ai_v_human"
    alias_valid = False
    while not alias_valid:
        alias = input("Enter player name: ")
        #Make sure that alias is a string <10 characters long and is not empty
        if isinstance(alias, str) and len(alias) <= 20 and len(alias) > 0:
            alias_valid = True
        else:
            print("Invalid alias. Alias must be a string <20 characters long and not empty.")
    print("Player {} is: {}".format(U.P2, alias))
    sleep(1)
    #start logfile
    logfile = koth.start_log_file('./logs/game_log_AI_vs_player', p1_alias="AI", p2_alias=alias)

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
    # create and reset pettingzoo env
    penv = PZE.parallel_env(game_params=GAME_PARAMS, training_randomize=False, plr_aliases=["AI", alias])
    obs = penv.reset()

    

    # Render pygame window
    penv.render(mode="human")
    #penv.screen_shot(file_name="./od2d_screen_shot_new.png")
    
    model_alpha = torch.jit.load(model_path_alpha)
    model_alpha.eval()

    # iterate through game with valid random actions
    while True:
        # Update rendered pygame window
        penv.render(mode="human")

        print("\n<==== Turn: {} | Phase: {} ====>".format(
            penv.kothgame.game_state[U.TURN_COUNT], 
            penv.kothgame.game_state[U.TURN_PHASE]))
        koth.print_scores(penv.kothgame)

        #Get actions from loaded policy to compare with actions from ray policy
        new_obs_tensor_alpha = torch.tensor(obs[U.P1]['observation'], dtype=torch.float32)

        #Get Action masks
        new_obs_tensor_am_alpha = torch.tensor(obs[U.P1]['action_mask'], dtype=torch.float32)
        new_obs_tensor_am_alpha = new_obs_tensor_am_alpha.reshape(-1,)
        
        #Concatenate the action mask and observation tensors
        new_obs_dict_alpha = {'obs':torch.cat((new_obs_tensor_am_alpha,new_obs_tensor_alpha),dim=0).unsqueeze(0)}

        #Get the actions from the loaded policy
        acts_alpha = model_alpha(new_obs_dict_alpha, [torch.tensor([0.0], dtype=torch.float32)], torch.tensor([0], dtype=torch.int64))      
    
        #Format the acts_model as a gym spaces touple which is the same as the action space tuple defined in 
        # pettingzoo_env.py as self.per_player = spaces.Tuple(tuple([self.per_token for _ in range(self.n_tokens_per_player)])) 
        # where self.per_token = spaces.Discrete(38) and self.n_tokens_per_player = 11
        #TODO: Get rid of hard coded values 11 and 38. 
        acts_alpha_reshaped = acts_alpha[0].reshape(11,38)
        acts_alpha_list = torch.argmax(acts_alpha_reshaped, dim=1).tolist()
        acts_alpha_tuple = tuple(acts_alpha_list)

        #Decode the actions from the model into the action dicts that can be used by koth
        actions_alpha_dict = penv.decode_discrete_player_action(U.P1,acts_alpha_tuple)

        #Get the actions from the player
        acts_received = False
        with concurrent.futures.ThreadPoolExecutor() as executor:
            t = executor.submit(koth.KOTHGame.get_input_actions, penv.kothgame, U.P2)
            while not acts_received:
                if t.done():
                    acts_received = True
                    actions_beta_dict = t.result()
                sleep(1)
                penv.render(mode="human")

        actions = {}
        actions.update(actions_alpha_dict)
        actions.update(actions_beta_dict)
        penv.actions = actions #Add actions to the penv sot that they can be rendered
        
        #koth.print_actions(actions)
        koth.log_game_to_file(penv.kothgame, logfile=logfile, actions=actions)

        # Update rendered pygame window
        penv.render(mode="human")

        # encode actions into flat gym space
        encoded_actions = penv.encode_all_discrete_actions(actions=actions)

        # apply encoded actions
        obs, rewards, dones, info = penv.step(actions=encoded_actions)

        # assert zero-sum game
        assert np.isclose(rewards[U.P1], -rewards[U.P2])

        #If game_sate is "MOVEMENT" Then print the engagement outcomes from the prior ENGAGEMENT phase
        if penv.kothgame.game_state[U.TURN_PHASE] == U.MOVEMENT and penv.kothgame.game_state[U.TURN_COUNT] > 0:
            koth.print_engagement_outcomes(penv.kothgame.engagement_outcomes)
            engagement_outcomes_dict = get_engagement_dict_from_list(penv.kothgame.engagement_outcomes)
            penv.actions = engagement_outcomes_dict #Add actions to the penv sot that they can be rendered
            penv._eg_outcomes_phase = True
            penv.kothgame.game_state[U.TURN_PHASE] = U.ENGAGEMENT
            penv.render(mode="human")
            sleep(5)
            penv._eg_outcomes_phase = False
            penv.kothgame.game_state[U.TURN_PHASE] = U.MOVEMENT

        # assert rewards only from final timestep
        if any([dones[d] for d in dones.keys()]):
            assert np.isclose(rewards[U.P1], 
                penv.kothgame.game_state[U.P1][U.SCORE] - penv.kothgame.game_state[U.P2][U.SCORE])
            break
        else:
            assert np.isclose(rewards[U.P1], 0.0)

    winner = None
    alpha_score = penv.kothgame.game_state[U.P1][U.SCORE]
    beta_score = penv.kothgame.game_state[U.P2][U.SCORE]
    if alpha_score > beta_score:
        winner = U.P1
    elif beta_score > alpha_score:
        winner = U.P2
    else:
        winner = 'draw'

    #Print final engagement outcomes
    koth.print_engagement_outcomes(penv.kothgame.engagement_outcomes)
    koth.log_game_to_file(penv.kothgame, logfile=logfile, actions=actions)
    log_game_final_to_csv(case_num, GAME_PARAMS,penv.kothgame, CSV_FILE_PATH, game_type, p1_alias=U.P1+":AI", p2_alias=U.P2+":"+alias, associated_logfile=logfile)

    cur_game_state = penv.kothgame.game_state
    if cur_game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel <= GAME_PARAMS.min_fuel:
        print(U.P1+" seeker out of fuel")
    if cur_game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel <= GAME_PARAMS.min_fuel:
        print(U.P2+" seeker out of fuel")
    if cur_game_state[U.P1][U.SCORE] >= GAME_PARAMS.win_score[U.P1]:
        print(U.P1+" reached Win Score")
    if cur_game_state[U.P2][U.SCORE]  >= GAME_PARAMS.win_score[U.P2]:
        print(U.P2+" reached Win Score")
    if cur_game_state[U.TURN_COUNT]  >= GAME_PARAMS.max_turns:
        print("max turns reached")
        
    print("\n====GAME FINISHED====\nWinner: {}\nScore: {}|{}\n=====================\n".format(winner, alpha_score, beta_score))

    penv.draw_win(winner)
    
    #Ask user to press spacebar to end the game and close the pygame window
    select_valid = 0
    while not select_valid:
        selection = input("Press spacebar and then return to end game: ")
        if selection == " ":
            select_valid = 1
        else:
            print("Invalid selection. Please press spacebar")
    return

def run_game_humanA_v_aiB(model_path_beta, GP, case_num):
    game_type = "human_v_ai"
    #Get the user's name:
    alias_valid = False
    while not alias_valid:
        alias = input("Enter player name: ")
        #Make sure that alias is a string <10 characters long and is not empty
        if isinstance(alias, str) and len(alias) <= 20 and len(alias) > 0:
            alias_valid = True
        else:
            print("Invalid alias. Alias must be a string <20 characters long and not empty.")
    print("Player {} is: {}".format(U.P1, alias))
    sleep(1)
    #start logfile
    logfile = koth.start_log_file('./logs/game_log_AI_vs_player', p1_alias=alias, p2_alias="AI")

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
    # create and reset pettingzoo env
    penv = PZE.parallel_env(game_params=GAME_PARAMS, training_randomize=False, plr_aliases=[alias,"AI"])
    obs = penv.reset()

    # Render pygame window
    penv.render(mode="human")
    #penv.screen_shot(file_name="./od2d_screen_shot_new.png")
    
    model_beta = torch.jit.load(model_path_beta)
    model_beta.eval()

    # iterate through game with valid random actions
    while True:
        # Update rendered pygame window
        penv.render(mode="human")

        print("\n<==== Turn: {} | Phase: {} ====>".format(
            penv.kothgame.game_state[U.TURN_COUNT], 
            penv.kothgame.game_state[U.TURN_PHASE]))
        koth.print_scores(penv.kothgame)

        #Get actions from loaded policy to compare with actions from ray policy
        new_obs_tensor_beta = torch.tensor(obs[U.P2]['observation'], dtype=torch.float32)

        #Get Action masks
        new_obs_tensor_am_beta = torch.tensor(obs[U.P2]['action_mask'], dtype=torch.float32)
        new_obs_tensor_am_beta = new_obs_tensor_am_beta.reshape(-1,)
        
        #Concatenate the action mask and observation tensors
        new_obs_dict_beta = {'obs':torch.cat((new_obs_tensor_am_beta,new_obs_tensor_beta),dim=0).unsqueeze(0)}

        #Get the actions from the loaded policy
        acts_beta = model_beta(new_obs_dict_beta, [torch.tensor([0.0], dtype=torch.float32)], torch.tensor([0], dtype=torch.int64))      
    
        #Format the acts_model as a gym spaces touple which is the same as the action space tuple defined in 
        # pettingzoo_env.py as self.per_player = spaces.Tuple(tuple([self.per_token for _ in range(self.n_tokens_per_player)])) 
        # where self.per_token = spaces.Discrete(38) and self.n_tokens_per_player = 11
        #TODO: Get rid of hard coded values 11 and 38. 
        acts_beta_reshaped = acts_beta[0].reshape(11,38)
        acts_beta_list = torch.argmax(acts_beta_reshaped, dim=1).tolist()
        acts_beta_tuple = tuple(acts_beta_list)

        #Decode the actions from the model into the action dicts that can be used by koth
        actions_beta_dict = penv.decode_discrete_player_action(U.P2,acts_beta_tuple)

        #Get the actions from the player
        acts_received = False
        with concurrent.futures.ThreadPoolExecutor() as executor:
            t = executor.submit(koth.KOTHGame.get_input_actions, penv.kothgame, U.P1)
            while not acts_received:
                if t.done():
                    acts_received = True
                    actions_alpha_dict = t.result()
                sleep(1)
                penv.render(mode="human")

        actions = {}
        actions.update(actions_alpha_dict)
        actions.update(actions_beta_dict)
        penv.actions = actions #Add actions to the penv sot that they can be rendered
        
        #koth.print_actions(actions)
        koth.log_game_to_file(penv.kothgame, logfile=logfile, actions=actions)

        # Update rendered pygame window
        penv.render(mode="human")

        # encode actions into flat gym space
        encoded_actions = penv.encode_all_discrete_actions(actions=actions)

        # apply encoded actions
        obs, rewards, dones, info = penv.step(actions=encoded_actions)

        # assert zero-sum game
        assert np.isclose(rewards[U.P1], -rewards[U.P2])

        #If game_sate is "MOVEMENT" Then print the engagement outcomes from the prior ENGAGEMENT phase
        if penv.kothgame.game_state[U.TURN_PHASE] == U.MOVEMENT and penv.kothgame.game_state[U.TURN_COUNT] > 0:
            koth.print_engagement_outcomes(penv.kothgame.engagement_outcomes)
            engagement_outcomes_dict = get_engagement_dict_from_list(penv.kothgame.engagement_outcomes)
            penv.actions = engagement_outcomes_dict #Add actions to the penv sot that they can be rendered
            penv._eg_outcomes_phase = True
            penv.kothgame.game_state[U.TURN_PHASE] = U.ENGAGEMENT
            penv.render(mode="human")
            sleep(5)
            penv._eg_outcomes_phase = False
            penv.kothgame.game_state[U.TURN_PHASE] = U.MOVEMENT

        # assert rewards only from final timestep
        if any([dones[d] for d in dones.keys()]):
            assert np.isclose(rewards[U.P1], 
                penv.kothgame.game_state[U.P1][U.SCORE] - penv.kothgame.game_state[U.P2][U.SCORE])
            break
        else:
            assert np.isclose(rewards[U.P1], 0.0)

    winner = None
    alpha_score = penv.kothgame.game_state[U.P1][U.SCORE]
    beta_score = penv.kothgame.game_state[U.P2][U.SCORE]
    if alpha_score > beta_score:
        winner = U.P1
    elif beta_score > alpha_score:
        winner = U.P2
    else:
        winner = 'draw'

    #Print final engagement outcomes
    koth.print_engagement_outcomes(penv.kothgame.engagement_outcomes)
    koth.log_game_to_file(penv.kothgame, logfile=logfile, actions=actions)
    log_game_final_to_csv(case_num, GAME_PARAMS,penv.kothgame, CSV_FILE_PATH, game_type, p1_alias=U.P1+":"+alias, p2_alias=U.P2+":AI", associated_logfile=logfile)

    cur_game_state = penv.kothgame.game_state
    if cur_game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel <= GAME_PARAMS.min_fuel:
        print(U.P1+" seeker out of fuel")
    if cur_game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel <= GAME_PARAMS.min_fuel:
        print(U.P2+" seeker out of fuel")
    if cur_game_state[U.P1][U.SCORE] >= GAME_PARAMS.win_score[U.P1]:
        print(U.P1+" reached Win Score")
    if cur_game_state[U.P2][U.SCORE]  >= GAME_PARAMS.win_score[U.P2]:
        print(U.P2+" reached Win Score")
    if cur_game_state[U.TURN_COUNT]  >= GAME_PARAMS.max_turns:
        print("max turns reached")
        
    print("\n====GAME FINISHED====\nWinner: {}\nScore: {}|{}\n=====================\n".format(winner, alpha_score, beta_score))

    penv.draw_win(winner)
    
    #Ask user to press spacebar to end the game and close the pygame window
    select_valid = 0
    while not select_valid:
        selection = input("Press spacebar and then return to end game: ")
        if selection == " ":
            select_valid = 1
        else:
            print("Invalid selection. Please press spacebar")
    return

def run_server_client_game(gs_host_addr, GP, case_num):
    #This client will NOT create the game server, the script that generates the AI agent will do that.
    #TODO: eventaully create a seperate script to make the game server, and then run two scripts, one for each agent that connects
    #NOTE: This script creates the 'beta' client and the other script will create the 'alpha' client.
    game_type = "server_client"
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
        router_addr="tcp://"+gs_host_addr+":{}".format(ROUTER_PORT_NUM),
        pub_addr="tcp://"+gs_host_addr+":{}".format(PUB_PORT_NUM),
        plr_alias=alias
    )

    # register clients as players in order, with random time between the two
    print("Registering client with alias {}...".format(plr_client.alias))
    plr_client.register_player_req()
    sleep(0.5)

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
    penv = PZE.parallel_env(game_params=GAME_PARAMS, training_randomize=False)
    
    # Start the rendered pygame window
    penv.render(mode="human")

    print("Player Alias: {}".format(plr_client.alias))
    print("Player ID: {}".format(plr_client.player_id))
    print("Player UUID: {}".format(plr_client.player_uuid))


    local_game = koth.KOTHGame(**GAME_PARAMS._asdict()) 

    logfile = koth.start_log_file('./logs/game_log_server_client')
    csv_logfile = './logs/logfile.csv'

    while not cur_game_state[GS.GAME_DONE]:

        print("\n<==== Turn: {} | Phase: {} ====>".format(cur_game_state[GS.TURN_NUMBER], cur_game_state[GS.TURN_PHASE]))

        turnphase = cur_game_state[GS.TURN_PHASE]
        
        if cur_game_state[GS.TURN_PHASE] == U.DRIFT:
            #send drift phase action request from penv client
            plr_client.send_action_req(context=cur_game_state[GS.TURN_PHASE], actions=[])

        else: #Game state is not DRIFT. Need to get new actions and send to server
            #update the local_game with the new game state from the server
            local_game.game_state, local_game.token_catalog, local_game.n_tokens_alpha, local_game.n_tokens_beta = GS.arbitrary_game_state_from_server(GP,cur_game_state)

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

        #if cur_game_state[GS.TURN_PHASE] == U.MOVEMENT and cur_game_state[GS.TURN_NUMBER] == 0:
        #    #Log the initial movement phase actions
        #    koth.log_game_to_file(local_game, logfile=logfile, actions=actions_dict)

        # wait for game state to advance
        while cur_game_state[GS.TURN_PHASE] == turnphase and not cur_game_state[GS.GAME_DONE]:
            sleep(1)
            print('waiting for turn phase {} to advance'.format(cur_game_state[GS.TURN_PHASE]))
            cur_game_state = plr_client.game_state

        if cur_game_state[GS.GAME_DONE] is True:
            local_game.game_state, local_game.token_catalog, local_game.n_tokens_alpha, local_game.n_tokens_beta = GS.arbitrary_game_state_from_server(GP,cur_game_state)
            koth.print_endgame_status(local_game)
            koth.log_game_to_file(local_game, logfile)
            if plr_client.player_id == U.P1:
                p1_alias = U.P1+":"+plr_client.alias
                p2_alias = U.P2
            elif plr_client.player_id == U.P2:
                p1_alias = U.P1
                p2_alias = U.P2+":"+plr_client.alias
            else:
                p1_alias = U.P1
                p2_alias = U.P2
            log_game_final_to_csv(case_num, GAME_PARAMS, local_game, CSV_FILE_PATH, game_type, p1_alias, p2_alias, associated_logfile=logfile)
            break

        #update the local_game with the new game state from the server and update the render
        if plr_client.engagement_outcomes is not None:
            local_game.engagement_outcomes = local_game.arbitrary_engagement_outcomes_from_server(plr_client.engagement_outcomes)[0]
            koth.print_engagement_outcomes(local_game.engagement_outcomes)
            penv.actions = local_game.arbitrary_engagement_outcomes_from_server(plr_client.engagement_outcomes)[1]
            plr_client.engagement_outcomes = None
        penv.kothgame = local_game
        if actions_dict is not None:
            koth.print_actions(actions_dict)
            koth.log_game_to_file(local_game, logfile=logfile, actions=actions_dict)
            actions_dict = None
        else:
            koth.log_game_to_file(local_game, logfile=logfile)
        penv.render(mode="human")

        local_game.game_state, local_game.token_catalog, local_game.n_tokens_alpha, local_game.n_tokens_beta = GS.arbitrary_game_state_from_server(GP,cur_game_state)
        
        #check if plr_client has attribute engagement_outcomes
        #if plr_client.engagement_outcomes is not None:
        #    with open(logfile, 'a') as f:
        #        print_engagement_outcomes_list(plr_client.engagement_outcomes, file=f)
        #    plr_client.engagement_outcomes = None

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

    if cur_game_state[GS.TOKEN_STATES][0]['fuel'] <= GAME_PARAMS.min_fuel:
        term_cond = "alpha out of fuel"
    elif cur_game_state[GS.TOKEN_STATES][1]['fuel'] <= GAME_PARAMS.min_fuel:
        term_cond = "beta out of fuel"
    elif cur_game_state[GS.SCORE_ALPHA] >= GAME_PARAMS.win_score[U.P1]:
        term_cond = "alpha reached Win Score"
    elif cur_game_state[GS.SCORE_BETA]  >= GAME_PARAMS.win_score[U.P2]:
        term_cond = "beta reached Win Score"
    elif cur_game_state[GS.TURN_NUMBER]  >= GAME_PARAMS.max_turns:
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
