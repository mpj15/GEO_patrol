# this is meant as a tool for running a complete koth game
# where policies are based on one or two saved policies from checkpoints

import numpy as np
import orbit_defender2d.utils.utils as U
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
import orbit_defender2d.king_of_the_hill.default_game_parameters as DGP
#import orbit_defender2d.king_of_the_hill.default_game_parameters as DGP
#import orbit_defender2d.king_of_the_hill.default_game_parameters_small as DGP
from orbit_defender2d.king_of_the_hill import koth
import torch
import orbit_defender2d.king_of_the_hill.pettingzoo_env as PZE
from time import sleep
import threading
import concurrent.futures



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


def get_engagement_dict_from_list(engagement_list):
    """
    Turns a list of engagement tuples or engagement outcome tuples into a list of dicts with the key as the token name and the tuple as the value
    """
    engagement_dict = {}
    for eng in engagement_list:
        engagement_dict[eng.attacker] = eng
    return engagement_dict

def run_game(model_path_beta):

    # create and reset pettingzoo env
    penv = PZE.parallel_env(game_params=GAME_PARAMS, training_randomize=False)
    obs = penv.reset()
    # Update rendered pygame window
    #penv.render(mode="debug")
    
    # Render pygame window
    penv.render(mode="human")
    penv.screen_shot(file_name="./od2d_screen_shot_new.png")
    
    model_beta = torch.jit.load(model_path_beta)
    model_beta.eval()

    #start logfile
    logfile = koth.start_log_file('./logs/game_log')

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
        acts_beta_split = torch.split(acts_beta_reshaped, 1, dim=0)
       

        acts_beta_list = []

        for i in range(11):

            acts_beta_list.append((torch.argmax(acts_beta_split[i]).item()))

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

    cur_game_state = penv.kothgame.game_state
    if cur_game_state[U.P1][U.TOKEN_STATES][0].satellite.fuel <= DGP.MIN_FUEL:
        print(U.P1+" seeker out of fuel")
    if cur_game_state[U.P2][U.TOKEN_STATES][0].satellite.fuel <= DGP.MIN_FUEL:
        print(U.P2+" seeker out of fuel")
    if cur_game_state[U.P1][U.SCORE] >= DGP.WIN_SCORE[U.P1]:
        print(U.P1+" reached Win Score")
    if cur_game_state[U.P2][U.SCORE]  >= DGP.WIN_SCORE[U.P2]:
        print(U.P2+" reached Win Score")
    if cur_game_state[U.TURN_COUNT]  >= DGP.MAX_TURNS:
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




if __name__ == "__main__":
    #run_pettingzoo_random_game()

    
    #model_path_beta = "./policies/model_offense.pt" #This was trained on 1200 iterations in RLlib
    #model_path_beta = "./policies/model_2400_08March_smallBoard.pt" #2400 iterations of board without outer ring. Trained on randagm init game params.

    model_path_beta = "./policies/model_3800_smallBoard_15March.pt" #3800 iterations of board without outer ring. Trained on randagm init game params.

    run_game(model_path_beta)