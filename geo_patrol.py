import time
import server_client_for_CLI as sc
import geo_patrol_games as gpg

GS_HOST_ADDR = "192.168.1.11"

MODEL_PATH_ALPHA = "./policies/model_O3_6500.pt"
MODEL_PATH_BETA = "./policies/model_D4_9600.pt"

def start_screen():
    #First, clear the screen
    print("\033[H\033[J")
    print("Welcome to GEO Patrol!")
    print("Please choose which game mode you would like to play:")
    print("1. AI vs AI")
    print("2. AI vs Player")
    print("3. Player vs Player")
    print("4. Exit")

def get_case_num():
    print("Please choose which case you would like to play:")
    print("0. Default Symmetric Case")
    print("1. Case 1")
    print("2. Case 2")
    print("3. Case 3")
    case_num = int(input("Enter your choice: "))
    #check if the input is valid
    while case_num < 0 or case_num > 3:
        print("Invalid choice. Please try again.")
        case_num = int(input("Enter your choice: "))
    return case_num

def run():
    while True:
        start_screen()
        gameMode = int(input("Enter your choice: "))
        #If user chooses 1, run AIvsAI.py
        if gameMode == 1:
            #run run_AI_vs_AI as a demo
            case_num = get_case_num()
            model_path_alpha = MODEL_PATH_ALPHA
            model_path_beta =  MODEL_PATH_BETA
            gpg.ai_v_ai_game_mode(model_path_alpha, model_path_beta, case_num=case_num)
            print("Game finished.")
            time.sleep(1)
        elif gameMode == 2:
            #run human vs AI game
            #For now, just run the AI as offense and human as defense
            model_path_alpha = MODEL_PATH_ALPHA
            case_num = get_case_num()
            if case_num == 1:
                # Run case 1
                gpg.human_v_ai_game_mode(model_path_alpha= model_path_alpha,model_path_beta=None, case_num=1)
                print("Game finished.")
                time.sleep(1)
            elif case_num == 2:
                # Run case 2
                gpg.human_v_ai_game_mode(model_path_alpha= model_path_alpha,model_path_beta=None, case_num=2)
                print("Game finished.")
                time.sleep(1)
            elif case_num == 3:
                # Run case 3
                gpg.human_v_ai_game_mode(model_path_alpha= model_path_alpha,model_path_beta=None, case_num=3)
                print("Game finished.")
                time.sleep(1)
            else:
                print("Invalid choice. Please try again.")
                time.sleep(1)
        elif gameMode == 3:
            # Run human vs human game from a remote machine.
            # Note this is for running a client to connect. 
            # You have to have a preconfigured remote machine
            # running the server (two_player_game_server.py). 
            # And another player also running this client.
            case_num = get_case_num()
            gpg.run_server_client_game_mode(GS_HOST_ADDR,case_num=case_num)
            print("Game finished.")
            time.sleep(1)
        elif gameMode == 4:
            #Exit
            print("Exiting game.")
            time.sleep(1)
            break
        else:
            print("Invalid choice. Please choose 1, 2, 3, or 4.")
            time.sleep(1)
    return

if __name__ == "__main__":
    run()   