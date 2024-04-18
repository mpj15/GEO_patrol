#import sys
#import threading
#import multiprocessing
import time
import torch_model_play as tmp
import torch_model_CLI as cli
import server_client_for_CLI as sc
import signal
import sys

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    run()

def start_screen():
    #First, clear the screen
    print("\033[H\033[J")
    print("Welcome to Orbit Defender 2d")
    print("Please choose which game mode you would like to play:")
    print("1. AI vs AI")
    print("2. AI vs Player")
    print("3. Player vs Player")
    print("4. Exit")

def run():
    while True:
        start_screen()
        gameMode = int(input("Enter your choice: "))
        #If user chooses 1, run AIvsAI.py
        if gameMode == 1:
            #run run_AI_vs_AI as a demo
            tmp.ai_v_ai()
            print("Game finished.")
            time.sleep(1)

        elif gameMode == 2:
            #run human vs AI game
            cli.human_v_ai()
            print("Game finished.")
            time.sleep(1)
        elif gameMode == 3:
            # Run human vs human game from a remote machine.
            # Note this is for running a client to connect. 
            # You have to have a preconfigured remote machine
            # running the server (two_player_game_server.py). 
            # And another player also running this client.
            sc.run_client()
            print("Game finished.")
            time.sleep(3)
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
    signal.signal(signal.SIGINT, signal_handler)
    run()   