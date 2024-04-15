import argparse
from train import train
from test import test


game_setup = {
    1: "ALE/SpaceInvaders-v5",
    2: "ALE/MsPacman-v5",
    3: "ALE/KungFuMaster-v5",
}

available_mode = {
    1: "rgb_array",
    2: "human"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser("deap_project")
    # Default to train for SpaceInvaders
    parser.add_argument("--gameId", help="Game you want to train", type=int, default=1)
    parser.add_argument("--mode", help="1. Train 2. Test model", type=int, default=1)
    args = parser.parse_args()

    if args.gameId not in game_setup:
        print("Invalid game ID provided.")
        exit(1)
    elif args.mode not in available_mode:
        print("Invalid mode provided.")
        exit(1)
    else:
        game_name = game_setup.get(args.gameId)
        mode = available_mode.get(args.mode)

    if mode == 'rgb_array':
        train(game_name)
    elif mode == 'human': 
        test(game_name)
    else :
        print("Invalid mode provided.")
        exit(1)