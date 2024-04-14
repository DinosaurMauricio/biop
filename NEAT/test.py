import os
import pickle
import neat
import gymnasium as gym
import numpy as np
import cv2



def test(game_name):



    # read current path folder
    path = os.path.dirname(__file__)


    game = game_name.split('/')[1].split('-')[0].lower()
    with open(f'{path}\winners\winner-{game}', 'rb') as f:
        print(f)
        c = pickle.load(f)

    print('Loaded genome:')
    print(c)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, f'configs\config_{game}.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)

    net = neat.nn.RecurrentNetwork.create(c, config)

    env = gym.make(game_name, render_mode="human")
    env.metadata['render_fps'] = 30
    observation, _ = env.reset()
    inx, iny, _ = env.observation_space.shape
    inx = int(inx/8)
    iny = int(iny/8)
    done = False
    while not done:
        observation = cv2.resize(observation, (inx, iny))
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)

        imgarrays = np.ndarray.flatten(observation)
        action = np.argmax(net.activate(imgarrays))
        observation, reward, done, truncated, info = env.step(action)
        env.render()
