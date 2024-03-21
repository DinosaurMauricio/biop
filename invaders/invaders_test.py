import os
import pickle
import neat
import gymnasium as gym
import numpy as np
import cv2


# read current path folder
path = os.path.dirname(__file__)


with open(path + '\winner-invaders_best', 'rb') as f:
    print(f)
    c = pickle.load(f)

print('Loaded genome:')
print(c)


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config_invaders.txt')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.RecurrentNetwork.create(c, config)

env = gym.make('ALE/SpaceInvaders-v5', render_mode="human")
env.metadata['render_fps'] = 30
observation, info_env = env.reset()
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
