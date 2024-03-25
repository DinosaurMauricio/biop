from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import gymnasium as gym
from deap import base, creator, tools, algorithms
import pickle
import cv2 as cv

from utils import model_weights_as_matrix, model_build

# get path
import os
path = os.path.dirname(os.path.abspath(__file__))

award = 0
env = gym.make("ALE/SpaceInvaders-v5", render_mode='human')
env = env.env
env.reset()
obs, _ = env.reset()
in_dimen = env.observation_space.shape
in_dimen = (env.observation_space.shape[0], env.observation_space.shape[1], 1)
out_dimen = env.action_space.n


# print('the in dimension is ', in_dimen)
model = model_build(in_dimen, out_dimen)
ind_size = model.count_params()


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
#'C:\Users\link5\Documents\SideProjects\AI\BioInspired-Project\atari_invaders_model_2.pkl'
with open ("C:\\Users\\link5\\Documents\\SideProjects\\AI\\BioInspired-Project\\invaders_deap\\atari_invaders_model_2.pkl", 'rb') as file:
#with open(path + "/atari_invaders_model_2.pkl", 'rb') as file:
    best = pickle.load(file)


best_weight = model_weights_as_matrix(model, best)

model.set_weights(best_weight)


done = False
# Convert observation to grayscale
obs = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
# Normalize pixel values to range [0, 1]
obs = obs / 255.0
while done == False:
    done = False
    while not done:
        obs = np.expand_dims(obs, axis=0)
        action_probs = model.predict(obs)
        #print(action_probs)
        action = np.argmax(action_probs)

        obs, reward, done, _, info = env.step(action)

        # Convert observation to grayscale
        obs = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
        # Normalize pixel values to range [0, 1]
        obs = obs / 255.0
        award += reward

print(award)
env.close()
