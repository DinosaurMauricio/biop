import numpy as np
import gymnasium as gym
import pickle
import cv2 as cv
import os

from deap import base, creator
from utils import model_weights_as_matrix, model_build

def test(game):
    award = 0
    path = os.path.dirname(os.path.abspath(__file__))
    env = gym.make(game, render_mode='human')
    env = env.env
    obs, _ = env.reset()

    in_dimen = (env.observation_space.shape[0], env.observation_space.shape[1], 1)
    out_dimen = env.action_space.n

    model = model_build(in_dimen, out_dimen)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    game_name = game.split('/')[1].split('-')[0].lower()
    
    with open (f"{path}\winners\winner-{game_name}.pkl", 'rb') as file:
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
            
            action = np.argmax(action_probs)

            obs, reward, done, _, _ = env.step(action)

            obs = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
            award += reward

    print(award)
    env.close()
