
import numpy as np
import gymnasium as gym
import random
import pickle
import multiprocessing
import time
import os
from deap import base, creator, tools, algorithms
from utils import model_build, evaluate


INIITIAL_WEIGTH_RANGE = (-1.0, 1.0)
POPULATION = 2
NGEN = 1

def initial_weight():
    return np.random.uniform(INIITIAL_WEIGTH_RANGE[0], INIITIAL_WEIGTH_RANGE[1])

def setup_toolbox(number_of_params, env):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("weight_bin", random.random)

    toolbox.register("individual", tools.initRepeat,
                    creator.Individual, initial_weight, n=number_of_params)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", evaluate, env=env)

    return toolbox

def setup_statistics():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Mean", np.mean)
    stats.register("Max", np.max)
    stats.register("Min", np.min)

    return stats


def train(game):
    # Record the start time
    start_time = time.time()
    env = gym.make(game, render_mode='rgb_array')
    env.reset()
    in_dimen = env.observation_space.shape
    out_dimen = env.action_space.n


    number_of_params = model.count_params()
    model = model_build(in_dimen, out_dimen)
    toolbox = setup_toolbox(number_of_params, env)
    
    random.seed(64)

    # Process Pool
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU count: {cpu_count}")
    pool = multiprocessing.Pool(cpu_count)
    toolbox.register("map", pool.map)

    stats = setup_statistics()


    pop = toolbox.population(n=POPULATION) 
    hof = tools.HallOfFame(2)

    pop, _ = algorithms.eaSimple(
        pop, toolbox, cxpb=0.85, mutpb=0.01, ngen=NGEN,  halloffame=hof, stats=stats)
    best_pop = sorted(pop, key=lambda ind: ind.fitness,
                      reverse=True)[0]  #
    pool.close()

    game_name = game.split('/')[1].split('-')[0].lower()

    path = os.path.dirname(__file__)

    os.makedirs(f'{path}/winners', exist_ok=True)
    with open(f"{path}\winners\winner-{game_name}.pkl", "wb") as cp_file:
        pickle.dump(best_pop, cp_file)

    # Record the end time
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
