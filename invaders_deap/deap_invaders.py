
import numpy as np
import gymnasium as gym
import random
from deap import base, creator, tools, algorithms
import pickle
import multiprocessing

from utils import model_build, evaluate
import time

# Record the start time
start_time = time.time()


env = gym.make("ALE/SpaceInvaders-v5")
env.reset()
in_dimen = env.observation_space.shape
out_dimen = env.action_space.n

# The rest of your code remains unchanged
model = model_build(in_dimen, out_dimen)
ind_size = model.count_params()
#print(out_dimen)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("weight_bin", random.random)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.weight_bin, n=ind_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.7)
toolbox.register("select", tools.selBest)
toolbox.register("evaluate", evaluate, env=env)

if __name__ == "__main__":
    random.seed(64)

    # Process Pool
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU count: {cpu_count}")
    pool = multiprocessing.Pool(cpu_count)
    toolbox.register("map", pool.map)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Mean", np.mean)
    stats.register("Max", np.max)
    stats.register("Min", np.min)

    pop = toolbox.population(n=20)  # n =20
    hof = tools.HallOfFame(2)

    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=0.7, mutpb=0.4, ngen=5,  halloffame=hof, stats=stats)
    best_pop = sorted(pop, key=lambda ind: ind.fitness,
                      reverse=True)[0]  # ngen =5
    pool.close()

    with open("atari_invaders_model_2.pkl", "wb") as cp_file:
        pickle.dump(best_pop, cp_file)

    # Record the end time
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
