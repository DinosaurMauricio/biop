import gymnasium as gym
import neat
import cv2
import numpy as np
import pickle
import visualize
import os
import multiprocessing

env = gym.make('ALE/SpaceInvaders-v5', render_mode="rgb_array")
imgarrays = []
env.metadata['render_fps'] = 30
path = os.path.dirname(__file__)


NUMBER_OF_GENERATIONS = 20

def eval_genomes(genome, config):
    #for genome_id, genome in genomes:
    ob, _ = env.reset()
    inx, iny, _ = env.observation_space.shape
    inx = int(inx/8)
    iny = int(iny/8)
    net = neat.nn.RecurrentNetwork.create(genome, config)
    fitness_current = 0
    early_stop_counter = 0
    done = False
    while not done:
        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        imgarrays = np.ndarray.flatten(ob)
        net_output = net.activate(imgarrays)
        numerical_input = net_output.index(max(net_output))
        ob, reward, done, truncated, _ = env.step(numerical_input)
        if truncated:
            break
        fitness_current += reward
        if reward > 0:
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if done or early_stop_counter > 600:
            done = True
            print(fitness_current)
        genome.fitness = fitness_current


#winner = p.run(eval_genomes, NUMBER_OF_GENERATIONS)
if __name__ == '__main__':
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     path + '\config_invaders.txt')

    p = neat.Population(config)
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genomes)
    winner = p.run(pe.evaluate, NUMBER_OF_GENERATIONS)


    with open("Winner.pkl", "wb") as f:
        pickle.dump(winner, f)
        f.close()

    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)