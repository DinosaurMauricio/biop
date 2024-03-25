import gymnasium as gym
import neat
import cv2
import numpy as np
import pickle
import visualize
import os

env = gym.make('ALE/SpaceInvaders-v5', render_mode="rgb_array")
imgarrays = []
env.metadata['render_fps'] = 30
path = os.path.dirname(__file__)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        ob, _ = env.reset()
        inx, iny, _ = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.FeedForwardNetwork.create(genome, config)

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

            #if reward > 0:
            #    early_stop_counter = 0
            #else:
            #    early_stop_counter += 1

            if done:# or early_stop_counter > 600:
                done = True
                print(genome_id, fitness_current)
            genome.fitness = fitness_current


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     path + '\config_invaders.txt')

p = neat.Population(config)
stats = neat.StatisticsReporter()
p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(stats)

NUMBER_OF_GENERATIONS = 25

winner = p.run(eval_genomes, NUMBER_OF_GENERATIONS)
visualize.plot_stats(stats, ylog=False, view=True)
visualize.plot_species(stats, view=True)


with open('winner-invaders', 'wb') as f:
    pickle.dump(winner, f, 1)