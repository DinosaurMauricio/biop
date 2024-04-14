import gymnasium as gym
import neat
import cv2
import numpy as np
import pickle
import visualize
import os
import time


MAX_DURATION = 600  # We assume it takes 600 time steps per episode
NUMBER_OF_GENERATIONS = 50

def train(game_name):
    env = gym.make(game_name, render_mode="rgb_array")
    env.metadata['render_fps'] = 30

    path = os.path.dirname(__file__)
    game = game_name.split('/')[1].split('-')[0].lower()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     f'{path}\configs\config_{game}.txt')
    
    p = neat.Population(config)
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)

    winner = p.run(lambda genomes, config: eval_genomes(env, genomes, config), NUMBER_OF_GENERATIONS)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


    with open(f'{path}\winners\winner-{game}', 'wb') as f:
        pickle.dump(winner, f, 1)

def eval_genomes(env, genomes, config):
    imgarrays = []
    for genome_id, genome in genomes:
        ob, info = env.reset()
        inx, iny, _ = env.observation_space.shape
        inx = int(inx/8)
        iny = int(iny/8)
        

        net = neat.nn.RecurrentNetwork.create(genome, config)

        fitness_current = 0
        done = False
        # Record the start time
        start_time = time.time()

        current_lives = info['lives']

        while not done:

            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)

            ob = cv2.GaussianBlur(ob, (3, 3), 0)
            noise = np.zeros(ob.shape, np.uint8)
            cv2.randn(noise, 0, 1) 

            # Add noise to the image
            ob = cv2.add(ob, noise)

            imgarrays = np.ndarray.flatten(ob)
            net_output = net.activate(imgarrays)

            numerical_input = net_output.index(max(net_output))

            ob, reward, done, truncated, info = env.step(numerical_input)

            if truncated:
                break

            fitness_current += reward
            elapsed_time = time.time() - start_time
            survival_reward = elapsed_time / MAX_DURATION  # Normalize reward by max duration
            fitness_current += survival_reward

            if info['lives'] < current_lives:
                current_lives = info['lives']
                fitness_current -= 5
                start_time = time.time()

            if done:
                done = True
                print(genome_id, fitness_current)
            genome.fitness = fitness_current
