
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import numpy as np
import cv2
import time

MAX_DURATION = 600  # Assuming 600 time steps per episode

def model_build(in_dimen, out_dimen):
    model = Sequential()
    model.add(Input(shape=in_dimen))
    model.add(Conv2D(16, (8, 8), strides=(4, 4),
              activation='relu'))
    model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(out_dimen, activation='softmax'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def model_weights_as_matrix(model, weights_vector):
    weights_matrix = []
    start = 0

    for _, layer in enumerate(model.layers):
        layer_weights = layer.get_weights()
        if layer.trainable:
            for l_weights in layer_weights:

                layer_weights_shape = l_weights.shape
                layer_weights_size = l_weights.size

                layer_weights_vector = weights_vector[start:start +
                                                      layer_weights_size]

                layer_weights_matrix = np.reshape(
                    layer_weights_vector, newshape=(layer_weights_shape))
                weights_matrix.append(layer_weights_matrix)

                start = start + layer_weights_size
        else:
            for l_weights in layer_weights:
                weights_matrix.append(l_weights)
    return weights_matrix


def evaluate(individual, env):
    # this has three dimensions
    in_dimen = env.observation_space.shape
    in_dimen = (env.observation_space.shape[0], env.observation_space.shape[1], 1)
    out_dimen = env.action_space.n

    obs, info = env.reset()

    # Convert observation to grayscale
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.GaussianBlur(obs, (3, 3), 0)

    noise = np.zeros(obs.shape, np.uint8)
    cv2.randn(noise, 0, 1) 
    obs = cv2.add(obs, noise)

    model = model_build(in_dimen, out_dimen)
    model.set_weights(model_weights_as_matrix(model, individual))
    start_time = time.time()  # Record the start time
    total_reward = 0
    current_lives = info['lives']
    done = False

    while not done:
        obs = np.expand_dims(obs, axis=0)
        action_probs = model.predict(obs)
        action = np.argmax(action_probs)

        obs, reward, done, _, info = env.step(action)

        # Convert observation to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.GaussianBlur(obs, (3, 3), 0)

        noise = np.zeros(obs.shape, np.uint8)
        cv2.randn(noise, 0, 1) 
        obs = cv2.add(obs, noise)

        elapsed_time = time.time() - start_time
        survival_reward = elapsed_time / MAX_DURATION  # Normalize reward by max duration
        total_reward += survival_reward

        if info['lives'] < current_lives:
            current_lives = info['lives']
            total_reward -= 5
            start_time = time.time()
        
        total_reward += reward

    return total_reward,