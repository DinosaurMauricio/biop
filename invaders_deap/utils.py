import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import numpy as np
from tensorflow.keras import Input
import cv2 as cv


#def model_build(in_dimen, out_dimen):
#    model = Sequential()
#    model.add(Input(shape=in_dimen))
#    model.add(Conv2D(16, (8, 8), strides=(4, 4),
#              activation='relu'))
#    model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))
#    model.add(Flatten())
#    model.add(Dense(256, activation='relu'))
#    model.add(Dense(out_dimen, activation='softmax'))
#    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#    return model

def model_build(in_dimen, out_dimen):
    model = Sequential()
    model.add(Input(shape=in_dimen))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(out_dimen, activation='softmax'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def model_weights_as_matrix(model, weights_vector):
    weights_matrix = []

    start = 0

    for layer_idx, layer in enumerate(model.layers):
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

    obs, _ = env.reset()

    # Convert observation to grayscale
    obs = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
    # Normalize pixel values to range [0, 1]
    obs = obs / 255.0

    model = model_build(in_dimen, out_dimen)
    model.set_weights(model_weights_as_matrix(model, individual))

    total_reward = 0
    done = False
    while not done:
        obs = np.expand_dims(obs, axis=0)
        action_probs = model.predict(obs)
        action = np.argmax(action_probs)
        print('the action is:', action)
        print('the action probs is:', action_probs)
        #action = 0 if action < 0.5 else 1


        #if action != 0 or action != 1:
        #    print(action)
        #    break
        #if action == 0:
        #    action = 5
        #elif action == 1:
        #    action = 4

        obs, reward, done, _, info = env.step(action)

        # Convert observation to grayscale
        obs = cv.cvtColor(obs, cv.COLOR_RGB2GRAY)
        # Normalize pixel values to range [0, 1]
        obs = obs / 255.0
        
        total_reward += reward

    return total_reward,
