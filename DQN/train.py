import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env



NUMBER_OF_TIMESTEPS = 300000
def train(game):

    env_vec = make_atari_env(game, n_envs=4, seed=0, wrapper_kwargs={"screen_size": 42})
    env_vec = VecFrameStack(env_vec, n_stack=4)

    for env in env_vec.envs:
        env.metadata['render_fps'] = 30

    model = DQN("CnnPolicy", env_vec, verbose=1)
    model.learn(total_timesteps=NUMBER_OF_TIMESTEPS, log_interval=4)

    game_name = game.split('/')[1].split('-')[0].lower()
    path = os.path.dirname(__file__) 
    
    model.save(f'{path}\winners\dqn_{game_name}', 'wb')
