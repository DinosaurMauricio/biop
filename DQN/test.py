from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env

import os


def test(game):
    
    path = os.path.dirname(os.path.abspath(__file__))
    
    game_name = game.split('/')[1].split('-')[0].lower()

    model_path = os.path.join(f'{path}/winners/dqn_{game_name}')

    model = DQN.load(model_path)

    vec_env = make_atari_env(game, n_envs=1, seed=0, env_kwargs={"render_mode": "human"},wrapper_kwargs={"screen_size": 42})

    for env in vec_env.envs:
        env.metadata['render_fps'] = 30

    vec_env = VecFrameStack(vec_env, n_stack=4)

    evaluate_policy(model, vec_env, n_eval_episodes=5, render=True)
