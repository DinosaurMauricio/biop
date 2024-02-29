
import gymnasium as gym
import keyboard

# Create the Pong environment
env = gym.make('Pong-v4',  render_mode="human")

# Set render fps in the environment metadata
env.metadata['render_fps'] = 30

# Reset the environment to its initial state
observation = env.reset()

# Play 1000 episodes
for _ in range(1000):
    # Render the environment
    env.render()

    # Randomly choose an action (0 or 1, which correspond to UP and DOWN in Pong)
    # action = env.action_space.sample()

    # Get keyboard input
    if keyboard.is_pressed('up'):
        action = 2  # Move the paddle up (action 2 in Pong)
    elif keyboard.is_pressed('down'):
        action = 3  # Move the paddle down (action 3 in Pong)
    else:
        action = 0  # Do nothing

    # Take a step in the environment by applying the chosen action
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode is done (game over), reset the environment
    if terminated:
        observation = env.reset()

# Close the environment
env.close()
