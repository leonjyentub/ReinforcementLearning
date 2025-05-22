import gymnasium as gym

env = gym.make("CarRacing-v3", render_mode="human", lap_complete_percent=0.95, domain_randomize=False)

observation, info = env.reset(seed=42)
for epoch in range(10000):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
    print(f"Epoch: {epoch}, Action: {action}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
env.close()
# The code above is a simple example of how to use the OpenAI Gym library to create a continuous action space environment for the Mountain Car problem.
# The Mountain Car problem is a classic reinforcement learning problem where the goal is to get a car up a hill by accelerating and decelerating.
# The car starts in a valley and must build up enough speed to reach the top of the hill.
# The code uses the `gymnasium` library to create the environment and simulate the car's movement.
# The `render_mode` parameter is set to "human" to visualize the car's movement in a window.
# The `goal_velocity` parameter is set to 0.1 to specify the desired velocity of the car at the top of the hill.
# The code runs for 100 epochs, where in each epoch, the car's action is sampled randomly from the action space.
# The car's observation, reward, termination status, and truncation status are printed for each epoch.
# The `terminated` flag indicates whether the episode has ended, and the `truncated` flag indicates whether the episode was truncated due to reaching a maximum number of steps.
# The `env.close()` method is called at the end to close the environment and release any resources used by it.