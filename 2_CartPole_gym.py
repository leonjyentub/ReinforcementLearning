import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset(seed=42)
for epoch in range(100):
    env.render()
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
    print(f"Epoch: {epoch}, Action: {action}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
env.close()
