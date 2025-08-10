import gymnasium as gym
import numpy as np

# Create CartPole environment
env = gym.make("CartPole-v1")

print("=== CartPole-v1 Environment Info ===")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Observation shape: {env.observation_space.shape}")
print(f"Number of actions: {env.action_space.n}")

print("\n=== Reset States (5 examples) ===")
for i in range(5):
    obs, info = env.reset()
    print(f"Reset {i+1}: {obs}")
    print(f"  Cart Position: {obs[0]:.6f}")
    print(f"  Cart Velocity: {obs[1]:.6f}")
    print(f"  Pole Angle: {obs[2]:.6f}")
    print(f"  Pole Angular Velocity: {obs[3]:.6f}")
    print()

print("=== Observation Space Details ===")
print("Index 0: Cart Position - position of cart along the track")
print("Index 1: Cart Velocity - velocity of the cart")
print("Index 2: Pole Angle - angle of the pole (radians)")
print("Index 3: Pole Angular Velocity - angular velocity of the pole")

print("\n=== Action Space ===")
print("0: Push cart to the left")
print("1: Push cart to the right")

print("\n=== Initial State Range ===")
print("All observations initialized uniformly in range [-0.05, 0.05]")

env.close()
