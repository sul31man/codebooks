#!/usr/bin/env python3
"""
Example integration of CUDA environment with SAC agent.
This shows how to use the environment with the existing SAC.py agent.
"""

import sys
import os
import torch
import numpy as np

# Add the parent directory to the path to import SAC
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment_wrapper import CUDAEnvironment

def test_sac_integration():
    """Test the integration between SAC agent and CUDA environment."""
    
    # Initialize the environment
    print("Initializing CUDA environment...")
    env = CUDAEnvironment(cols=25, rows=23, Ka=1, num_sims=100)
    
    # Reset environment
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Simulate SAC agent interaction
    print("\nSimulating SAC agent interaction...")
    
    total_reward = 0.0
    episode_rewards = []
    
    for episode in range(5):  # Run 5 episodes
        episode_reward = 0.0
        state = env.reset()
        done = False
        step_count = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while not done and step_count < 20:  # Max 20 steps per episode
            # Simulate SAC agent action (random for this example)
            # In real usage, this would be: action, log_prob = matrix_policy(state)
            action = torch.randn(25)  # 25-dimensional action from SAC agent
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            
            # Print progress
            if step_count % 5 == 0:
                print(f"  Step {step_count}: Reward={reward:.4f}, Hit Rate={info['hit_rate']:.4f}")
            
            state = next_state
        
        episode_rewards.append(episode_reward)
        total_reward += episode_reward
        
        print(f"  Episode {episode + 1} finished: Total reward={episode_reward:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Average episode reward: {np.mean(episode_rewards):.4f}")
    print(f"Best hit rate achieved: {env.best_hit_rate:.4f}")
    
    # Get final codebook
    codebook = env.get_codebook()
    print(f"Final codebook shape: {codebook.shape}")
    
    return env, episode_rewards

def integrate_with_sac_agent():
    """
    Example of how to integrate with the actual SAC agent.
    This function shows the interface that would be used in SAC.py
    """
    
    # Initialize environment (this would be in SAC.py)
    env = CUDAEnvironment(cols=25, rows=23, Ka=1, num_sims=1000)
    
    # Example of how the SAC training loop would look:
    num_episodes = 1000
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # This is where the SAC agent would generate an action
            # action, log_prob = matrix_policy(state)
            
            # For this example, use random action
            action = torch.randn(25)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience in replay buffer (in SAC.py)
            # buffer.push(state, action, next_state, reward, done)
            
            episode_reward += reward
            state = next_state
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward={episode_reward:.4f}, Best Hit Rate={env.best_hit_rate:.4f}")

if __name__ == "__main__":
    print("Testing CUDA Environment Integration with SAC Agent")
    print("=" * 50)
    
    try:
        # Test basic integration
        env, rewards = test_sac_integration()
        
        print("\n" + "=" * 50)
        print("Integration test completed successfully!")
        print("The environment is ready to be used with your SAC agent.")
        
        # Show how to use with actual SAC agent
        print("\nTo integrate with your SAC agent, replace the random actions")
        print("in the example above with actual SAC agent actions:")
        print("  action, log_prob = matrix_policy(state)")
        
    except Exception as e:
        print(f"Error during integration test: {e}")
        print("Please ensure CUDA is installed and the library can be compiled.")
        sys.exit(1) 