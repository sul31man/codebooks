#!/usr/bin/env python3
"""
Production-Ready SAC Training Script
GPU-accelerated codebook optimization with parallel CUDA environment.
"""

import torch
import numpy as np
import time
import argparse
from pathlib import Path

from environment_wrapper import CUDAEnvironment
from SAC import SACAgent, ReplayBuffer

def train_sac(args):
    """Main training loop for SAC with CUDA environment."""
    print("🚀 Starting Production SAC Training...")
    print(f"📊 Episodes: {args.episodes}")
    print(f"🎯 Ka values: {args.ka_values}")
    print(f"⚡ Device: {args.device}")
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize environment
    env = CUDAEnvironment(
        Ka_values=args.ka_values,
        num_sims=args.num_sims
    )
    
    # Initialize SAC agent
    agent = SACAgent(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        alpha=args.alpha,
        tau=args.tau,
        gamma=args.gamma,
        device=device
    )
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_capacity,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        device=device
    )
    
    # Training loop
    total_steps = 0
    best_reward = float('-inf')
    training_start = time.time()
    episode_rewards = []
    episode_times = []
    
    print("\n📈 Training Progress:")
    print("Episode | Steps | Reward    | Best      | Hit Rate | Time/Ep (s) | Throughput")
    print("-" * 75)
    
    for episode in range(args.episodes):
        episode_start = time.time()
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            # Select action
            action = agent.select_action(state, evaluate=False)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store in replay buffer
            replay_buffer.push(
                state,
                action.cpu().numpy() if isinstance(action, torch.Tensor) else action,
                reward,
                next_state,
                done
            )
            
            # Update agent
            if replay_buffer.size > args.batch_size:
                losses = agent.update(
                    replay_buffer,
                    batch_size=args.batch_size,
                    updates=args.updates_per_step
                )
            
            # Update statistics
            total_steps += 1
            steps += 1
            episode_reward += reward
            
            # Update state
            state = next_state
            
            # Check if episode is done
            if done:
                break
        
        # Episode statistics
        episode_time = time.time() - episode_start
        episode_rewards.append(episode_reward)
        episode_times.append(episode_time)
        
        # Update best reward
        best_reward = max(best_reward, episode_reward)
        
        # Calculate throughput (simulations per second)
        throughput = len(env.Ka_values) * env.num_sims / episode_time
        
        # Progress update
        print(f"{episode+1:7d} | {steps:5d} | {episode_reward:9.4f} | {best_reward:9.4f} | "
              f"{info['hit_rate']:8.4f} | {episode_time:11.2f} | {throughput:10.1f}")
        
        # Detailed info every N episodes
        if (episode + 1) % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            avg_time = np.mean(episode_times[-args.log_interval:])
            
            print(f"\n📊 Stats (Episodes {episode+1-args.log_interval+1}-{episode+1}):")
            print(f"   Average reward: {avg_reward:.4f}")
            print(f"   Average time: {avg_time:.2f}s")
            print(f"   Hit rates by Ka: {info['hit_rates']}")
            print(f"   Buffer size: {replay_buffer.size}")
            print("-" * 75)
        
        # Save checkpoint
        if (episode + 1) % args.save_interval == 0:
            save_path = Path(args.save_dir) / f"sac_checkpoint_ep{episode+1}.pt"
            save_path.parent.mkdir(exist_ok=True)
            agent.save(save_path)
            print(f"💾 Saved checkpoint: {save_path}")
    
    # Training complete
    total_time = time.time() - training_start
    
    print(f"\n🎉 Training completed!")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📊 Average time per episode: {total_time/args.episodes:.2f}s")
    print(f"🏆 Final best reward: {best_reward:.4f}")
    print(f"📈 Final hit rate: {info['hit_rate']:.4f}")
    
    # Save final model
    final_path = Path(args.save_dir) / "sac_final.pt"
    final_path.parent.mkdir(exist_ok=True)
    agent.save(final_path)
    
    # Save final codebook
    final_codebook = env.get_codebook()
    codebook_path = Path(args.save_dir) / "final_codebook.npy"
    np.save(codebook_path, final_codebook)
    
    print(f"💾 Saved final model: {final_path}")
    print(f"💾 Saved final codebook: {codebook_path}")
    
    return agent, env, episode_rewards

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="SAC Training for CUDA Codebook Optimization")
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--updates-per-step', type=int, default=1, help='Number of updates per step')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    # Environment parameters
    parser.add_argument('--ka-values', type=int, nargs='+', default=[5, 12, 20, 27, 35],
                        help='Ka values to simulate in parallel')
    parser.add_argument('--num-sims', type=int, default=100, help='Number of simulations per Ka')
    
    # Network parameters
    parser.add_argument('--state-dim', type=int, default=10, help='State dimension')
    parser.add_argument('--action-dim', type=int, default=1024, help='Action dimension')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden layer dimension')
    
    # SAC parameters
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='Temperature parameter')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update parameter')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--buffer-capacity', type=int, default=100000, help='Replay buffer size')
    
    # Logging and saving
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save-interval', type=int, default=50, help='Save interval')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')
    
    args = parser.parse_args()
    
    # Validate CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Run training
    train_sac(args)

if __name__ == "__main__":
    main() 