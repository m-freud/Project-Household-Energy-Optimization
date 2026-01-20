"""
Reinforcement Learning Training Script

Trains RL agents using Stable-Baselines3 for household energy management.
Provides training, evaluation, and model persistence.

Requirements:
    pip install gymnasium stable-baselines3 tensorboard
    
Usage:
    python -m src.simulation.policies.rl_train
"""

import numpy as np
from pathlib import Path
import pickle

try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Error: stable-baselines3 not installed")
    print("Install with: pip install gymnasium stable-baselines3 tensorboard")
    exit(1)

from src.simulation.policies.rl_environment import EnergyManagementEnv
from src.simulation.requirements.charge_requirements import basic_charge_requirements


def make_env(rank: int = 0):
    """
    Create environment factory for parallel training.
    
    Args:
        rank: Index of the environment
    """
    def _init():
        env = EnergyManagementEnv(
            charge_requirements=basic_charge_requirements,
            constraint_penalty=100.0,
            early_completion_bonus=1.0
        )
        env = Monitor(env)  # Wrap for logging
        return env
    return _init


def train_ppo(
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    save_dir: str = "models/ppo_energy",
    tensorboard_log: str = "logs/ppo_energy"
):
    """
    Train PPO agent.
    
    PPO (Proximal Policy Optimization) is a robust on-policy algorithm
    that works well for continuous control tasks.
    
    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        tensorboard_log: Directory for tensorboard logs
    """
    print("=" * 80)
    print(" " * 25 + "TRAINING PPO AGENT")
    print("=" * 80)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Save directory: {save_dir}")
    print()
    
    # Create directories
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
    
    # Create vectorized environment
    if n_envs > 1:
        env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])
    
    # Create evaluation environment
    eval_env = Monitor(EnergyManagementEnv(
        charge_requirements=basic_charge_requirements,
        constraint_penalty=100.0
    ))
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best",
        log_path=f"{save_dir}/eval",
        eval_freq=10_000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=f"{save_dir}/checkpoints",
        name_prefix="ppo_model"
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    print("Starting training...")
    print("Monitor progress with: tensorboard --logdir logs/")
    print()
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f"{save_dir}/final_model")
    print(f"\n✓ Training complete! Model saved to {save_dir}/final_model")
    
    return model


def train_sac(
    total_timesteps: int = 100_000,
    n_envs: int = 1,
    save_dir: str = "models/sac_energy",
    tensorboard_log: str = "logs/sac_energy"
):
    """
    Train SAC agent.
    
    SAC (Soft Actor-Critic) is an off-policy algorithm that often achieves
    better sample efficiency than PPO.
    
    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        tensorboard_log: Directory for tensorboard logs
    """
    print("=" * 80)
    print(" " * 25 + "TRAINING SAC AGENT")
    print("=" * 80)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Save directory: {save_dir}")
    print()
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env = Monitor(EnergyManagementEnv(
        charge_requirements=basic_charge_requirements,
        constraint_penalty=100.0
    ))
    
    # Create evaluation environment
    eval_env = Monitor(EnergyManagementEnv(
        charge_requirements=basic_charge_requirements,
        constraint_penalty=100.0
    ))
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best",
        log_path=f"{save_dir}/eval",
        eval_freq=5_000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=f"{save_dir}/checkpoints",
        name_prefix="sac_model"
    )
    
    # Create SAC model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
    )
    
    print("Starting training...")
    print("Monitor progress with: tensorboard --logdir logs/")
    print()
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f"{save_dir}/final_model")
    print(f"\n✓ Training complete! Model saved to {save_dir}/final_model")
    
    return model


def evaluate_model(model_path: str, n_episodes: int = 10):
    """
    Evaluate trained model.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to evaluate
    """
    print(f"\nEvaluating model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print()
    
    # Load model
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "sac" in model_path.lower():
        model = SAC.load(model_path)
    else:
        print("Unknown model type")
        return
    
    # Create environment
    env = EnergyManagementEnv(
        charge_requirements=basic_charge_requirements,
        constraint_penalty=100.0
    )
    
    # Evaluate
    episode_rewards = []
    episode_costs = []
    episode_violations = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_costs.append(info.get('episode_cost', 0))
        episode_violations.append(info.get('constraint_violations', 0))
        
        print(f"Episode {ep+1}: Reward={episode_reward:.2f}, Cost={info.get('episode_cost', 0):.2f}, Violations={info.get('constraint_violations', 0)}")
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Cost: {np.mean(episode_costs):.2f} ± {np.std(episode_costs):.2f}")
    print(f"Average Violations: {np.mean(episode_violations):.2f}")
    print()


def rl_policy_wrapper(model_path: str):
    """
    Create policy function compatible with simulation interface.
    
    Args:
        model_path: Path to trained model
        
    Returns:
        Policy function that takes household and returns controls
    """
    # Load model once
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "sac" in model_path.lower():
        model = SAC.load(model_path)
    else:
        raise ValueError("Unknown model type")
    
    def rl_policy(household):
        """Policy function for simulation."""
        # Create temporary environment to get observation
        env = EnergyManagementEnv(
            charge_requirements=household.charge_requirements
        )
        env.household = household
        env.current_step = household.current_timestep
        
        # Get observation
        obs = env._get_observation()
        
        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        
        # Scale to controls
        controls = env._scale_action(action)
        
        return controls
    
    return rl_policy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL agent for energy management")
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "sac"],
                       help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=100_000,
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments (PPO only)")
    parser.add_argument("--eval-only", type=str, default=None,
                       help="Path to model for evaluation only")
    
    args = parser.parse_args()
    
    if args.eval_only:
        evaluate_model(args.eval_only)
    else:
        if args.algorithm == "ppo":
            train_ppo(total_timesteps=args.timesteps, n_envs=args.n_envs)
        elif args.algorithm == "sac":
            train_sac(total_timesteps=args.timesteps)
