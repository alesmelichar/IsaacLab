import torch
import torch.nn as nn
import os
import glob
from datetime import datetime

# import the skrl components to build the RL system
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import GaussianNoise
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

# Function to browse and select checkpoint files
def browse_checkpoints(env_name):
    # Base checkpoint directory
    base_path = f"/home/urkui-3/Documents/Ales/IsaacLab/runs/torch/{env_name}/"
    
    if not os.path.exists(base_path):
        print(f"Base checkpoint directory not found: {base_path}")
        custom_path = input("Enter a custom base directory path: ")
        if custom_path and os.path.exists(custom_path):
            base_path = custom_path
        else:
            print("Invalid directory, please enter the full checkpoint path:")
            return input("Checkpoint path: ")
    
    # Find latest directory by modification time
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # First check standard location 
    standard_path = os.path.join(base_path, "td3", "checkpoints")
    if os.path.exists(standard_path) and os.path.isdir(standard_path):
        checkpoint_files = glob.glob(os.path.join(standard_path, "*.pt"))
        if checkpoint_files:
            # Sort by modification time, newest first
            checkpoint_files.sort(key=os.path.getmtime, reverse=True)
            newest_checkpoint = checkpoint_files[0]
            print(f"\nFound latest checkpoint in standard location: {os.path.basename(newest_checkpoint)}")
            use_latest = input("Use this file? (y/n, default: y): ")
            if not use_latest or use_latest.lower() == 'y':
                return newest_checkpoint
    
    # If we get here, either no standard checkpoint exists or user doesn't want to use it
    if subdirs:
        # Sort subdirectories by modification time, newest first
        full_subdirs = [os.path.join(base_path, d) for d in subdirs]
        full_subdirs.sort(key=os.path.getmtime, reverse=True)
        subdirs = [os.path.basename(d) for d in full_subdirs]
        
        print("\nAvailable subdirectories (sorted by newest first):")
        subdirs = ["<current directory>"] + subdirs
        for i, subdir in enumerate(subdirs, 1):
            print(f"{i}: {subdir}")
        
        # Let user select a subdirectory
        subdir_choice = input("\nSelect a subdirectory (number) or enter a custom path: ")
        
        try:
            idx = int(subdir_choice) - 1
            if 0 <= idx < len(subdirs):
                if idx == 0:  # Current directory
                    selected_dir = base_path
                else:
                    selected_dir = os.path.join(base_path, subdirs[idx])
            else:
                print("Invalid selection, using base directory")
                selected_dir = base_path
        except ValueError:
            # If not a number, treat as custom path
            if os.path.exists(subdir_choice) and os.path.isdir(subdir_choice):
                selected_dir = subdir_choice
            else:
                print("Invalid directory, using base directory")
                selected_dir = base_path
    else:
        # No subdirectories found
        selected_dir = base_path
    
    # Now look for checkpoint files in the selected directory
    checkpoint_files = glob.glob(os.path.join(selected_dir, "*.pt"))
    
    # Check for nested 'checkpoints' directory if no files found
    if not checkpoint_files:
        checkpoints_dir = os.path.join(selected_dir, "checkpoints")
        if os.path.exists(checkpoints_dir) and os.path.isdir(checkpoints_dir):
            checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.pt"))
            if checkpoint_files:
                selected_dir = checkpoints_dir
    
    # If still no files found, look in td3/checkpoints
    if not checkpoint_files:
        td3_checkpoints_dir = os.path.join(selected_dir, "td3", "checkpoints")
        if os.path.exists(td3_checkpoints_dir) and os.path.isdir(td3_checkpoints_dir):
            checkpoint_files = glob.glob(os.path.join(td3_checkpoints_dir, "*.pt"))
            if checkpoint_files:
                selected_dir = td3_checkpoints_dir
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {selected_dir}")
        return input("Enter full checkpoint path: ")
    
    # Sort checkpoint files by modification time, newest first
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"\nAvailable checkpoint files in {selected_dir} (sorted by newest first):")
    for i, file_path in enumerate(checkpoint_files, 1):
        file_name = os.path.basename(file_path)
        mod_time = os.path.getmtime(file_path)
        mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i}: {file_name} (modified: {mod_time_str})")
    
    # Let user select a file
    selection = input("\nSelect a checkpoint file (number) or enter a custom path: ")
    
    try:
        idx = int(selection) - 1
        if 0 <= idx < len(checkpoint_files):
            return checkpoint_files[idx]
    except ValueError:
        # If not a number, treat as custom path
        pass
    
    # If we got here, either selection was invalid or user entered custom path
    if os.path.exists(selection) and selection.endswith('.pt'):
        return selection
    else:
        print("Selection not valid, using latest checkpoint")
        # Return the newest checkpoint file
        if checkpoint_files:
            return checkpoint_files[0]  # Already sorted, first is newest
        else:
            # Last resort - standard path
            standard_path = f"/home/urkui-3/Documents/Ales/IsaacLab/runs/torch/{env_name}/td3/checkpoints/best_agent.pt"
            print(f"Falling back to standard path: {standard_path}")
            return standard_path
        
# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (deterministic models) using mixins
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, self.num_actions),
                                 nn.Tanh())

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# Define available environments
AVAILABLE_ENVS = {
    "1": "Isaac-Lift-Cube-Franka-v0",
    "2": "Isaac-Repose-Cube-Shadow-Direct-v0",
}

# Environment selection
print("\nAvailable environments:")
for key, name in AVAILABLE_ENVS.items():
    print(f"{key}: {name}")
    
env_choice = input(f"Select environment (1-{len(AVAILABLE_ENVS)}): ")
env_name = AVAILABLE_ENVS.get(env_choice)
if env_name is None:
    env_name = "Isaac-Lift-Cube-Franka-v0"  # Default
    print(f"Invalid choice, using default: {env_name}")
else:
    print(f"Selected: {env_name}")

# Mode selection
print("\nSelect mode:")
print("1: Train")
print("2: Evaluate")
mode_choice = input("Enter choice (1-2): ")
run_mode = "eval" if mode_choice == "2" else "train"
print(f"Selected mode: {run_mode}")

# Checkpoint path for evaluation
# Checkpoint path for evaluation
checkpoint_path = None
if run_mode == "eval":
    print("\nCheckpoint selection:")
    print("1: Browse checkpoints")
    print("2: Enter path manually")
    checkpoint_choice = input("Select option (1-2): ")
    
    if checkpoint_choice == "1":
        checkpoint_path = browse_checkpoints(env_name)
    else:
        checkpoint_path = input("\nEnter checkpoint path (leave empty for default): ")
        
    if not checkpoint_path:
        checkpoint_path = f"/home/urkui-3/Documents/Ales/IsaacLab/runs/torch/{env_name}/td3/checkpoints/best_agent.pt"
    
    print(f"Using checkpoint: {checkpoint_path}")
    
# Number of environments
if run_mode == "train":
    default_envs = 64
else:
    default_envs = 1

# Let user input number of environments
env_input = input(f"\nEnter number of environments (default: {default_envs}): ")
try:
    num_envs = int(env_input) if env_input.strip() else default_envs
    if num_envs <= 0:
        num_envs = default_envs
        print(f"Invalid number, using default: {default_envs}")
except ValueError:
    num_envs = default_envs
    print(f"Invalid input, using default: {default_envs}")

print(f"Using {num_envs} environment{'s' if num_envs > 1 else ''}")

# Load environment
env = load_isaaclab_env(task_name=env_name, num_envs=num_envs)
env = wrap_env(env)
device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=500000, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# TD3 requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/td3.html#models
models = {}
models["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/td3.html#configuration-and-hyperparameters
cfg = TD3_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = GaussianNoise(0, 0.2, device=device)
cfg["smooth_regularization_noise"] = GaussianNoise(0, 0.2, device=device)
cfg["smooth_regularization_clip"] = 0.5
cfg["gradient_steps"] = 1
cfg["batch_size"] = 4096
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 5e-4
cfg["critic_learning_rate"] = 5e-4
cfg["random_timesteps"] = 5000
cfg["learning_starts"] = 5000
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 800
cfg["experiment"]["checkpoint_interval"] = 8000
cfg["experiment"]["directory"] = f"runs/torch/{env_name}"

agent = TD3(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1600000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if run_mode == "train":
    print(f"\nStarting training on {env_name}...")
    trainer.train()
else:
    print(f"\nStarting evaluation on {env_name} using checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)
    trainer.eval()