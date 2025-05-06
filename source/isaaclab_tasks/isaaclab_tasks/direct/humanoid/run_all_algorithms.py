import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise, GaussianNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import gc

# Common parameters
n_steps = int(0.5e6)
n_runs = 5
memory_size = 15625
num_envs = 256

# Define model classes (moved to top for reuse)

# Shared model for PPO
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
        )

        self.mean_layer = nn.Sequential(
            nn.Linear(256, self.num_actions),
            nn.Tanh()
        )

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Sequential(
            nn.Linear(256, 1),
        )

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}

# Stochastic Actor for SAC
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                clip_log_std=True, min_log_std=-5, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, self.num_actions),
                                nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}

# Deterministic Actor for DDPG and TD3
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

# Critic for SAC, DDPG and TD3
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


# Load the environment
env = load_isaaclab_env(task_name="Isaac-Humanoid-Direct-v0", num_envs=num_envs)
env = wrap_env(env)
device = env.device

# Algorithm configuration factory
def get_algorithm_config(algorithm):
    if algorithm == "PPO":
        cfg = PPO_DEFAULT_CONFIG.copy()
        cfg["rollouts"] = 32
        cfg["learning_epochs"] = 8
        cfg["mini_batches"] = 8
        cfg["discount_factor"] = 0.99
        cfg["lambda"] = 0.95
        cfg["learning_rate"] = 3e-4
        cfg["learning_rate_scheduler"] = KLAdaptiveRL
        cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
        cfg["random_timesteps"] = 0
        cfg["learning_starts"] = 0
        cfg["grad_norm_clip"] = 1.0
        cfg["ratio_clip"] = 0.2
        cfg["value_clip"] = 0.2
        cfg["clip_predicted_values"] = True
        cfg["entropy_loss_scale"] = 0.0
        cfg["value_loss_scale"] = 1.0
        cfg["kl_threshold"] = 0
        cfg["time_limit_bootstrap"] = True
        cfg["experiment"]["directory"] = "runs/torch/Isaac-Humanoid-Direct-v0/PPO2/"
        
    elif algorithm == "SAC":
        cfg = SAC_DEFAULT_CONFIG.copy()
        cfg["gradient_steps"] = 1
        cfg["batch_size"] = 4096
        cfg["discount_factor"] = 0.99
        cfg["polyak"] = 0.005
        cfg["actor_learning_rate"] = 3e-4
        cfg["critic_learning_rate"] = 3e-4
        cfg["random_timesteps"] = 1000
        cfg["learning_starts"] = 1000
        cfg["grad_norm_clip"] = 0
        cfg["learn_entropy"] = True
        cfg["entropy_learning_rate"] = 5e-3
        cfg["initial_entropy_value"] = 0.2
        cfg["experiment"]["directory"] = "runs/torch/Isaac-Humanoid-Direct-v0/SAC_03/"
        
    elif algorithm == "DDPG":
        cfg = DDPG_DEFAULT_CONFIG.copy()
        cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=0.5, device=device)
        cfg["gradient_steps"] = 1
        cfg["batch_size"] = 4096
        cfg["discount_factor"] = 0.99
        cfg["polyak"] = 0.005
        cfg["actor_learning_rate"] = 3e-4
        cfg["critic_learning_rate"] = 3e-4
        cfg["random_timesteps"] = 1000
        cfg["learning_starts"] = 1000
        cfg["experiment"]["directory"] = "runs/torch/Isaac-Humanoid-Direct-v0/DDPG/"
        
    elif algorithm == "TD3":
        cfg = TD3_DEFAULT_CONFIG.copy()
        cfg["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
        cfg["smooth_regularization_noise"] = GaussianNoise(0, 0.2, device=device)
        cfg["smooth_regularization_clip"] = 0.5
        cfg["gradient_steps"] = 1
        cfg["batch_size"] = 4096
        cfg["discount_factor"] = 0.99
        cfg["polyak"] = 0.005
        cfg["actor_learning_rate"] = 3e-4
        cfg["critic_learning_rate"] = 3e-4
        cfg["random_timesteps"] = 1000
        cfg["learning_starts"] = 1000
        cfg["experiment"]["directory"] = "runs/torch/Isaac-Humanoid-Direct-v0/TD3_03/"
    
    # Common configurations for all algorithms
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    if algorithm in ["PPO", "SAC"]:
        cfg["value_preprocessor"] = RunningStandardScaler
        cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    
    # Common logging configurations
    cfg["experiment"]["write_interval"] = 800
    cfg["experiment"]["checkpoint_interval"] = 10000
    
    return cfg


# Create models based on algorithm type
def create_models(algorithm):
    models = {}
    
    if algorithm == "PPO":
        models["policy"] = Shared(env.observation_space, env.action_space, device)
        models["value"] = models["policy"]  # Shared model
    
    elif algorithm == "SAC":
        models["policy"] = StochasticActor(env.observation_space, env.action_space, device)
        models["critic_1"] = Critic(env.observation_space, env.action_space, device)
        models["critic_2"] = Critic(env.observation_space, env.action_space, device)
        models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
        models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)
    
    elif algorithm == "DDPG":
        models["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
        models["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
        models["critic"] = Critic(env.observation_space, env.action_space, device)
        models["target_critic"] = Critic(env.observation_space, env.action_space, device)
    
    elif algorithm == "TD3":
        models["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
        models["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
        models["critic_1"] = Critic(env.observation_space, env.action_space, device)
        models["critic_2"] = Critic(env.observation_space, env.action_space, device)
        models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
        models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)
    
    return models


# Initialize and run agent based on algorithm type
def run_algorithm(algorithm):
    print(f"\n[INFO] Running {algorithm} experiments")
    cfg = get_algorithm_config(algorithm)
    global env
    for i in range(n_runs):

        set_seed(i)
        print(f"[INFO] Running {algorithm} experiment {i}")
        
        # Set experiment name
        cfg["experiment"]["experiment_name"] = f"run_{i}"
        
        # Create memory based on algorithm
        if algorithm == "PPO":
            memory = RandomMemory(memory_size=32, num_envs=env.num_envs, device=device)
        else:
            memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=device)
        
        # Create models
        models = create_models(algorithm)
        
        # Create agent
        if algorithm == "PPO":
            agent = PPO(models=models, memory=memory, cfg=cfg, 
                        observation_space=env.observation_space, 
                        action_space=env.action_space, device=device)
        elif algorithm == "SAC":
            agent = SAC(models=models, memory=memory, cfg=cfg,
                        observation_space=env.observation_space,
                        action_space=env.action_space, device=device)
        elif algorithm == "DDPG":
            agent = DDPG(models=models, memory=memory, cfg=cfg,
                        observation_space=env.observation_space,
                        action_space=env.action_space, device=device)
        elif algorithm == "TD3":
            agent = TD3(models=models, memory=memory, cfg=cfg,
                        observation_space=env.observation_space,
                        action_space=env.action_space, device=device)
        
        # Configure and run trainer
        cfg_trainer = {"timesteps": n_steps, "headless": True, 'close_environment_at_exit': False}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
        trainer.train()

        if algorithm == "TD3" and i == 4:
            trainer.env.close()

def main():
    # Set initial seed
    set_seed()
    # Run each algorithm
    #algorithms = ["PPO", "SAC", "DDPG", "TD3"]
    algorithms = ["SAC"]
    
    for algorithm in algorithms:
        run_algorithm(algorithm)

if __name__ == "__main__":
    main()