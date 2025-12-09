import argparse
import os
import torch
import torch.nn as nn
import gymnasium as gym

from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler


# ---------------------------
# Models: Gaussian policy & Value
# ---------------------------
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device,
                 clip_actions=True, clip_log_std=True,
                 min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, self.num_actions)
        )
        # one learnable log-std per action dim
        self.log_std_param = nn.Parameter(torch.full((self.num_actions,), -0.5))  # ~σ=0.61

    def compute(self, inputs, role):
        mean = self.net(inputs["states"])
        log_std = self.log_std_param.expand_as(mean)
        return mean, log_std, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def make_env(hardcore=False, render=False, seed=42):
    # Gymnasium requires render_mode set at creation time if you want visuals
    # (trainer's "headless" flag doesn't change rendering for Gymnasium envs)
    # BipedalWalker: obs Box(24,), act Box(4,) in [-1, 1]
    # https://gymnasium.farama.org/environments/box2d/bipedal_walker/
    render_mode = "human" if render else None
    env = gym.make("BipedalWalker-v3", hardcore=hardcore, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return wrap_env(env)


def main():
    parser = argparse.ArgumentParser(description="PPO on BipedalWalker (skrl, PyTorch)")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--hardcore", action="store_true", help="Use hardcore variant")
    parser.add_argument("--render", action="store_true", help="Render (Gymnasium needs render_mode at make())")
    parser.add_argument("--eval_only", type=str, default="", help="Path to checkpoint for evaluation only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="bipedalwalker_ppo")
    parser.add_argument("--entropy", type=float, default=0.0, help="Entropy coeff (try 0.0–0.01)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollouts", type=int, default=2048, help="Transitions per update (per env)")
    parser.add_argument("--epochs", type=int, default=10, help="Learning epochs per update")
    parser.add_argument("--minibatches", type=int, default=32, help="Mini-batches per update")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    env = make_env(hardcore=args.hardcore, render=args.render, seed=args.seed)
    device = env.device

    # ---------------------------
    # Models
    # ---------------------------
    policy = Policy(env.observation_space, env.action_space, device, clip_actions=True)
    value = Value(env.observation_space, env.action_space, device)
    models = {"policy": policy, "value": value}

    # ---------------------------
    # Memory (on-policy: set to rollouts)
    # ---------------------------
    memory = RandomMemory(memory_size=args.rollouts, num_envs=env.num_envs, device=device)

    # ---------------------------
    # PPO config (SB3-like defaults mapped to skrl)
    # ---------------------------
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg.update({
        "rollouts": args.rollouts,
        "learning_epochs": args.epochs,
        "mini_batches": args.minibatches,

        "learning_rate": args.lr,
        "discount_factor": 0.99,
        "lambda": 0.95,

        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,

        "entropy_loss_scale": args.entropy,
        "value_loss_scale": 0.5,
        "grad_norm_clip": 0.5,

        # State normalization helps on BW
        "state_preprocessor": RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space},

        "experiment": {
            "directory": "runs",
            "experiment_name": args.exp_name,
            "write_interval": "auto",         # ~100 writes over training
            "checkpoint_interval": "auto",    # ~10 checkpoints over training
            "store_separately": False,
            "wandb": False,
            "wandb_kwargs": {}
        }
    })

    agent = PPO(models=models,
                memory=memory if not args.eval_only else None,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

    # ---------------------------
    # Eval-only path
    # ---------------------------
    if args.eval_only:
        agent.load(args.eval_only)
        trainer = SequentialTrainer(env=env, agents=agent, cfg={"timesteps": 1_000, "headless": False})
        trainer.eval()
        return

    # ---------------------------
    # Train then short eval
    # ---------------------------
    trainer = SequentialTrainer(env=env, agents=agent, cfg={"timesteps": args.timesteps, "headless": not args.render})
    trainer.train()

    # Load "best" checkpoint (saved automatically) if present, then render a short eval
    best_ckpt = os.path.join("runs", args.exp_name, "checkpoints", "best_agent.pt")
    if os.path.isfile(best_ckpt):
        agent.load(best_ckpt)
    eval_steps = min(10_000, max(2_000, args.rollouts))  # quick watch
    trainer = SequentialTrainer(env=env, agents=agent, cfg={"timesteps": eval_steps, "headless": False})
    trainer.eval()


if __name__ == "__main__":
    main()
