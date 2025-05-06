import os
import time
import numpy as np
import torch
import torch.nn as nn

import isaaclab
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env

from evotorch import Problem
from evotorch.algorithms import PGPE
from evotorch.logging import StdOutLogger
from evotorch.neuroevolution import GymNE

# ----- global settings -------------------------------------------------------
device = torch.device("cpu")            # keep EvoTorch happy
SEED   = 42                              # base RNG seed
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------------------------------------------------------


class HumanoidPolicy(nn.Module):
    """Simple MLP policy for the IsaacLab Humanoid task"""
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 256),     nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh(),
        )
        self.to(device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.device != device:
            x = x.to(device)

        x = torch.nan_to_num(x, nan=0.0)
        out = self.net(x)
        return torch.nan_to_num(out, nan=0.0)


# -----------------------------------------------------------------------------


class SafeIsaacEnvWrapper:
    """Tiny gym‑style wrapper adding NaN‑sanitising and numpy<>torch glue"""

    def __init__(self, task_name: str, num_envs: int = 64, seed: int = SEED):
        print(f"Creating IsaacLab task '{task_name}'  |  num_envs = {num_envs}")
        self.env = wrap_env(
            load_isaaclab_env(task_name=task_name,
                              num_envs=num_envs,
                              headless=True)
        )
        self.num_envs         = num_envs
        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space

    # gym‑like API -------------------------------------------------------------

    def reset(self):
        obs, info = self.env.reset()
        obs = torch.nan_to_num(obs, nan=0.0)
        return obs.cpu().numpy()

    def step(self, act):
        if isinstance(act, np.ndarray):
            act = torch.from_numpy(act).float()
        act = torch.clamp(act, -1.0, 1.0)

        obs, rew, term, trunc, info = self.env.step(act)
        obs  = torch.nan_to_num(obs,  nan=0.0)
        rew  = torch.nan_to_num(rew,  nan=0.0)
        term = term.cpu().numpy()
        trunc = trunc.cpu().numpy()

        done = np.logical_or(term, trunc)
        return (obs.cpu().numpy(),
                rew.cpu().numpy(),
                done,
                info)

    def close(self):
        self.env.close()


# -----------------------------------------------------------------------------


class ImprovedGymNE(GymNE):
    """GymNE subclass that works with both old (‘env’) and new (‘env_maker’) APIs"""

    def __init__(self, network, env_maker, **kwargs):
        # Try the new name first (0.8.0+). If that fails, fall back to the old.
        try:
            super().__init__(network=network, env_maker=env_maker, **kwargs)
        except TypeError as e:
            if "env_maker" in str(e):
                super().__init__(network=network, env=env_maker, **kwargs)
            else:
                raise

    def _compute_fitness(self, rewards):
        # GymNE upstream wants a NumPy vector
        if isinstance(rewards, np.ndarray):
            rewards = np.nan_to_num(rewards, nan=0.0)
        else:          # Torch tensor
            rewards = torch.nan_to_num(rewards, nan=0.0).cpu().numpy()

        fitness = super()._compute_fitness(rewards)
        # make sure nothing has gone NaN
        fitness = np.nan_to_num(fitness, nan=-1000.0)
        return fitness


# -----------------------------------------------------------------------------


def train_humanoid(
    task_name: str = "Isaac-Humanoid-Direct-v0",
    num_envs: int = 64,
    popsize: int = 64,
    generations: int = 100,
    save_dir: str = "./results",
    seed: int = SEED,
):
    os.makedirs(save_dir, exist_ok=True)

    # --- build a one‑shot env just to read dimensions ------------------------
    tmp_env = SafeIsaacEnvWrapper(task_name, num_envs=num_envs, seed=seed)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    tmp_env.close()
    print(f"Obs dim = {obs_dim} | Act dim = {act_dim}")

    # --- network & EvoTorch plumbing ----------------------------------------
    network = HumanoidPolicy(obs_dim, act_dim)

    def make_env():
        return SafeIsaacEnvWrapper(task_name, num_envs=num_envs, seed=seed+123)

    problem = ImprovedGymNE(
        network        = network,
        env_maker      = make_env,
        num_actors     = 1,               # one actor controlling vectorised envs
        initial_bounds = (-0.1, 0.1),     # small init range
    )

    searcher = PGPE(
        problem               = problem,
        popsize               = popsize,
        center_learning_rate  = 0.01125,
        stdev_learning_rate   = 0.10,
        optimizer_config      = {"max_speed": 0.015},
        radius_init           = 0.27,
        num_interactions      = 150_000,
        popsize_max           = popsize,
        seed                  = seed,
    )

    StdOutLogger(searcher, interval=1)

    print(f"\n=== Evolution: {generations} generations, pop = {popsize} ===")
    t0 = time.time()
    searcher.run(generations)
    t_total = time.time() - t0
    print(f"Done in {t_total:.1f} s")

    # --- save artefacts ------------------------------------------------------
    best_sol     = searcher.status["best"]          # NumPy (num_params,)
    best_fitness = searcher.status["best_eval"]

    best_sol_t   = torch.tensor(best_sol, dtype=torch.float32)
    torch.save(best_sol_t, os.path.join(save_dir, "best_solution.pt"))

    problem.load_parameters(best_sol_t)
    torch.save(network.state_dict(),
               os.path.join(save_dir, "best_network.pt"))

    with open(os.path.join(save_dir, "metadata.txt"), "w") as fh:
        fh.write(f"task            : {task_name}\n")
        fh.write(f"generations     : {generations}\n")
        fh.write(f"population size : {popsize}\n")
        fh.write(f"best fitness    : {best_fitness}\n")
        fh.write(f"wall‑clock (s)  : {t_total:.1f}\n")

    print(f"Best fitness = {best_fitness}")
    return best_fitness, network


# -----------------------------------------------------------------------------


def main():
    print("=== Humanoid PGPE demo ===")
    outdir = os.path.join(os.getcwd(), "humanoid_results")
    train_humanoid(save_dir=outdir)


if __name__ == "__main__":
    main()
