"""
Steady-State Neuroevolution in PyTorch (asynchronous, RL-style)

Key ideas
---------
- Steady-state GA: evolve one (or a small batch of) offspring at a time and inject
  them into the population, instead of waiting for a full generation.
- Asynchronous evaluation: evaluate offspring in parallel processes; as soon as one
  finishes, update population and launch a new one. No global barrier.
- Elite archive ("bucket of best-so-far"): maintains the top-K solutions seen so far.
- Simple Gaussian mutation (optionally uniform-mask mutation). Optional uniform crossover.
- Works with Gymnasium environments (discrete or continuous actions).

This file is self-contained. Example usage is at the bottom (CartPole-v1 by default).

Notes
-----
- The environment is usually the bottleneck. This design keeps all workers busy and avoids
  idle time waiting for an entire generation to finish. Conceptually similar to on-policy RL
  loops where data is continuously collected and the policy is updated online, except here we
  update the population (and elite archive) online with gradient-free search.
- For serious use in continuous control, consider increasing the network size and the
  number of evaluation episodes to reduce variance.

Requirements
------------
- gymnasium (preferred) or gym as a fallback
- torch

Author: Aleš & ChatGPT
License: MIT
"""
from __future__ import annotations

import math
import time
import random
import dataclasses
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- Env import (Gymnasium preferred, fallback to gym) ----
try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore


# ----------------------------
# Utility: MLP policy network
# ----------------------------
class MLPPolicy(nn.Module):
    """Minimal MLP for both discrete and continuous action spaces.

    Discrete: outputs logits; action = argmax(logits)
    Continuous (Box): outputs tanh activations scaled to the action space bounds.
    """

    def __init__(self, obs_dim: int, action_space, hidden_sizes=(64, 64)):
        super().__init__()
        self.is_discrete = hasattr(action_space, "n")
        self.action_space = action_space

        layers: List[nn.Module] = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h

        if self.is_discrete:
            out_dim = int(action_space.n)
        else:
            out_dim = int(np.prod(action_space.shape))
        layers += [nn.Linear(last_dim, out_dim)]

        self.net = nn.Sequential(*layers)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, obs_dim]
        logits = self.net(x)
        if self.is_discrete:
            action = torch.argmax(logits, dim=-1).item()
            return np.array(action, dtype=np.int64)
        else:
            # Map to [-1, 1] then to env bounds
            raw = torch.tanh(logits).squeeze(0).cpu().numpy()
            low, high = self.action_space.low, self.action_space.high
            act = low + (raw + 1.0) * 0.5 * (high - low)
            return act.astype(np.float32)


# ---------------------------------------
# Genome helpers (flatten/unflatten params)
# ---------------------------------------

def num_params_of(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def vectorize_params(model: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector([p.detach() for p in model.parameters()])


def assign_from_vector(model: nn.Module, vec: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(vec, [p for p in model.parameters()])


# --------------------
# Mutation & crossover
# --------------------
@dataclass
class MutationCfg:
    sigma: float = 0.05  # std for Gaussian noise
    p: float = 1.0       # per-parameter mutation probability (mask)
    clip: Optional[float] = None  # clip perturbations to [-clip, clip] if provided


def gaussian_mutation(parent: torch.Tensor, cfg: MutationCfg, rng: torch.Generator) -> torch.Tensor:
    if cfg.p >= 1.0:
        noise = torch.randn_like(parent) * cfg.sigma
    else:
        mask = (torch.rand_like(parent) < cfg.p).float()
        noise = torch.randn_like(parent) * cfg.sigma * mask
    if cfg.clip is not None:
        noise = torch.clamp(noise, -cfg.clip, cfg.clip)
    return parent + noise


def uniform_crossover(a: torch.Tensor, b: torch.Tensor, p: float = 0.5, rng: Optional[torch.Generator] = None) -> torch.Tensor:
    if rng is None:
        rng = torch.Generator().manual_seed(torch.seed())
    mask = (torch.rand_like(a) < p).float()
    return a * mask + b * (1.0 - mask)


# ----------------
# Elite container
# ----------------
@dataclass
class Elite:
    params: torch.Tensor
    fitness: float


class EliteArchive:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.elites: List[Elite] = []

    def add(self, params: torch.Tensor, fitness: float) -> None:
        # Insert and keep sorted descending by fitness
        self.elites.append(Elite(params.detach().clone(), float(fitness)))
        self.elites.sort(key=lambda e: e.fitness, reverse=True)
        if len(self.elites) > self.capacity:
            self.elites.pop(-1)

    def best(self) -> Elite:
        return self.elites[0]

    def sample(self, k: int = 1) -> List[Elite]:
        if len(self.elites) == 0:
            raise ValueError("Elite archive empty")
        idxs = np.random.randint(0, len(self.elites), size=k)
        return [self.elites[i] for i in idxs]

    def __len__(self) -> int:
        return len(self.elites)


# -------------------------
# Population (steady-state)
# -------------------------
@dataclass
class Individual:
    params: torch.Tensor
    fitness: float = -float("inf")


class Population:
    def __init__(self, pop_size: int, param_dim: int):
        self.pop: List[Individual] = [Individual(torch.zeros(param_dim)) for _ in range(pop_size)]

    def update_fitness(self, idx: int, fitness: float):
        self.pop[idx].fitness = float(fitness)

    def replace(self, idx: int, params: torch.Tensor, fitness: float):
        self.pop[idx] = Individual(params.detach().clone(), float(fitness))

    def argmin(self) -> int:
        return int(np.argmin([ind.fitness for ind in self.pop]))

    def argmax(self) -> int:
        return int(np.argmax([ind.fitness for ind in self.pop]))

    def worst_k_indices(self, k: int) -> List[int]:
        order = np.argsort([ind.fitness for ind in self.pop])  # ascending
        return list(map(int, order[:k]))

    def best(self) -> Individual:
        return self.pop[self.argmax()]

    def sample_indices(self, n: int) -> List[int]:
        return random.sample(range(len(self.pop)), n)


# --------------------------
# Evaluation worker function
# --------------------------

def evaluate_genome(
    genome: np.ndarray,
    env_name: str,
    hidden_sizes: Tuple[int, ...],
    n_episodes: int = 1,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> float:
    """Evaluate a parameter vector by running n_episodes and returning mean episodic reward.

    This function is executed in a separate process; it creates its own env and model.
    """
    # Create env locally in the worker
    try:
        import gymnasium as gym_w
    except Exception:
        import gym as gym_w  # type: ignore

    env = gym_w.make(env_name)
    obs_space = env.observation_space
    if hasattr(obs_space, "shape"):
        obs_dim = int(np.prod(obs_space.shape))
    else:
        raise RuntimeError("Only Box observation spaces are supported.")

    policy = MLPPolicy(obs_dim, env.action_space, hidden_sizes=hidden_sizes)
    vec = torch.as_tensor(genome, dtype=torch.float32)
    assign_from_vector(policy, vec)

    rng = np.random.default_rng(seed)
    total = 0.0
    for ep in range(n_episodes):
        if seed is not None:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        else:
            obs, _ = env.reset()
        done = False
        steps = 0
        ret = 0.0
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += float(reward)
            steps += 1
            done = bool(terminated or truncated)
            if max_steps is not None and steps >= max_steps:
                break
        total += ret
    env.close()
    return total / float(n_episodes)


# ----------------------------------
# Steady-state Neuroevolution engine
# ----------------------------------
@dataclass
class NEConfig:
    env_name: str = "CartPole-v1"
    hidden_sizes: Tuple[int, ...] = (64, 64)
    pop_size: int = 128
    elite_capacity: int = 16
    tournament_k: int = 3
    mutation: MutationCfg = MutationCfg()
    crossover_rate: float = 0.0  # 0 = disabled; else probability of crossover before mutation
    n_episodes: int = 1
    max_steps: Optional[int] = None
    max_workers: int = 8
    seed: int = 42
    # Run control
    max_seconds: float = 120.0
    target_return: Optional[float] = None  # if None, will try to use env.spec.reward_threshold
    log_every: float = 5.0
    prefer_elites_prob: float = 0.7  # probability to pick parents from elite archive


class SteadyStateNE:
    def __init__(self, cfg: NEConfig):
        self.cfg = cfg
        self.device = torch.device("cpu")  # inference-only here; env is bottleneck
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # Build a dummy env to infer sizes and reward threshold
        env = gym.make(cfg.env_name)
        obs_space = env.observation_space
        if hasattr(obs_space, "shape"):
            obs_dim = int(np.prod(obs_space.shape))
        else:
            raise RuntimeError("Only Box observation spaces are supported.")
        self.policy_template = MLPPolicy(obs_dim, env.action_space, hidden_sizes=cfg.hidden_sizes)

        self.param_dim = num_params_of(self.policy_template)
        self.population = Population(cfg.pop_size, self.param_dim)
        self.elites = EliteArchive(cfg.elite_capacity)

        # Initialize population around 0 with small noise
        base = vectorize_params(self.policy_template)
        for i in range(cfg.pop_size):
            self.population.pop[i].params = gaussian_mutation(base, MutationCfg(sigma=0.1, p=1.0), torch.Generator().manual_seed(cfg.seed + i))

        # Determine target return if not provided
        if cfg.target_return is None and getattr(env, "spec", None) is not None:
            try:
                self.target_return = float(env.spec.reward_threshold)
            except Exception:
                self.target_return = None
        else:
            self.target_return = cfg.target_return
        env.close()

    def _tournament_select(self, k: int) -> int:
        idxs = self.population.sample_indices(k)
        best_idx = max(idxs, key=lambda i: self.population.pop[i].fitness)
        return best_idx

    def _select_parents(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Choose from elites with some probability
        use_elite = (len(self.elites) > 0) and (random.random() < self.cfg.prefer_elites_prob)
        if use_elite:
            p1 = self.elites.sample(1)[0].params
        else:
            i1 = self._tournament_select(self.cfg.tournament_k)
            p1 = self.population.pop[i1].params

        p2: Optional[torch.Tensor] = None
        if self.cfg.crossover_rate > 0.0 and random.random() < self.cfg.crossover_rate:
            # Either from elites or tournament
            if len(self.elites) > 0 and random.random() < 0.5:
                p2 = self.elites.sample(1)[0].params
            else:
                i2 = self._tournament_select(self.cfg.tournament_k)
                p2 = self.population.pop[i2].params
        return p1, p2

    def _make_child(self, rng: torch.Generator) -> torch.Tensor:
        p1, p2 = self._select_parents()
        child = p1
        if p2 is not None:
            child = uniform_crossover(p1, p2, p=0.5, rng=rng)
        child = gaussian_mutation(child, self.cfg.mutation, rng)
        return child

    def _replace_strategy(self, fitness: float) -> int:
        # Replace a random individual from the worst quartile (more exploratory than always replacing the absolute worst)
        k = max(1, len(self.population.pop) // 4)
        worst_idxs = self.population.worst_k_indices(k)
        return random.choice(worst_idxs)

    def run(self) -> Tuple[torch.Tensor, float]:
        """Run asynchronous steady-state evolution. Returns best params and fitness."""
        cfg = self.cfg
        start = time.time()
        last_log = start
        best_f = -float("inf")
        best_params = None

        # Warm up: evaluate a small random subset to seed elites
        warm = min(8, cfg.pop_size)
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
            warm_futs = []
            for i in range(warm):
                genome = self.population.pop[i].params.cpu().numpy().astype(np.float32)
                fut = ex.submit(
                    evaluate_genome,
                    genome,
                    cfg.env_name,
                    cfg.hidden_sizes,
                    cfg.n_episodes,
                    cfg.max_steps,
                    cfg.seed + i,
                )
                warm_futs.append((i, fut))
            for i, fut in warm_futs:
                fit = float(fut.result())
                self.population.update_fitness(i, fit)
                self.elites.add(self.population.pop[i].params, fit)
                if fit > best_f:
                    best_f = fit
                    best_params = self.population.pop[i].params.detach().clone()

        # Main async loop: fill worker pool with offspring evaluations and update as they finish
        n_submitted = 0
        futures = []
        rng_base = torch.Generator().manual_seed(cfg.seed + 12345)
        with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
            # Pre-fill futures
            for _ in range(cfg.max_workers):
                child = self._make_child(rng_base)
                fut = ex.submit(
                    evaluate_genome,
                    child.cpu().numpy().astype(np.float32),
                    cfg.env_name,
                    cfg.hidden_sizes,
                    cfg.n_episodes,
                    cfg.max_steps,
                    cfg.seed + n_submitted + 10_000,
                )
                futures.append((child, fut))
                n_submitted += 1

            while True:
                # Check time/target criteria
                now = time.time()
                if cfg.max_seconds is not None and (now - start) >= cfg.max_seconds:
                    break

                done_any = False
                # Consume any finished futures without blocking too long
                for j, (child_params, fut) in list(enumerate(futures)):
                    if fut.done():
                        done_any = True
                        fit = float(fut.result())
                        # Replacement
                        idx = self._replace_strategy(fit)
                        self.population.replace(idx, child_params, fit)
                        # Update elites and best
                        self.elites.add(child_params, fit)
                        if fit > best_f:
                            best_f = fit
                            best_params = child_params.detach().clone()

                        # Refill this slot with a new child
                        new_child = self._make_child(rng_base)
                        new_fut = ex.submit(
                            evaluate_genome,
                            new_child.cpu().numpy().astype(np.float32),
                            cfg.env_name,
                            cfg.hidden_sizes,
                            cfg.n_episodes,
                            cfg.max_steps,
                            cfg.seed + n_submitted + 10_000,
                        )
                        futures[j] = (new_child, new_fut)
                        n_submitted += 1

                # Logging
                if (now - last_log) >= cfg.log_every:
                    evals_per_sec = n_submitted / max(1e-6, (now - start))
                    msg = f"t={now - start:.1f}s | best={best_f:.2f} | elites={len(self.elites)} | submitted={n_submitted} | ~{evals_per_sec:.2f} eval/s"
                    print(msg, flush=True)
                    last_log = now

                # Target return reached?
                if self.target_return is not None and best_f >= self.target_return:
                    break

                # If none finished very recently, small sleep to avoid busy spin
                if not done_any:
                    time.sleep(0.01)

        assert best_params is not None, "No evaluations performed."
        return best_params, best_f


# ----------------
# Demo / Usage
# ----------------
if __name__ == "__main__":
    cfg = NEConfig(
        env_name="BipedalWalker-v3",  # Try also CartPole-v1, MountainCarContinuous-v0, LunarLander-v2, etc.
        hidden_sizes=(64, 64),
        pop_size=32,
        elite_capacity=16,
        tournament_k=3,
        mutation=MutationCfg(sigma=0.05, p=1.0, clip=0.5),
        crossover_rate=0.0,  # Try 0.1 for a taste of crossover
        n_episodes=2,        # average over 2 eps to reduce variance
        max_steps=None,
        max_workers=8,
        seed=123,
        max_seconds=120.0,
        target_return=None,  # will use env.spec.reward_threshold if available
        log_every=5.0,
        prefer_elites_prob=0.7,
    )

    engine = SteadyStateNE(cfg)
    best_params, best_f = engine.run()
    print("\nFinished! Best fitness:", best_f)

    # Optional: run one evaluation episode with the best policy for a visual/manual check
    try:
        import gymnasium as gym
    except Exception:
        import gym  # type: ignore

    env = gym.make(cfg.env_name, render_mode=None)
    obs, _ = env.reset(seed=cfg.seed + 555)

    # Build policy and assign best params
    obs_dim = int(np.prod(env.observation_space.shape))
    policy = MLPPolicy(obs_dim, env.action_space, hidden_sizes=cfg.hidden_sizes)
    assign_from_vector(policy, torch.as_tensor(best_params))

    done = False
    ret = 0.0
    steps = 0
    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        ret += float(reward)
        steps += 1
        done = bool(terminated or truncated)
    env.close()
    print(f"Rollout with best policy: return={ret:.2f}, steps={steps}")
