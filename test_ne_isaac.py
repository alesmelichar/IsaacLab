"""
Neuroevo-Torch: a single-file, torch-first, GPU-ready, vectorized neuroevolution library
=======================================================================================

Design goals
------------
- **Torch-first**: all heavy math uses PyTorch, runs on CPU or CUDA.
- **Vectorized**: batch ops over population; avoid Python loops across individuals.
- **Module-friendly**: clean classes for networks, operators (selection, crossover, mutation), and the optimizer.
- **Copy-efficient**: genomes live in a single flat tensor; no deepcopy of models.
- **Parallel eval**: supports vectorized Gymnasium / Isaac Lab envs; evaluate `num_envs == population_size` in parallel (or in batches).

Quick peek
----------
```python
from neuroevo_torch import (
    FunctionalMLP, Neuroevolution,
    BestSelection, RandomSelection, TournamentSelection, RouletteWheelSelection,
    AdditiveMutation, GlobalMutation, KPointCrossover, Block
)

# Define policy structure (obs_dim/action_dim can be read from env)
policy = FunctionalMLP([obs_dim, 128, 128, action_dim], hidden_activation="tanh", output_activation="tanh")

# Reproduction plan: 5 elites + 45 mutated from random parents (with crossover)
plan = [
    Block(count=5, selection=BestSelection(), transforms=[]),
    Block(count=45,
          selection=RandomSelection(replace=True),
          transforms=[KPointCrossover(k=1), AdditiveMutation(p_gene=0.1, sigma=0.02), GlobalMutation(p_indiv=0.1, sigma=0.05)])
]

ne = Neuroevolution(policy=policy, population_size=50, plan=plan, device="cuda")
# Provide your evaluator (callable that maps genomes -> fitness tensor)
# or use the provided helper `make_gym_vectorized_evaluator` (see bottom).
```

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple
import math
import torch

Tensor = torch.Tensor

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

_ACTIVATIONS = {
    "identity": lambda x: x,
    "tanh": torch.tanh,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "elu": torch.nn.ELU,
    "leaky_relu": torch.nn.functional.leaky_relu,
}


def get_activation(name: str) -> Callable[[Tensor], Tensor]:
    if name not in _ACTIVATIONS:
        raise KeyError(f"Unknown activation '{name}'. Available: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]


@dataclass(frozen=True)
class LayerSpec:
    in_dim: int
    out_dim: int
    # Slices into flat genome
    w_slice: slice
    b_slice: slice


class FunctionalMLP:
    """MLP whose parameters live in a flat vector (genome) for vectorized evolution.

    - Stores shapes/slices to pack/unpack parameters.
    - Provides batched forward where each sample can use its own parameters.
    - No torch.nn.Module instances are created per individual.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        hidden_activation: str = "tanh",
        output_activation: Optional[str] = None,
        action_bounds: Optional[Tuple[float, float]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device | str] = None,
    ) -> None:
        assert len(layer_sizes) >= 2, "Need at least [in, out]"
        self.layer_sizes = list(map(int, layer_sizes))
        self.hidden_act = get_activation(hidden_activation)
        self.output_act = get_activation(output_activation) if output_activation else None
        self.action_bounds = action_bounds
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else None

        self.layers: List[LayerSpec] = []
        offset = 0
        for i in range(len(self.layer_sizes) - 1):
            in_d, out_d = self.layer_sizes[i], self.layer_sizes[i + 1]
            w_n = in_d * out_d
            b_n = out_d
            w_slice = slice(offset, offset + w_n)
            b_slice = slice(offset + w_n, offset + w_n + b_n)
            self.layers.append(LayerSpec(in_d, out_d, w_slice, b_slice))
            offset += w_n + b_n
        self.num_params: int = offset

    # ----- parameter (un)flattening -------------------------------------------------
    def unflatten_single(self, genome: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """Unflatten a single genome (D,) -> list of (W, b) tensors.
        Returns: list of (W[out,in], b[out])
        """
        assert genome.ndim == 1 and genome.shape[0] == self.num_params
        out: List[Tuple[Tensor, Tensor]] = []
        for ls in self.layers:
            w = genome[ls.w_slice].view(ls.out_dim, ls.in_dim)
            b = genome[ls.b_slice]
            out.append((w, b))
        return out

    def unflatten_batch(self, genomes: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """Unflatten batch genomes (P, D) -> list of (W, b) with batch dims.
        Returns: list of (W[P,out,in], b[P,out])
        """
        assert genomes.ndim == 2 and genomes.shape[1] == self.num_params
        P = genomes.shape[0]
        outs: List[Tuple[Tensor, Tensor]] = []
        for ls in self.layers:
            W = genomes[:, ls.w_slice].view(P, ls.out_dim, ls.in_dim)
            b = genomes[:, ls.b_slice].view(P, ls.out_dim)
            outs.append((W, b))
        return outs

    # ----- forward ---------------------------------------------------------------
    def forward_batched_params(self, x: Tensor, params: List[Tuple[Tensor, Tensor]]) -> Tensor:
        """Forward for vectorized params.

        Args:
            x: (P, in_dim) input where row p matches params[p].
            params: list of (W[P,out,in], b[P,out])
        Returns:
            y: (P, out_dim)
        """
        h = x
        last = len(params) - 1
        for i, (W, b) in enumerate(params):
            # h_i_out[p, j] = sum_k h[p, k] * W[p, j, k] + b[p, j]
            h = torch.einsum("pi,pok->po", h, W) + b
            if i < last:
                h = self.hidden_act(h)
        if self.output_act is not None:
            h = self.output_act(h)
        if self.action_bounds is not None:
            lo, hi = self.action_bounds
            # scale from [-1,1] to [lo, hi]
            h = (h + 1) * 0.5 * (hi - lo) + lo
        return h

    # ----- init helpers ----------------------------------------------------------
    def sample_init(self, population_size: int, scale: float = 0.1, device: Optional[str | torch.device] = None,
                    dtype: Optional[torch.dtype] = None, generator: Optional[torch.Generator] = None) -> Tensor:
        """Xavier-like random initialization of flat genomes.
        Returns tensor of shape (P, D).
        """
        dev = torch.device(device) if device is not None else (self.device or torch.device("cpu"))
        dt = dtype or self.dtype
        g = generator
        P, D = population_size, self.num_params
        genomes = torch.empty((P, D), device=dev, dtype=dt)
        # Layer-wise init to keep reasonable scales
        for ls in self.layers:
            fan_in = ls.in_dim
            std = scale / math.sqrt(max(1, fan_in))
            # Same init across population for a given parameter block but add small noise per individual
            base_w = torch.randn((ls.out_dim, ls.in_dim), device=dev, dtype=dt, generator=g) * std
            base_b = torch.randn((ls.out_dim,), device=dev, dtype=dt, generator=g) * std
            genomes[:, ls.w_slice] = base_w.flatten().unsqueeze(0)
            genomes[:, ls.b_slice] = base_b.unsqueeze(0)
            # jitter per individual
            genomes[:, ls.w_slice] += (torch.randn((P, ls.out_dim * ls.in_dim), device=dev, dtype=dt, generator=g) * std * 0.1)
            genomes[:, ls.b_slice] += (torch.randn((P, ls.out_dim), device=dev, dtype=dt, generator=g) * std * 0.1)
        return genomes


# -----------------------------------------------------------------------------
# Operators: selections, crossovers, mutations
# -----------------------------------------------------------------------------

class Operator:
    """Base class for reproduction operators."""
    def __call__(self, genomes: Tensor, fitness: Tensor, n: int, rng: Optional[torch.Generator] = None) -> Tensor:
        raise NotImplementedError


# ---- Selection ----------------------------------------------------------------
class Selection(Operator):
    pass


class BestSelection(Selection):
    def __init__(self, minimize: bool = False):
        self.minimize = minimize

    def __call__(self, genomes: Tensor, fitness: Tensor, n: int, rng: Optional[torch.Generator] = None) -> Tensor:
        # Return selected genomes (copied view) shape (n, D)
        k = min(n, genomes.shape[0])
        order = torch.argsort(fitness, descending=not self.minimize)
        idx = order[:k]
        return genomes[idx]


class RandomSelection(Selection):
    def __init__(self, replace: bool = True):
        self.replace = replace

    def __call__(self, genomes: Tensor, fitness: Tensor, n: int, rng: Optional[torch.Generator] = None) -> Tensor:
        P = genomes.shape[0]
        idx = torch.randint(P, (n,), device=genomes.device, generator=rng, dtype=torch.long,)
        if not self.replace:
            idx = torch.randperm(P, device=genomes.device, generator=rng)[:n]
        return genomes[idx]


class TournamentSelection(Selection):
    def __init__(self, tournament_size: int = 3, minimize: bool = False):
        self.k = int(tournament_size)
        self.minimize = minimize

    def __call__(self, genomes: Tensor, fitness: Tensor, n: int, rng: Optional[torch.Generator] = None) -> Tensor:
        P = genomes.shape[0]
        # sample n * k competitors
        comp_idx = torch.randint(P, (n, self.k), device=genomes.device, generator=rng)
        comp_fit = fitness[comp_idx]
        winners = comp_idx.gather(1, torch.argmin(comp_fit, dim=1, keepdim=True) if self.minimize else torch.argmax(comp_fit, dim=1, keepdim=True)).squeeze(1)
        return genomes[winners]


class RouletteWheelSelection(Selection):
    def __init__(self, minimize: bool = False, eps: float = 1e-8):
        self.minimize = minimize
        self.eps = eps

    def __call__(self, genomes: Tensor, fitness: Tensor, n: int, rng: Optional[torch.Generator] = None) -> Tensor:
        f = fitness
        if self.minimize:
            # Higher prob for lower fitness => invert by rank (stable)
            ranks = torch.argsort(torch.argsort(f))
            probs = (ranks.float() + 1.0)
        else:
            # Shift to positive
            fmin = torch.min(f)
            probs = (f - fmin) + self.eps
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, n, replacement=True, generator=rng)
        return genomes[idx]


# ---- Crossover ----------------------------------------------------------------
class Crossover(Operator):
    pass


class KPointCrossover(Crossover):
    def __init__(self, k: int = 1):
        assert k >= 1
        self.k = int(k)

    def __call__(self, parents: Tensor, fitness: Tensor, n: int, rng: Optional[torch.Generator] = None) -> Tensor:
        """Perform k-point crossover on randomly paired parents.

        Args:
            parents: (M, D) parent pool to sample pairs from (with replacement)
            n: number of children to produce (even -> pairs produce 2 children; odd -> last child from first mask)
        Returns:
            children: (n, D)
        """
        device = parents.device
        D = parents.shape[1]
        # sample pair indices
        P = parents.shape[0]
        pair_count = (n + 1) // 2
        idx_a = torch.randint(P, (pair_count,), device=device, generator=rng)
        idx_b = torch.randint(P, (pair_count,), device=device, generator=rng)
        A = parents[idx_a]
        B = parents[idx_b]
        # cuts in [1, D-1)
        if D < 2:
            children = torch.vstack((A, B))[:n]
            return children
        cuts = torch.randint(1, max(2, D), (pair_count, self.k), device=device, generator=rng)
        cuts, _ = torch.sort(torch.clamp(cuts, 1, D - 1), dim=1)
        # Build parity mask: for each position j, parity = (# cuts <= j) % 2
        pos = torch.arange(D, device=device).view(1, 1, D)
        parity = (pos >= cuts.unsqueeze(-1)).sum(dim=1) % 2  # (pair_count, D)
        mask_child1 = parity.bool()
        # child1 takes from A where parity==0, from B where parity==1
        child1 = torch.where(mask_child1, B, A)
        child2 = torch.where(mask_child1, A, B)
        children = torch.vstack((child1, child2))[:n]
        return children


# ---- Mutations ----------------------------------------------------------------
class Mutation(Operator):
    pass


class AdditiveMutation(Mutation):
    def __init__(self, p_gene: float = 0.05, sigma: float = 0.02):
        self.p_gene = float(p_gene)
        self.sigma = float(sigma)

    def __call__(self, genomes: Tensor, fitness: Tensor, n: int, rng: Optional[torch.Generator] = None) -> Tensor:
        # Select n genomes (by default: identity use "genomes" as input pool). Contract: caller passes selected parents.
        # Here, assume input `genomes` already represents the parents to transform. If `n` differs, resample with replacement.
        if genomes.shape[0] != n:
            idx = torch.randint(genomes.shape[0], (n,), device=genomes.device, generator=rng)
            genomes = genomes[idx]
        noise = torch.randn_like(genomes) * self.sigma
        if self.p_gene < 1.0:
            mask = (torch.rand_like(genomes) < self.p_gene)
            noise = noise * mask
        return genomes + noise


class GlobalMutation(Mutation):
    def __init__(self, p_indiv: float = 0.1, sigma: float = 0.05):
        self.p_indiv = float(p_indiv)
        self.sigma = float(sigma)

    def __call__(self, genomes: Tensor, fitness: Tensor, n: int, rng: Optional[torch.Generator] = None) -> Tensor:
        if genomes.shape[0] != n:
            idx = torch.randint(genomes.shape[0], (n,), device=genomes.device, generator=rng)
            genomes = genomes[idx]
        noise = torch.randn_like(genomes) * self.sigma
        mask = (torch.rand((n, 1), device=genomes.device, dtype=genomes.dtype, generator=rng) < self.p_indiv).to(genomes.dtype)
        return genomes + noise * mask


# -----------------------------------------------------------------------------
# Reproduction plan (elites + transformed offspring in blocks)
# -----------------------------------------------------------------------------

@dataclass
class Block:
    """A reproduction block: select a pool, then transform it to produce `count` children.

    Examples
    --------
    - Elites: `Block(count=5, selection=BestSelection(), transforms=[])`
    - Random parents -> crossover+mutate: `Block(count=45, selection=RandomSelection(), transforms=[KPointCrossover(1), AdditiveMutation(0.1)])`
    """
    count: int
    selection: Selection
    transforms: List[Operator]


# -----------------------------------------------------------------------------
# Neuroevolution optimizer
# -----------------------------------------------------------------------------

class Neuroevolution:
    def __init__(
        self,
        policy: FunctionalMLP,
        population_size: int,
        plan: List[Block],
        device: Optional[str | torch.device] = None,
        dtype: torch.dtype = torch.float32,
        init_scale: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.policy = policy
        self.population_size = int(population_size)
        self.plan = plan
        self.dtype = dtype
        self.device = torch.device(device) if device else torch.device("cpu")
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)
        self.population: Tensor = policy.sample_init(self.population_size, scale=init_scale, device=self.device, dtype=dtype, generator=self.rng)
        self.fitness: Optional[Tensor] = None
        self.generation: int = 0

    # ----- lifecycle ------------------------------------------------------------
    @torch.no_grad()
    def ask(self) -> Tensor:
        """Return current population genomes (view)."""
        return self.population

    @torch.no_grad()
    def tell(self, fitness: Tensor) -> None:
        assert fitness.shape[0] == self.population_size
        self.fitness = fitness.detach().to(self.population.device)

    @torch.no_grad()
    def step(self) -> None:
        """Create next generation using the configured reproduction plan."""
        assert self.fitness is not None, "Call tell(fitness) before step()"
        P, D = self.population.shape
        new_pop_parts: List[Tensor] = []
        genomes = self.population
        fit = self.fitness
        for block in self.plan:
            # 1) select a pool (size >= block.count ideally)
            pool = block.selection(genomes, fit, max(block.count, min(P, block.count)), rng=self.rng)
            children = pool
            # 2) sequential transforms; each transform must output exactly `block.count` individuals
            for op in block.transforms:
                children = op(children, fit, block.count, rng=self.rng)
            # If no transform: take the top `block.count` from pool (elites)
            if children.shape[0] != block.count:
                # fallback: crop/pad by sampling from pool
                if children.shape[0] > block.count:
                    children = children[: block.count]
                else:
                    extra_idx = torch.randint(pool.shape[0], (block.count - children.shape[0],), device=pool.device, generator=self.rng)
                    children = torch.vstack([children, pool[extra_idx]])
            new_pop_parts.append(children)
        new_pop = torch.vstack(new_pop_parts)
        # If plan doesn't sum exactly to population_size, fix it
        if new_pop.shape[0] != P:
            if new_pop.shape[0] > P:
                new_pop = new_pop[:P]
            else:
                pad_idx = torch.randint(self.population.shape[0], (P - new_pop.shape[0],), device=self.device, generator=self.rng)
                new_pop = torch.vstack([new_pop, self.population[pad_idx]])
        self.population = new_pop.to(self.device)
        self.fitness = None
        self.generation += 1

    # ----- helpers --------------------------------------------------------------
    @torch.no_grad()
    def best(self, k: int = 1, minimize: bool = False) -> Tuple[Tensor, Tensor]:
        assert self.fitness is not None, "No fitness yet. Call tell(fitness)."
        idx = torch.argsort(self.fitness, descending=not minimize)[:k]
        return self.population[idx], self.fitness[idx]


# -----------------------------------------------------------------------------
# Evaluation helpers (Gymnasium / Isaac Lab vectorized envs)
# -----------------------------------------------------------------------------

def _ensure_tensor(x: Tensor | "np.ndarray", device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    import numpy as np  # local import to avoid hard dep if not needed
    return torch.as_tensor(x, device=device, dtype=dtype)


@torch.no_grad()
def evaluate_population_in_vector_env(
    policy: FunctionalMLP,
    genomes: Tensor,
    env,
    rollout_horizon: int,
    obs_key: Optional[str] = None,
    action_clip: bool = True,
) -> Tensor:
    """Evaluate all individuals in parallel on a vectorized env.

    Contract for `env` (compatible with Isaac Lab / Gymnasium vector envs):
      - `env.reset()` returns observations (Tensor or ndarray or Dict[str,Tensor]) of shape (P, obs_dim)
      - `env.step(actions)` where actions has shape matching `env.action_space`. Isaac Lab expects torch on the env device.
      - Env is vectorized with `num_envs == P == genomes.shape[0]`.
      - Individual sub-envs auto-reset inside `step` (Isaac Lab behaviour). We therefore optimize for *throughput* and compute average return over a fixed horizon.

    Returns: fitness tensor shape (P,) with mean return per step over horizon.
    """
    device_env = getattr(env, "device", None)
    if device_env is None:
        # Isaac Lab: env.unwrapped.device
        device_env = getattr(getattr(env, "unwrapped", env), "device", torch.device("cpu"))
    device_env = torch.device(device_env)

    P = genomes.shape[0]
    params = policy.unflatten_batch(genomes.to(policy.dtype))

    # reset once (Isaac Lab recommendation)
    obs = env.reset()
    if isinstance(obs, tuple):  # gymnasium may return (obs, info)
        obs, _ = obs
    if isinstance(obs, dict) and obs_key is not None:
        obs = obs[obs_key]
    obs = _ensure_tensor(obs, device_env, policy.dtype)

    returns = torch.zeros((P,), device=device_env, dtype=policy.dtype)

    for _ in range(int(rollout_horizon)):
        # policy per-env forward (each env uses its own genome)
        actions = policy.forward_batched_params(obs, params)
        if action_clip:
            # Clip to [-1, 1]; Isaac Lab scales internally via cfg.action_scale
            actions = actions.clamp_(-1.0, 1.0)
        step_out = env.step(actions)
        # Gymnasium: (obs, reward, terminated, truncated, info)
        if len(step_out) == 5:
            obs, rew, terminated, truncated, info = step_out
        else:
            # Isaac Lab sometimes returns dict obs only
            obs, rew, terminated, truncated, info = step_out[0], step_out[1], step_out[2], step_out[3], step_out[4]
        if isinstance(obs, dict) and obs_key is not None:
            obs = obs[obs_key]
        obs = _ensure_tensor(obs, device_env, policy.dtype)
        rew = _ensure_tensor(rew, device_env, policy.dtype).view(-1)
        returns.add_(rew)
    # mean return per step
    return returns / float(rollout_horizon)


# Convenience factory to build an evaluator for Gym/IsaacLab tasks
class GymVectorEvaluator:
    """Callable evaluator compatible with Neuroevolution.tell/ask loop.

    Creates a vectorized env where `num_envs == population_size` and evaluates a population
    over a fixed horizon. Reuses the env across calls for speed (keeps the simulator alive).

    Notes
    -----
    - **Do not create a second Isaac Lab env in the same process** unless the previous one was
      properly closed. Pass an existing `env` if you've already created it with `gym.make(...)`.
    - This avoids the Isaac Lab error: *"Simulation context already exists. Cannot create a new one."*
    """

    def __init__(self, env_id: Optional[str] = None, population_size: Optional[int] = None, device: str = "cuda",
                 obs_key: Optional[str] = None, rollout_horizon: int = 256, headless: bool = True, cfg: Optional[object] = None,
                 env: Optional[object] = None):
        import gymnasium as gym
        self.env_id = env_id
        self.obs_key = obs_key
        self.rollout_horizon = int(rollout_horizon)
        if env is not None:
            # Reuse already-created env (recommended for Isaac Lab)
            self.env = env
        else:
            assert env_id is not None, "Provide env or env_id"
            # Create env. For Isaac Lab, pass cfg built from omni.isaac.lab_tasks.utils.parse_env_cfg
            if cfg is not None:
                self.env = gym.make(env_id, cfg=cfg)
            else:
                # Regular Gym vector envs can be created with vector.make; here we assume env supports num_envs=population_size via kwargs
                try:
                    self.env = gym.make(env_id, num_envs=population_size)
                except TypeError:
                    # Fallback: vectorize manually if plain single-env
                    from gymnasium.vector import SyncVectorEnv
                    assert population_size is not None, "population_size required when auto-vectorizing"
                    self.env = SyncVectorEnv([lambda: gym.make(env_id) for _ in range(population_size)])
        # Ensure reset before first step
        self.env.reset()

    def __call__(self, policy: FunctionalMLP, genomes: Tensor) -> Tensor:
        return evaluate_population_in_vector_env(policy, genomes, self.env, self.rollout_horizon, obs_key=self.obs_key)

# -----------------------------------------------------------------------------
# Example usage (Isaac Lab: Cartpole, Reach)
from isaaclab.app import AppLauncher
app = AppLauncher(headless=False).app  # must create the Kit app first

import gymnasium as gym
import isaaclab_tasks                        # pip-style package name
from isaaclab_tasks.utils import parse_env_cfg

# Create vectorized env: one sub-env per individual (population)
pop_size = 64
cartpole_id = "Isaac-Cartpole-v0"
cartpole_cfg = parse_env_cfg(cartpole_id, device="cuda:0", num_envs=pop_size, use_fabric=True)
cartpole_env = gym.make(cartpole_id, cfg=cartpole_cfg)

# Pick observation key/dim (Dict vs Box)
from gymnasium.spaces import Box, Dict as DictSpace
from gymnasium.spaces.utils import flatdim

def _pick_obs_key(space):
    if isinstance(space, Box):
        return None, int(space.shape[-1])
    if isinstance(space, DictSpace):
        for key in ("policy", "obs", "observation", "state"):
            if key in space.spaces and isinstance(space.spaces[key], Box):
                return key, int(space.spaces[key].shape[-1])
        # fallback: flatten
        return None, int(flatdim(space))
    raise TypeError(f"Unsupported observation space: {type(space)}")

obs_key_cp, obs_dim = _pick_obs_key(cartpole_env.observation_space)
action_dim = int(cartpole_env.action_space.shape[-1])


policy = FunctionalMLP([obs_dim, 64, 64, action_dim], hidden_activation="tanh", output_activation="tanh")
plan = [
    Block(count=8, selection=BestSelection(), transforms=[]),  # elites
    Block(count=56, selection=RandomSelection(replace=True), transforms=[KPointCrossover(1), AdditiveMutation(0.1, 0.02), GlobalMutation(0.1, 0.05)]),
]

evo = Neuroevolution(policy, population_size=pop_size, plan=plan, device="cuda", seed=0)

# IMPORTANT: reuse the env instead of creating a new one inside the evaluator
cp_evaluator = GymVectorEvaluator(env=cartpole_env, obs_key=obs_key_cp, rollout_horizon=256)

for gen in range(50):
    genomes = evo.ask()
    fitness = cp_evaluator(policy, genomes)
    evo.tell(fitness)
    best_g, best_f = evo.best()
    print(f"Gen {gen:03d} | avg={fitness.mean().item():.3f} | best={best_f.item():.3f}")
    evo.step()

# Cleanly close Cartpole before creating the next env
cartpole_env.close()
