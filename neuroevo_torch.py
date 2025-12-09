# neuroevo_torch.py
from __future__ import annotations
import math
from dataclasses import dataclass
from time import time
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
import tqdm

Tensor = torch.Tensor

_ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": F.relu,
    "elu": F.elu,
    "gelu": F.gelu,
    "sigmoid": torch.sigmoid,
    "identity": lambda x: x,
}

def _fan_in_uniform_init(shape: Tuple[int, ...], a: float = 1.0, generator: Optional[torch.Generator] = None, device=None, dtype=None) -> Tensor:
    if len(shape) < 2:
        bound = 1.0 / math.sqrt(shape[0]) if shape[0] > 0 else 1.0
    else:
        fan_in = shape[-1]
        bound = a / math.sqrt(fan_in) if fan_in > 0 else 1.0
    return (torch.rand(shape, generator=generator, device=device, dtype=dtype) * 2 - 1) * bound

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

class _TensorboardLogger:
    """Minimal SKRL-like TB logger: uses timesteps as global_step."""
    def __init__(self, log_dir: Optional[str] = None, writer: Optional["SummaryWriter"] = None):
        self.writer = writer if writer is not None else (SummaryWriter(log_dir=log_dir) if SummaryWriter else None)
        self.timesteps = 0  # cumulative env steps

    def add_timesteps(self, steps: int):
        if self.writer:
            self.timesteps += int(steps)
            # also log the total timesteps as a scalar so something shows up immediately
            self.writer.add_scalar("Timesteps/total", float(self.timesteps), self.timesteps)

    def log_rewards(self, mean_reward: float, max_reward: float):
        if self.writer:
            step = self.timesteps
            self.writer.add_scalar("Rewards/mean", float(mean_reward), step)
            self.writer.add_scalar("Rewards/max",  float(max_reward),  step)

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()

    def log_episode_length_stats(self, min_len: float, mean_len: float, max_len: float):
        if self.writer:
            step = self.timesteps
            self.writer.add_scalar("EpisodeLength/min",  float(min_len),  step)
            self.writer.add_scalar("EpisodeLength/mean", float(mean_len), step)
            self.writer.add_scalar("EpisodeLength/max",  float(max_len),  step)


class BatchedMLPPolicy:
    def __init__(
        self,
        obs_dim: int,
        hidden_layers: Sequence[int],
        act_dim: int,
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        action_space: str = "discrete",
        action_low: Optional[Union[float, np.ndarray]] = None,
        action_high: Optional[Union[float, np.ndarray]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        assert activation in _ACTIVATIONS, f"Unknown activation: {activation}"
        if output_activation:
            assert output_activation in _ACTIVATIONS, f"Unknown output activation: {output_activation}"
        self.obs_dim = obs_dim
        self.hidden_layers = list(hidden_layers)
        self.act_dim = act_dim
        self.activation = activation
        self.output_activation = output_activation
        self.action_space = action_space
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        if action_space not in ("discrete", "continuous"):
            raise ValueError("action_space must be 'discrete' or 'continuous'")
        if action_space == "continuous":
            if action_low is None or action_high is None:
                raise ValueError("For continuous actions, provide action_low and action_high")
            self.action_low = torch.as_tensor(action_low, device=self.device, dtype=self.dtype).view(1, -1)
            self.action_high = torch.as_tensor(action_high, device=self.device, dtype=self.dtype).view(1, -1)

            self._a = ((self.action_high - self.action_low) / 2.0).to(self.device, self.dtype)
            self._m = ((self.action_high + self.action_low) / 2.0).to(self.device, self.dtype)

        else:
            self.action_low = None
            self.action_high = None
        self.layer_in_dims = [self.obs_dim] + self.hidden_layers
        self.layer_out_dims = self.hidden_layers + [self.act_dim]
        self.param_shapes: List[Tuple[str, Tuple[int, int]]] = []
        total = 0
        for din, dout in zip(self.layer_in_dims, self.layer_out_dims):
            self.param_shapes.append(("W", (dout, din)))
            total += dout * din
            self.param_shapes.append(("b", (dout,)))
            total += dout
        self.param_dim = total

        self._act = _ACTIVATIONS[activation]
        self._out_act = _ACTIVATIONS[output_activation] if output_activation else None


    def sample_population(self, pop_size: int, scale: float = 1.0, generator: Optional[torch.Generator] = None) -> Tensor:
        flat = torch.empty((pop_size, self.param_dim), device=self.device, dtype=self.dtype)
        offset = 0
        for kind, shape in self.param_shapes:
            n = int(math.prod(shape))
            block = _fan_in_uniform_init((pop_size, n), a=scale, generator=generator, device=self.device, dtype=self.dtype)
            flat[:, offset:offset + n] = block
            offset += n
        return flat

    def unflatten(self, flat_params: Tensor):
        B = flat_params.shape[0]
        layers = []
        offset = 0
        for kind, shape in self.param_shapes:
            n = int(math.prod(shape))
            slab = flat_params[:, offset:offset + n]
            offset += n
            if kind == "W":
                W = slab.view(B, shape[0], shape[1])#.transpose(1, 2).contiguous() 
                layers.append((W, None))
            else:
                b = slab.view(B, shape[0])
                W, _ = layers[-1]
                layers[-1] = (W, b)
        return layers

    def forward(self, obs: Tensor, flat_params: Tensor) -> Tensor:
        """
        Standard forward that unflattens every call (simple but a bit heavier).
        """
        assert obs.dim() == 2 and obs.shape[1] == self.obs_dim, f"obs shape should be (B, {self.obs_dim})"
        assert flat_params.dim() == 2 and flat_params.shape[0] == obs.shape[0], "batch size mismatch"
        layers = self.unflatten(flat_params)
        x = obs
        for idx, (W, b) in enumerate(layers):
            x = torch.bmm(W, x.unsqueeze(2)).squeeze(2)
            x = x.add_(b)

            is_last = (idx == len(layers) - 1)
            if not is_last:
                x = self._act(x)
            else:
                if self.output_activation:
                    x = self._out_act(x)
                else:
                    x = x
        if self.action_space == "continuous":
            x = torch.tanh(x)
            return self._m + self._a * x
        else:
            return x


    def forward_from_layers(self, obs: Tensor, layers: List[Tuple[Tensor, Tensor]]) -> Tensor:
        """
        Faster forward when the same parameters are reused across many calls (e.g., entire episode).
        `layers` are from `self.unflatten(flat_params)` once per generation.
        """
        x = obs
        for idx, (W, b) in enumerate(layers):
            x = torch.bmm(W, x.unsqueeze(2)).squeeze(2)
            x = x.add_(b)
            is_last = (idx == len(layers) - 1)
            if not is_last:
                x = self._act(x)
            else:
                if self.output_activation:
                    x = self._out_act(x)
                else:
                    x = x
        if self.action_space == "continuous":
            x = torch.tanh(x)
            return self._m + self._a * x
        else:
            return x

    def act_discrete(self, obs: Tensor, flat_params: Tensor) -> Tensor:
        logits = self.forward(obs, flat_params)
        return torch.argmax(logits, dim=-1)
    
    def to(self, device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        # Move continuous-action scalers if present
        if self.action_space == "continuous":
            for attr in ("action_low", "action_high", "_a", "_m"):
                t = getattr(self, attr, None)
                if t is not None:
                    setattr(self, attr, t.to(self.device, self.dtype))
        return self
    
def _resolve_device(requested, default_device, allow_fallback=True):
    """
    requested: 'cpu' | 'cuda' | 'auto' | 'policy' | torch.device | None
    default_device: torch.device (e.g., policy.device)
    """
    if requested in (None, 'policy', 'auto'):
        return default_device
    dev = torch.device(requested) if not isinstance(requested, torch.device) else requested
    print(dev)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        if allow_fallback:
            print("[neuroevo] CUDA requested but not available; falling back to CPU.")
            return torch.device('cpu')
        raise RuntimeError("CUDA requested but not available and allow_fallback=False")
    return dev


class Operator: ...
class Selection(Operator):
    def __init__(self, k: int, minimize: bool = False):
        self.k = k
        self.minimize = minimize
    def select(self, fitness: Tensor, rng: Optional[torch.Generator] = None) -> Tensor:
        raise NotImplementedError

class BestSelection(Selection):
    def select(self, fitness: Tensor, rng: Optional[torch.Generator] = None) -> Tensor:
        _, idx = torch.topk(fitness, self.k, largest=not self.minimize)
        return idx

class RandomSelection(Selection):
    def select(self, fitness: Tensor, rng: Optional[torch.Generator] = None) -> Tensor:
        n = fitness.shape[0]
        return torch.randint(0, n, (self.k,), generator=rng, device=fitness.device)

class TournamentSelection(Selection):
    def __init__(self, k: int, tournament_size: int = 3, minimize: bool = False):
        super().__init__(k=k, minimize=minimize)
        self.tournament_size = tournament_size
    def select(self, fitness: Tensor, rng: Optional[torch.Generator] = None) -> Tensor:
        n = fitness.shape[0]
        idx = torch.randint(0, n, (self.k, self.tournament_size), generator=rng, device=fitness.device)
        fit = fitness[idx]
        if self.minimize:
            winners = torch.argmin(fit, dim=1)
        else:
            winners = torch.argmax(fit, dim=1)
        return idx[torch.arange(self.k, device=fitness.device), winners]

class RouletteWheelSelection(Selection):
    def select(self, fitness: Tensor, rng: Optional[torch.Generator] = None) -> Tensor:
        scores = fitness if not self.minimize else -fitness
        # log-sum-exp style stabilization
        probs = torch.softmax(scores - scores.max(), dim=0)
        return torch.multinomial(probs, num_samples=self.k, replacement=True, generator=rng).to(fitness.device)


class Mutation(Operator):
    def mutate(self, genomes: Tensor, rng: Optional[torch.Generator] = None) -> Tensor:
        raise NotImplementedError

class AdditiveMutation(Mutation):
    def __init__(self, rate: float = 0.1, sigma: float = 0.1):
        self.rate = float(rate)
        self.sigma = float(sigma)
    def mutate(self, genomes: Tensor, rng: Optional[torch.Generator] = None) -> Tensor:
        u = torch.rand(genomes.shape, device=genomes.device, dtype=genomes.dtype, generator=rng)
        mask = u < self.rate
        noise = torch.randn(genomes.shape, device=genomes.device, dtype=genomes.dtype, generator=rng)
        return genomes + (self.sigma * noise) * mask.to(genomes.dtype)

class GlobalMutation(Mutation):
    def __init__(self, rate: float = 0.1, sigma: float = 0.5):
        self.rate = float(rate)
        self.sigma = float(sigma)
    def mutate(self, genomes: Tensor, rng: Optional[torch.Generator] = None) -> Tensor:
        B, _ = genomes.shape
        indiv_mask = (torch.rand((B, 1), device=genomes.device, dtype=genomes.dtype, generator=rng) < self.rate)
        noise = torch.randn(genomes.shape, device=genomes.device, dtype=genomes.dtype, generator=rng) * self.sigma
        return genomes + noise * indiv_mask.to(genomes.dtype)

class Crossover(Operator):
    def crossover(self, parents: Tensor, num_offspring: int, rng: Optional[torch.Generator] = None) -> Tensor:
        raise NotImplementedError

class XPointCrossover(Crossover):
    def __init__(self, x_points: int = 1):
        assert x_points >= 1
        self.x_points = int(x_points)
    def crossover(self, parents: Tensor, num_offspring: int, rng: Optional[torch.Generator] = None) -> Tensor:
        B, P = parents.shape
        if B < 2:
            idx = torch.randint(0, B, (num_offspring,), generator=rng, device=parents.device)
            return parents[idx].clone()
        mom_idx = torch.randint(0, B, (num_offspring,), generator=rng, device=parents.device)
        dad_idx = torch.randint(0, B, (num_offspring,), generator=rng, device=parents.device)
        moms = parents[mom_idx]
        dads = parents[dad_idx]
        if P <= 2:
            mask = torch.zeros((num_offspring, P), device=parents.device, dtype=torch.bool)
            mask[:, ::2] = True
        else:
            pts = torch.randint(1, max(2, P-1), (num_offspring, self.x_points), generator=rng, device=parents.device)
            pts, _ = torch.sort(pts, dim=1)
            flip = torch.zeros((num_offspring, P + 1), device=parents.device, dtype=torch.int8)
            flip[:, 0] = 1  # start with mom
            flip.scatter_(1, pts, 1)
            mask = (torch.cumsum(flip, dim=1)[:, :-1] % 2).to(torch.bool)  # True = mom, False = dad
        children = torch.where(mask, moms, dads)
        return children  # no clone

@dataclass
class EvolutionStats:
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_genome: Tensor

class EnvAdapter:
    def __init__(self, num_envs: int, device: torch.device, obs_dim: int, act_dim: int, discrete: bool):
        self.num_envs = num_envs        # capacity
        self.device = device            # 'cpu' for Gym, 'cuda' for Isaac
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.discrete = discrete

    def reset(self) -> torch.Tensor:
        """Reset all sub-envs. Return obs [num_envs, obs_dim] on self.device."""
        raise NotImplementedError

    def step(self, actions: torch.Tensor):
        """Step all sub-envs with actions (int for discrete, float for continuous).
        Returns (obs, reward, terminated, truncated), all torch tensors on self.device."""
        raise NotImplementedError
class GymAdapter(EnvAdapter):
    def __init__(self, env_fns, discrete: bool, dtype=torch.float32):
        import gymnasium as gym
        self.env = gym.vector.SyncVectorEnv(env_fns)
        obs_space = self.env.single_observation_space
        act_space = self.env.single_action_space
        super().__init__(
            num_envs=self.env.num_envs,
            device=torch.device("cpu"),
            obs_dim=obs_space.shape[0],
            act_dim=(act_space.n if discrete else act_space.shape[0]),
            discrete=discrete,
        )
        self.dtype = dtype

    def reset(self) -> torch.Tensor:
        obs, _ = self.env.reset()
        return torch.as_tensor(obs, device=self.device, dtype=self.dtype)

    def step(self, actions: torch.Tensor):
        # Convert to numpy on CPU for Gym
        if self.discrete:
            act_np = actions.to(torch.int64).cpu().numpy()
        else:
            act_np = actions.to(self.dtype).cpu().numpy()
        obs, r, term, trunc, _ = self.env.step(act_np)
        return (
            torch.as_tensor(obs, device=self.device, dtype=self.dtype),
            torch.as_tensor(r,   device=self.device, dtype=self.dtype),
            torch.as_tensor(term, device=self.device, dtype=torch.bool),
            torch.as_tensor(trunc, device=self.device, dtype=torch.bool),
        )

class IsaacAdapter(EnvAdapter):
    def __init__(self, isaac_env, discrete: bool, dtype=torch.float32):
        # isaac_env must already be configured with num_envs (capacity) on CUDA
        obs_dim = isaac_env.observation_spec().shape[-1]   # adjust to your API
        act_dim = isaac_env.action_spec().shape[-1]        # adjust to your API
        super().__init__(
            num_envs=isaac_env.num_envs,
            device=torch.device("cuda"),
            obs_dim=obs_dim,
            act_dim=act_dim,
            discrete=discrete,
        )
        self.env = isaac_env
        self.dtype = dtype

    def reset(self) -> torch.Tensor:
        # Return CUDA tensor
        return self.env.reset()  # should be [num_envs, obs_dim] on CUDA

    def step(self, actions: torch.Tensor):
        # Pass CUDA tensor directly
        obs, r, term, trunc = self.env.step(actions)
        return obs, r, term, trunc

class Neuroevolution:
    def __init__(
        self,
        policy: BatchedMLPPolicy,
        population_size: int,
        operators: List[List[Operator]],
        evaluator: Callable[[Tensor], Tensor],
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tb_log_dir: Optional[str] = None,                 # <-- NEW
        tb_writer: Optional["SummaryWriter"] = None,      # <-- NEW
    ):
        self.population_size = int(population_size)
        self.operators: List[List[Operator]] = operators

        self.device = policy.device if device is None else torch.device(device)
        self.dtype = dtype if dtype is not None else policy.dtype
        self.policy = policy.to(self.device, self.dtype)

        print(f"[neuroevo] Using device: {self.device}, device: {device}, dtype: {self.dtype}")

        self.seed = seed
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)
        total_k = 0
        for group in self.operators:
            if not group or not isinstance(group[0], Selection):
                raise ValueError("Each operators group must start with a Selection (e.g., BestSelection(k)).")
            total_k += group[0].k
        if total_k != self.population_size:
            raise ValueError(f"Sum of selection k across groups ({total_k}) must equal population_size ({self.population_size}).")
        self.genomes: Tensor = self.policy.sample_population(self.population_size, generator=self.rng)
        self.best_genome: Optional[Tensor] = None
        self.best_fitness: Optional[float] = None
        self.evaluator = evaluator

        self.tb = _TensorboardLogger(log_dir=tb_log_dir, writer=tb_writer) if (tb_log_dir or tb_writer) else None
        setattr(self.policy, "_tb_logger", self.tb)

    def evaluate(self) -> Tuple[Tensor, EvolutionStats]:
        with torch.inference_mode():
            fitness = self.evaluator(self.genomes)
        mean = float(fitness.mean())
        std  = float(fitness.std(unbiased=False))

        best_idx = int(torch.argmax(fitness).item())
        best = float(fitness[best_idx].item())
        if self.best_fitness is None or best > self.best_fitness:
            self.best_fitness = best
            self.best_genome = self.genomes[best_idx].clone()

        stats = EvolutionStats(
            best_fitness=float(self.best_fitness),
            mean_fitness=mean,
            std_fitness=std,
            best_genome=self.best_genome.clone() if self.best_genome is not None else self.genomes[best_idx].clone(),
        )
        return fitness, stats
   
    def _produce_next_generation(self, fitness: Tensor) -> Tensor:
        new_pop = torch.empty_like(self.genomes)
        offset = 0
        for group in self.operators:
            sel: Selection = group[0]  # type: ignore
            k = sel.k
            idx = sel.select(fitness, rng=self.rng).to(self.genomes.device)
            base = self.genomes[idx]
            current = base
            for op in group[1:]:
                if isinstance(op, Crossover):
                    current = op.crossover(current, num_offspring=k, rng=self.rng)
                elif isinstance(op, Mutation):
                    current = op.mutate(current, rng=self.rng)
                else:
                    raise ValueError(f"Unknown operator type in pipeline: {op.__class__.__name__}")
            new_pop[offset:offset+k] = current
            offset += k 
        assert new_pop.shape == self.genomes.shape
        return new_pop

    def step(self) -> EvolutionStats:
        fitness, stats = self.evaluate()
        with torch.no_grad():
            self.genomes = self._produce_next_generation(fitness)
        return stats

    def fit(
        self,
        num_generations: int,
        callback: Optional[Callable[[int, EvolutionStats], None]] = None,
        progress: bool = False,                 # NEW
        progress_desc: str = "Generations",     # NEW
    ) -> EvolutionStats:
        stats = None
        iterator = range(num_generations)
        pbar = None

        if progress and tqdm is not None:
            pbar = tqdm(iterator, desc=progress_desc, unit="gen")
        else:
            pbar = iterator  # plain range

        for g in pbar:
            stats = self.step()
            if callback is not None:
                callback(g, stats)

            if self.tb is not None:
                self.tb.log_rewards(mean_reward=stats.mean_fitness, max_reward=stats.best_fitness)

            # # update a live postfix if using tqdm
            # if progress and tqdm is not None:
            #     # show best & mean for the generation just completed
            #     try:
            #         pbar.set_postfix(best=f"{stats.best_fitness:.3f}",
            #                          mean=f"{stats.mean_fitness:.3f}",
            #                          std=f"{stats.std_fitness:.3f}")
            #     except Exception:
            #         pass
        if self.tb is not None and self.tb.writer is not None:
            self.tb.writer.flush()
        return stats

    @staticmethod
    def make_gym_vector_evaluator(
        env_fn: Callable[[], "gym.Env"],
        policy: BatchedMLPPolicy,
        max_steps: int,
        discrete: bool,
        vec_size: Optional[int] = None,          # capacity of the vectorized env; None => match pop_size each call
        rollouts_per_genome: int = 1,
    ) -> Callable[[Tensor], Tensor]:
        """
        Returns an evaluator(genomes)->fitness that:
          - builds a GymAdapter (CPU) of size = vec_size (or pop_size if None),
          - streams the population in fixed-size chunks (waves) equal to the adapter capacity,
          - pads the last partial wave so we can reuse the same env instance (no rebuilds).
        """
        env_holder = {"adapter": None, "cap": None}

        def _ensure_adapter(capacity: int) -> GymAdapter:
            if env_holder["adapter"] is None or env_holder["cap"] != capacity:
                env_fns = [env_fn for _ in range(capacity)]
                env_holder["adapter"] = GymAdapter(env_fns, discrete=discrete, dtype=policy.dtype)
                env_holder["cap"] = capacity
            return env_holder["adapter"]

        def _evaluator(genomes: Tensor) -> Tensor:
            pop_size = genomes.shape[0]
            cap = vec_size or pop_size
            adapter = _ensure_adapter(cap)

            fitness = torch.empty((pop_size,), device=adapter.device, dtype=policy.dtype)

            def _pad_to(t: torch.Tensor, target: int) -> Tuple[torch.Tensor, int]:
                B = t.shape[0]
                if B == target:
                    return t, B
                deficit = target - B
                pad_idx = torch.arange(deficit, device=t.device) % B
                return torch.cat([t, t[pad_idx]], dim=0), B  # returns (padded, valid_original_len)

            with torch.inference_mode():
                K = int(rollouts_per_genome)
                per_wave = cap if K == 1 else max(1, cap // K)

                for s in range(0, pop_size, per_wave):
                    e = min(s + per_wave, pop_size)
                    chunk = genomes[s:e]

                    if K == 1:
                        padded, valid = _pad_to(chunk, cap)
                        wave_returns = evaluate_population(
                            policy=policy,
                            genomes=padded,
                            env=adapter,
                            max_steps=max_steps,
                            rollouts_per_genome=1,
                            mode="sync",
                        )
                        fitness[s:e] = wave_returns[:valid]
                    else:
                        expanded = chunk.repeat_interleave(K, dim=0)  # [len(chunk)*K, ...]
                        expanded_len = expanded.shape[0]
                        padded, _ = _pad_to(expanded, cap)
                        wave_returns = evaluate_population(
                            policy=policy,
                            genomes=padded,
                            env=adapter,
                            max_steps=max_steps,
                            rollouts_per_genome=1,
                            mode="sync",
                        )
                        trimmed = wave_returns[:expanded_len]
                        averaged = trimmed.view(-1, K).mean(dim=1)
                        fitness[s:e] = averaged

            return fitness

        return _evaluator


    
    def to(self, device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None, reseed: bool = True):
        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        self.genomes = self.genomes.to(self.device, self.dtype)
        if self.best_genome is not None:
            self.best_genome = self.best_genome.to(self.device, self.dtype)
        # Recreate RNG on target device (optionally reseed to keep determinism)
        if reseed and self.seed is not None:
            self.rng = torch.Generator(device=self.device)
            self.rng.manual_seed(self.seed)
        else:
            self.rng = torch.Generator(device=self.device)
        return self
 

def evaluate_population(
    policy: BatchedMLPPolicy,
    genomes: torch.Tensor,
    env: EnvAdapter,
    max_steps: int,
    rollouts_per_genome: int = 1,
    mode: str = "auto",   # "auto" | "sync" | "waves"  (we usually call "sync" on fixed-size chunks)
) -> torch.Tensor:
    """
    Runs one full evaluation and returns fitness per genome.
    The policy is moved to env.device to avoid device mismatch (e.g., _m/_a for continuous scaling).
    """
    # Ensure policy tensors (incl. _m/_a) live on the same device as env
    policy.to(env.device, policy.dtype)

    device = env.device
    dtype = policy.dtype
    pop_size = genomes.shape[0]

    if mode == "auto":
        mode = "sync" if pop_size <= env.num_envs else "waves"
    tb = getattr(policy, "_tb_logger", None)

    def _eval_wave(genome_slice: torch.Tensor) -> torch.Tensor:
        B = genome_slice.shape[0]
        steps_taken = 0
        # Unflatten once per wave on env device
        layers = policy.unflatten(genome_slice.to(device, dtype))
        obs = env.reset()

        # Sanity: env.reset() must match the wave size
        assert obs.shape[0] == B, f"EnvAdapter reset size ({obs.shape[0]}) != wave size ({B})."

        rewards = torch.zeros((B,), device=device, dtype=dtype)
        dones = torch.zeros((B,), device=device, dtype=torch.bool)

        ep_len  = torch.zeros((B,), device=device, dtype=torch.int32)

        with torch.inference_mode():
            for _ in range(max_steps):
                # count a step for all still-alive episodes
                ep_len += (~dones).to(torch.int32)            # <-- NEW
                steps_taken += 1

                logits_or_actions = policy.forward_from_layers(obs, layers)
                actions = logits_or_actions.argmax(dim=-1) if env.discrete else logits_or_actions
                obs, r, term, trunc = env.step(actions)
                step_done = (term | trunc)
                rewards += r * (~dones)
                dones |= step_done
                if torch.all(dones):
                    break

        # --- NEW: log min/mean/max episode length for this wave
        if tb is not None:
            min_len  = float(ep_len.min().item())
            mean_len = float(ep_len.to(torch.float32).mean().item())
            max_len  = float(ep_len.max().item())
            tb.log_episode_length_stats(min_len=min_len, mean_len=mean_len, max_len=max_len)

            # keep your existing timesteps accounting
            tb.add_timesteps(steps_taken * B)

        return rewards

    if rollouts_per_genome > 1:
        # Repeat genomes and average results K-wise
        reps = rollouts_per_genome
        expanded = genomes.repeat_interleave(reps, dim=0)
        if mode == "sync":
            return _eval_wave(expanded).view(pop_size, reps).mean(dim=1)
        else:
            # waves mode: caller should chunk by env.num_envs outside and call mode="sync";
            # but keep a simple fallback here by chunking anyway.
            cap = env.num_envs
            out = torch.empty((expanded.shape[0],), device=device, dtype=dtype)
            for s in range(0, expanded.shape[0], cap):
                e = min(s + cap, expanded.shape[0])
                # Build a temporary view of size (e-s) and expect the adapter to match it
                # (the wrappers below ensure adapter size == chunk size)
                out[s:e] = _eval_wave(expanded[s:e])
            return out.view(pop_size, reps).mean(dim=1)

    if mode == "sync":
        return _eval_wave(genomes)

    # "waves" mode fallback (normally handled by wrappers): chunk manually
    cap = env.num_envs
    out = torch.empty((pop_size,), device=device, dtype=dtype)
    for s in range(0, pop_size, cap):
        e = min(s + cap, pop_size)
        out[s:e] = _eval_wave(genomes[s:e])
    return out

# Isaac-style GPU evaluator adapter
def make_isaac_like_evaluator(
    policy: BatchedMLPPolicy,
    isaac_env,                  # prebuilt Isaac env on CUDA with fixed capacity N
    max_steps: int,
    discrete: bool,
    rollouts_per_genome: int = 1,
) -> Callable[[Tensor], Tensor]:
    """
    Returns an evaluator(genomes)->fitness that:
      - wraps the given Isaac env with IsaacAdapter (CUDA),
      - streams the population through the fixed-capacity env in waves,
      - pads the final partial wave to reuse the same env instance,
      - keeps all tensors on GPU (no host hops).
    """
    adapter = IsaacAdapter(isaac_env, discrete=discrete, dtype=policy.dtype)
    cap = adapter.num_envs  # fixed capacity from Isaac env

    def _pad_to(t: torch.Tensor, target: int) -> Tuple[torch.Tensor, int]:
        B = t.shape[0]
        if B == target:
            return t, B
        deficit = target - B
        pad_idx = torch.arange(deficit, device=t.device) % B
        return torch.cat([t, t[pad_idx]], dim=0), B

    def _evaluator(genomes: Tensor) -> Tensor:
        pop_size = genomes.shape[0]
        fitness = torch.empty((pop_size,), device=adapter.device, dtype=policy.dtype)

        with torch.inference_mode():
            for s in range(0, pop_size, cap):
                e = min(s + cap, pop_size)
                chunk = genomes[s:e].to(adapter.device, policy.dtype)
                padded, valid = _pad_to(chunk, cap)

                wave_returns = evaluate_population(
                    policy=policy,
                    genomes=padded,
                    env=adapter,
                    max_steps=max_steps,
                    rollouts_per_genome=rollouts_per_genome,
                    mode="sync",  # adapter size == cap always
                )
                fitness[s:e] = wave_returns[:valid]
        return fitness

    return _evaluator


class EnvAdapter:
    def __init__(self, num_envs: int, device: torch.device, obs_dim: int, act_dim: int, discrete: bool):
        self.num_envs = num_envs
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.discrete = discrete

        # NEW: whether we can reset only completed env slots
        self.supports_partial_reset: bool = False

    def reset(self) -> torch.Tensor:
        raise NotImplementedError

    def step(self, actions: torch.Tensor):
        raise NotImplementedError

    # NEW: optional per-slot reset (return new obs ONLY for the masked indices)
    def reset_indices(self, done_mask: torch.Tensor) -> Optional[torch.Tensor]:
        return None


class GymAdapter(EnvAdapter):
    def __init__(self, env_fns, discrete: bool, dtype=torch.float32):
        import gymnasium as gym
        self.env = gym.vector.SyncVectorEnv(env_fns)
        obs_space = self.env.single_observation_space
        act_space = self.env.single_action_space
        super().__init__(
            num_envs=self.env.num_envs,
            device=torch.device("cpu"),
            obs_dim=obs_space.shape[0],
            act_dim=(act_space.n if discrete else act_space.shape[0]),
            discrete=discrete,
        )
        self.dtype = dtype

        # NEW: detect partial-reset capability on the vector env
        # Gymnasium >=0.27: SyncVectorEnv.reset_done(indices) exists
        self._has_reset_done = hasattr(self.env, "reset_done")
        self.supports_partial_reset = self._has_reset_done

    def reset(self) -> torch.Tensor:
        obs, _ = self.env.reset()
        return torch.as_tensor(obs, device=self.device, dtype=self.dtype)

    def step(self, actions: torch.Tensor):
        if self.discrete:
            act_np = actions.to(torch.int64).cpu().numpy()
        else:
            act_np = actions.to(self.dtype).cpu().numpy()
        obs, r, term, trunc, _ = self.env.step(act_np)
        return (
            torch.as_tensor(obs,   device=self.device, dtype=self.dtype),
            torch.as_tensor(r,     device=self.device, dtype=self.dtype),
            torch.as_tensor(term,  device=self.device, dtype=torch.bool),
            torch.as_tensor(trunc, device=self.device, dtype=torch.bool),
        )

    def reset_indices(self, done_mask: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._has_reset_done:
            return None
        idx = done_mask.nonzero(as_tuple=False).squeeze(1).cpu().numpy().tolist()
        if len(idx) == 0:
            return torch.empty((0, self.obs_dim), device=self.device, dtype=self.dtype)
        # Gymnasium's reset_done returns observations for the specified env ids
        obs = self.env.reset_done(idx)
        # 'obs' is an array only for those indices in 'idx' and in the same order
        return torch.as_tensor(obs, device=self.device, dtype=self.dtype)


class IsaacAdapter(EnvAdapter):
    def __init__(self, isaac_env, discrete: bool, dtype=torch.float32):
        obs_dim = isaac_env.observation_spec().shape[-1]
        act_dim = isaac_env.action_spec().shape[-1]
        super().__init__(
            num_envs=isaac_env.num_envs,
            device=torch.device("cuda"),
            obs_dim=obs_dim,
            act_dim=act_dim,
            discrete=discrete,
        )
        self.env = isaac_env
        self.dtype = dtype
        # Assume Isaac-style envs can reset a subset efficiently
        self.supports_partial_reset = hasattr(self.env, "reset_indices") or hasattr(self.env, "reset")

    def reset(self) -> torch.Tensor:
        return self.env.reset()

    def step(self, actions: torch.Tensor):
        obs, r, term, trunc = self.env.step(actions)
        return obs, r, term, trunc

    def reset_indices(self, done_mask: torch.Tensor) -> Optional[torch.Tensor]:
        # Try a flexible API: reset only the done slots on device, return the new obs for those slots
        if hasattr(self.env, "reset_indices"):
            return self.env.reset_indices(done_mask)
        # Fallback: API that accepts a boolean or integer mask
        if hasattr(self.env, "reset"):
            return self.env.reset(done_mask)
        return None


# ===========================================
# Async / Steady-State Neuroevolution (NEW)
# ===========================================
@dataclass
class AsyncStats:
    evaluations: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    best_genome: Tensor


class AsyncNeuroevolution:
    """
    Steady-state, asynchronous GA that:
      - Maintains a population and its fitness.
      - Continuously produces offspring using your operator pipeline (selection -> crossover/mutation).
      - Evaluates offspring in a vectorized env pool. As soon as one env finishes an episode,
        the child is scored, replacement happens immediately (no generation barrier), and a new child
        is injected into that slot and reset (partial reset if supported, otherwise wave-by-wave).

    Notes:
      * Uses 'maximize' fitness convention (same as your synchronous path).
      * Parent selection reads from the *evaluated* population.
      * Replacement policy defaults to 'replace worst if child better'.
    """
    def __init__(
        self,
        policy: BatchedMLPPolicy,
        population_size: int,
        operators: List[Operator],  # A single pipeline: [Selection, (Crossover|Mutation)...]
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        minimize: bool = False,     # for selection direction; replacement assumes maximize by default
        tb_log_dir: Optional[str] = None,                 # <-- NEW
        tb_writer: Optional["SummaryWriter"] = None,      # <-- NEW
    ):
        assert len(operators) >= 1 and isinstance(operators[0], Selection), \
            "operators must start with a Selection (e.g., TournamentSelection(k=8))."
        self.policy = policy.to(device or policy.device, dtype or policy.dtype)
        self.device = self.policy.device
        self.dtype = self.policy.dtype
        self.population_size = int(population_size)
        self.operators = operators
        self.minimize = bool(minimize)

        self.rng = torch.Generator(device=self.device)
        self.seed = seed
        if seed is not None:
            self.rng.manual_seed(seed)

        # Population + fitness
        self.genomes: Tensor = self.policy.sample_population(self.population_size, generator=self.rng)
        self.fitness: Tensor = torch.full((self.population_size,), float("-inf"), device=self.device, dtype=self.dtype)
        self._have_eval_mask: torch.Tensor = torch.zeros((self.population_size,), device=self.device, dtype=torch.bool)

        self.best_genome: Optional[Tensor] = None
        self.best_fitness: Optional[float] = None

        self.tb = _TensorboardLogger(log_dir=tb_log_dir, writer=tb_writer) if (tb_log_dir or tb_writer) else None
        setattr(self.policy, "_tb_logger", self.tb)

    # ---------- helpers ----------
    def _select_parents(self) -> Tensor:
        """Use the first Selection op to pull a parent pool from evaluated individuals."""
        sel: Selection = self.operators[0]  # type: ignore
        # Restrict to evaluated indices for meaningful selection
        mask = self._have_eval_mask
        if not torch.any(mask):
            # Fallback: random selection from full population
            idx = torch.randint(0, self.population_size, (max(2, sel.k),), generator=self.rng, device=self.device)
            return idx
        valid_fitness = self.fitness.clone()
        # Mask out non-evaluated by sending them to -inf (or +inf if minimize)
        if self.minimize:
            valid_fitness[~mask] = float("inf")
        else:
            valid_fitness[~mask] = float("-inf")
        # The Selection operator expects a full fitness vector; it will pick sel.k indices
        return sel.select(valid_fitness, rng=self.rng)

    def _produce_offspring(self, num_offspring: int = 1) -> Tensor:
        """
        Produce 'num_offspring' children by:
          - selecting a pool with operators[0] (Selection),
          - applying the rest of the pipeline (Crossover/Mutation).
        """
        parents_idx = self._select_parents()
        parent_pool = self.genomes[parents_idx]  # [k, D]

        current = parent_pool
        for op in self.operators[1:]:
            if isinstance(op, Crossover):
                current = op.crossover(current, num_offspring=num_offspring, rng=self.rng)
            elif isinstance(op, Mutation):
                current = op.mutate(current, rng=self.rng)
            else:
                raise ValueError(f"Unknown operator {op.__class__.__name__} in async pipeline.")
        # Ensure shape [num_offspring, D]
        if current.dim() == 1:
            current = current.unsqueeze(0)
        return current[:num_offspring].contiguous()

    def _replace_if_better(self, child: Tensor, child_fit: float):
        """Replace the worst individual if the child is better (maximize)."""
        # Compute worst among evaluated; if none evaluated, insert greedily into first slot
        if torch.any(self._have_eval_mask):
            evaluated_idx = torch.where(self._have_eval_mask)[0]
            worst_idx = evaluated_idx[torch.argmin(self.fitness[evaluated_idx])]
            worst_fit = float(self.fitness[worst_idx].item())
            if child_fit > worst_fit:
                self.genomes[worst_idx] = child
                self.fitness[worst_idx] = torch.as_tensor(child_fit, device=self.device, dtype=self.dtype)
                self._have_eval_mask[worst_idx] = True
        else:
            self.genomes[0] = child
            self.fitness[0] = torch.as_tensor(child_fit, device=self.device, dtype=self.dtype)
            self._have_eval_mask[0] = True

        # Track global best
        if self.best_fitness is None or child_fit > self.best_fitness:
            self.best_fitness = float(child_fit)
            self.best_genome = child.clone()

    # ---------- bootstrapping ----------
    def bootstrap_population(
        self,
        evaluator: Callable[[Tensor], Tensor],
        init_batches: Optional[int] = None,
    ):
        """
        Quickly evaluate the initial population (optionally in a few batches) to seed selection pressure.
        """
        pop_size = self.population_size
        if init_batches is None or init_batches <= 1:
            with torch.inference_mode():
                fit = evaluator(self.genomes)
            self.fitness[:] = fit.to(self.device, self.dtype)
            self._have_eval_mask[:] = True
        else:
            n = int(math.ceil(pop_size / init_batches))
            with torch.inference_mode():
                for s in range(0, pop_size, n):
                    e = min(s + n, pop_size)
                    self.fitness[s:e] = evaluator(self.genomes[s:e]).to(self.device, self.dtype)
                    self._have_eval_mask[s:e] = True

        # Best tracking
        best_idx = int(torch.argmax(self.fitness).item())
        self.best_fitness = float(self.fitness[best_idx].item())
        self.best_genome = self.genomes[best_idx].clone()

    # ---------- main async runner ----------
    def run_async(
        self,
        env_adapter: EnvAdapter,
        max_steps: int,
        total_evaluations: int,
        rollouts_per_child: int = 1,
        report_every: int = 0,
        callback: Optional[Callable[[AsyncStats], None]] = None,

        # ---- pretty tqdm (NEW, all optional) ----
        progress: bool = False,
        progress_desc: str = "Async GA",
        pbar_colour: Optional[str] = "cyan",         # needs tqdm>=4.66
        pbar_leave: bool = True,
        pbar_smoothing: float = 0.05,
        pbar_bar_format: Optional[str] = (
            "{l_bar}{bar}| {n_fmt}/{total_fmt} • {elapsed}<{remaining} • {rate_fmt} {postfix}"
        ),
        progress_window: int = 200,                  # moving window for recent avg
    ) -> AsyncStats:
        

        # make sure policy tensors live with env
        self.policy.to(env_adapter.device, self.policy.dtype)

        cap = env_adapter.num_envs
        device = env_adapter.device
        dtype = self.policy.dtype
        
        slot_ep_len = torch.zeros((cap,), device=device, dtype=torch.int32)

        ep_min   = float("inf")
        ep_max   = 0.0
        ep_sum   = 0.0
        ep_count = 0
        
        # pool
        pool_genomes = self._produce_offspring(num_offspring=cap).to(device, dtype)
        pool_acc_returns = torch.zeros((cap,), device=device, dtype=dtype)
        pool_acc_counts  = torch.zeros((cap,), device=device, dtype=torch.int32)
        layers = self.policy.unflatten(pool_genomes)
        obs = env_adapter.reset()

        completed = 0
        recent_returns = deque(maxlen=int(max(10, progress_window)))

        # pretty tqdm
        pbar = None
        if progress and tqdm is not None:
            try:
                pbar = tqdm.tqdm(
                    total=total_evaluations,
                    desc=progress_desc,
                    unit="child",
                    dynamic_ncols=True,
                    smoothing=pbar_smoothing,
                    leave=pbar_leave,
                    colour=pbar_colour,                    # silently ignored on older tqdm
                    bar_format=pbar_bar_format,
                    mininterval=0.1,
                )
            except TypeError:
                # tqdm too old for 'colour' or some args
                pbar = tqdm.tqdm(
                    total=total_evaluations,
                    desc=progress_desc,
                    unit="child",
                    dynamic_ncols=True,
                    smoothing=pbar_smoothing,
                    leave=pbar_leave,
                    bar_format=pbar_bar_format,
                    mininterval=0.1,
                )

        def _postfix_update():
            if pbar is None:
                return
            best = (self.best_fitness if self.best_fitness is not None else float("-inf"))
            rec = (sum(recent_returns) / len(recent_returns)) if recent_returns else float("nan")
            # compact, stable postfix
            pbar.set_postfix(
                best=f"{best:.2f}",
                recent=f"{rec:.2f}",
            )

        def _refresh_layers():
            nonlocal layers
            del layers
            layers = self.policy.unflatten(pool_genomes)

        with torch.inference_mode():
            step_budget = max_steps
            while completed < total_evaluations:
                logits_or_actions = self.policy.forward_from_layers(obs, layers)
                actions = logits_or_actions.argmax(dim=-1) if env_adapter.discrete else logits_or_actions
                obs, r, term, trunc = env_adapter.step(actions)
                slot_ep_len += 1

                if self.tb is not None:
                    self.tb.add_timesteps(env_adapter.num_envs)

                done = (term | trunc)
                pool_acc_returns += r

                if torch.any(done):
                    done_idx = torch.where(done)[0]
                    for i in done_idx.tolist():
                        pool_acc_counts[i] += 1
                        ep_len_i = int(slot_ep_len[i].item())      # <-- NEW

                        # multi-rollout per child
                        if pool_acc_counts[i] < rollouts_per_child:
                            if env_adapter.supports_partial_reset:
                                m = (torch.arange(cap, device=device) == i)
                                new_obs = env_adapter.reset_indices(done & m)
                                if new_obs is not None and new_obs.numel() > 0:
                                    obs[i] = new_obs[0]
                            else:
                                obs = env_adapter.reset()
                            slot_ep_len[i] = 0   
                            continue

                        # finalize this child
                        child_return = float(pool_acc_returns[i].item()) / max(1, int(pool_acc_counts[i].item()))
                        recent_returns.append(child_return)
                        ep_min = min(ep_min, float(ep_len_i))
                        ep_max = max(ep_max, float(ep_len_i))
                        ep_sum += float(ep_len_i)
                        ep_count += 1
                        if self.tb is not None and ep_count > 0:
                            self.tb.log_episode_length_stats(
                                min_len=ep_min,
                                mean_len=(ep_sum / ep_count),
                                max_len=ep_max,
                            )
                        child = pool_genomes[i].to(self.device, self.dtype)
                        self._replace_if_better(child, child_return)

                        completed += 1
                        if self.tb is not None:
                            have = torch.any(self._have_eval_mask)
                            mean_fit = float(self.fitness[self._have_eval_mask].mean().item()) if have else float("nan")
                            self.tb.log_rewards(mean_reward=mean_fit,
                                                max_reward=(self.best_fitness if self.best_fitness is not None else float("-inf")))

                        # if pbar is not None:
                        #     pbar.update(1)
                        #     # keep it very lightweight every tick
                        #     _postfix_update()

                        if report_every > 0 and callback is not None and (completed % report_every == 0):
                            have = torch.any(self._have_eval_mask)
                            mean_fit = float(self.fitness[self._have_eval_mask].mean().item()) if have else float("nan")
                            std_fit  = float(self.fitness[self._have_eval_mask].std(unbiased=False).item()) if have else float("nan")
                            callback(AsyncStats(
                                evaluations=completed,
                                best_fitness=(self.best_fitness if self.best_fitness is not None else float("-inf")),
                                mean_fitness=mean_fit,
                                std_fitness=std_fit,
                                best_genome=(self.best_genome.clone() if self.best_genome is not None else child.clone()),
                            ))

                        # inject a new child
                        pool_genomes[i] = self._produce_offspring(num_offspring=1)[0].to(device, dtype)
                        pool_acc_returns[i] = torch.tensor(0.0, device=device, dtype=dtype)
                        pool_acc_counts[i]  = torch.tensor(0, device=device, dtype=torch.int32)
                        slot_ep_len[i] = 0
                    _refresh_layers()

                    if env_adapter.supports_partial_reset:
                        new_obs = env_adapter.reset_indices(done)
                        if new_obs is not None and new_obs.numel() > 0:
                            obs[done] = new_obs
                    else:
                        obs = env_adapter.reset()

                # soft horizon: force-finish occasionally to respect max_steps
                step_budget -= 1
                if step_budget <= 0:
                    done_all = torch.ones((cap,), dtype=torch.bool, device=device)
                    for i in range(cap):
                        pool_acc_counts[i] += 1
                        child_return = float(pool_acc_returns[i].item()) / max(1, int(pool_acc_counts[i].item()))
                        recent_returns.append(child_return)

                        ep_len_i = int(slot_ep_len[i].item())
                        ep_min = min(ep_min, float(ep_len_i))
                        ep_max = max(ep_max, float(ep_len_i))
                        ep_sum += float(ep_len_i)
                        ep_count += 1

                        child = pool_genomes[i].to(self.device, self.dtype)
                        self._replace_if_better(child, child_return)

                        completed += 1
                        if pbar is not None:
                            pbar.update(1)
                            _postfix_update()

                        # new child
                        pool_genomes[i] = self._produce_offspring(num_offspring=1)[0].to(device, dtype)
                        pool_acc_returns[i] = torch.tensor(0.0, device=device, dtype=dtype)
                        pool_acc_counts[i]  = torch.tensor(0, device=device, dtype=torch.int32)
                        slot_ep_len[i] = 0
                    if self.tb is not None and ep_count > 0:
                        self.tb.log_episode_length_stats(
                            min_len=ep_min,
                            mean_len=(ep_sum / ep_count),
                            max_len=ep_max,
                        )   
                    _refresh_layers()
                    if env_adapter.supports_partial_reset:
                        new_obs = env_adapter.reset_indices(done_all)
                        if new_obs is not None and new_obs.numel() > 0:
                            obs[done_all] = new_obs
                    else:
                        obs = env_adapter.reset()
                    step_budget = max_steps

        if pbar is not None:
            pbar.close()

        have = torch.any(self._have_eval_mask)
        mean_fit = float(self.fitness[self._have_eval_mask].mean().item()) if have else float("nan")
        std_fit  = float(self.fitness[self._have_eval_mask].std(unbiased=False).item()) if have else float("nan")

        if self.tb is not None and self.tb.writer is not None:
            self.tb.writer.flush()

        return AsyncStats(
            evaluations=completed,
            best_fitness=(self.best_fitness if self.best_fitness is not None else float("-inf")),
            mean_fitness=mean_fit,
            std_fitness=std_fit,
            best_genome=(self.best_genome.clone() if self.best_genome is not None else self.genomes[0].clone()),
        )


# =========================================================
# Convenience: build an evaluator for bootstrapping (NEW)
# =========================================================
def make_gym_vector_evaluator_async_bootstrap(
    env_fn: Callable[[], "gym.Env"],
    policy: BatchedMLPPolicy,
    max_steps: int,
    discrete: bool,
    vec_size: Optional[int] = None,
    rollouts_per_genome: int = 1,
) -> Callable[[Tensor], Tensor]:
    """
    Same idea as make_gym_vector_evaluator, just a thin wrapper so you can quickly
    seed the async run with initial fitness values.
    """
    return Neuroevolution.make_gym_vector_evaluator(
        env_fn=env_fn,
        policy=policy,
        max_steps=max_steps,
        discrete=discrete,
        vec_size=vec_size,
        rollouts_per_genome=rollouts_per_genome,
    )
