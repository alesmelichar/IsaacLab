"""
Neuroevolution library for gymnasium-compatible environments.

Features:
- Lightweight neural network class with easy access to weights/biases and activations
- Operator base class and a set of selection/mutation/crossover operators
- Vectorized population representation using torch for GPU acceleration
- Support for parallel evaluation using gymnasium Vector Envs (SyncVectorEnv/SubprocVectorEnv)
- Example usages for discrete and continuous action spaces

Notes:
- The module focuses on speed and minimal copies. Parameter flatten/unflatten operations are vectorized.
- The API aims to be flexible: operator_groups is a list of (selection, mutation, crossover, group_size)
  where group_size can be absolute (int) or a fraction of population (float in (0,1]).

"""

from __future__ import annotations

import math
import random
from typing import Callable, List, Sequence, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional gymnasium import; keep it optional for other environments
try:
    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
except Exception:
    gym = None
    SyncVectorEnv = None
    AsyncVectorEnv = None


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#########################
# Utility functions
#########################

def _to_tensor(x, dtype=torch.float32, device=DEVICE):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=dtype, device=device)


def flatten_parameters(param_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Flatten a list of parameter tensors into a 1D tensor."""
    if not param_list:
        return torch.tensor([], device=DEVICE)
    return torch.cat([p.contiguous().view(-1) for p in param_list], dim=0)


def unflatten_parameters(flat: torch.Tensor, shapes: Sequence[Tuple[int, ...]]) -> List[torch.Tensor]:
    """Unflatten a 1D tensor into list of tensors with shapes.

    Args:
        flat: 1D tensor containing all parameters.
        shapes: sequence of shapes for tensors.
    Returns:
        list of tensors reshaped accordingly.
    """
    out = []
    idx = 0
    for s in shapes:
        n = int(np.prod(s))
        chunk = flat[idx: idx + n]
        out.append(chunk.view(*s))
        idx += n
    return out


#########################
# Neural Network class
#########################
class BatchedNet:
    """Batched forward for SimpleNet-like MLP with per-sample parameters."""

    def __init__(self, layer_sizes: Sequence[int], activation=F.tanh, output_activation=None, device=DEVICE):
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        self.device = device

        # shapes and flat slicing plan (weights first, then biases) matching SimpleNet
        self._shapes_w = []
        self._shapes_b = []
        for i in range(len(layer_sizes)-1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            self._shapes_w.append((out_dim, in_dim))
            self._shapes_b.append((out_dim,))
        # precompute flat indices
        self._idx = []
        off = 0
        for s in self._shapes_w + self._shapes_b:
            n = int(np.prod(s))
            self._idx.append((off, off+n, s))
            off += n
        self.genome_dim = off

    def split_params(self, P_params: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        P_params: (P, D) flat genomes (weights then biases).
        Returns:
            Ws: list of (P, out, in)
            Bs: list of (P, out)
        """
        assert P_params.dim() == 2 and P_params.size(1) == self.genome_dim
        Ws, Bs = [], []
        for k, (start, end, shape) in enumerate(self._idx):
            part = P_params[:, start:end].view(P_params.size(0), *shape)
            (Ws if k < len(self._shapes_w) else Bs).append(part)
        return Ws, Bs

    @torch.no_grad()
    def forward_batch(self, x: torch.Tensor, Ws: List[torch.Tensor], Bs: List[torch.Tensor]) -> torch.Tensor:
        """
        x: (B, in) observations
        Ws: [ (B, out, in), ... ]
        Bs: [ (B, out), ... ]
        returns: (B, out_last)
        """
        h = x
        L = len(Ws)
        for i in range(L):
            # h: (B, in_i); W: (B, out_i, in_i); b: (B, out_i)
            # out[b] = W[b] @ h[b] + b[b]
            # -> use bmm: (B,1,in) x (B,in,out) -> (B,1,out) -> squeeze
            h = torch.bmm(h.unsqueeze(1), Ws[i].transpose(1, 2)).squeeze(1) + Bs[i]
            if i < L - 1:
                h = self.activation(h)
            elif self.output_activation is not None:
                h = self.output_activation(h)
        return h

class SimpleNet:
    """Simple fully-connected network with easy access to weights/biases.

    The network stores weights and biases as lists of tensors. Provides
    methods to get/set flattened parameters and to run forward passes.
    """

    def __init__(self, layer_sizes: Sequence[int], activation: Callable = F.tanh, output_activation: Optional[Callable] = None, device: torch.device = DEVICE):
        assert len(layer_sizes) >= 2
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.output_activation = output_activation
        self.device = device

        # Initialize weights and biases
        self.weights: List[torch.Tensor] = []
        self.biases: List[torch.Tensor] = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            # Xavier initialization
            w = torch.randn(out_dim, in_dim, device=self.device) * math.sqrt(2.0 / (in_dim + out_dim))
            b = torch.zeros(out_dim, device=self.device)
            self.weights.append(w)
            self.biases.append(b)

        # Cache shapes for flatten/unflatten
        self._param_shapes = [tuple(w.shape) for w in self.weights] + [tuple(b.shape) for b in self.biases]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = F.linear(x, w, b)
            if i < len(self.weights) - 1:
                x = self.activation(x)
            else:
                if self.output_activation is not None:
                    x = self.output_activation(x)
        return x

    def get_flat_params(self) -> torch.Tensor:
        """Return flattened parameters (weights then biases)."""
        # Guarantee contiguous
        parts = [p.contiguous().view(-1) for p in self.weights + self.biases]
        if parts:
            return torch.cat(parts, dim=0).detach().clone().to(self.device)
        else:
            return torch.empty(0, device=self.device)

    def set_flat_params(self, flat: torch.Tensor):
        """Set parameters from a flat 1D tensor."""
        assert flat.numel() == sum(int(np.prod(s)) for s in self._param_shapes)
        parts = unflatten_parameters(flat, self._param_shapes)
        n_w = len(self.weights)
        for i in range(n_w):
            self.weights[i].data.copy_(parts[i])
        for j in range(len(self.biases)):
            self.biases[j].data.copy_(parts[n_w + j])

    def num_params(self) -> int:
        return sum(int(np.prod(s)) for s in self._param_shapes)

    def cpu(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i].cpu()
        for i in range(len(self.biases)):
            self.biases[i] = self.biases[i].cpu()
        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i].to(device)
        for i in range(len(self.biases)):
            self.biases[i] = self.biases[i].to(device)
        self.device = device


#########################
# Operator base classes
#########################

class Operator:
    """Base operator class. Subclasses should implement __call__.

    The operator is expected to accept and return the flat population tensor (P, genome_dim)
    along with fitnesses if needed.
    """

    def __call__(self, population: torch.Tensor, fitness: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        raise NotImplementedError


#########################
# Mutations
#########################

class AdditiveMutation(Operator):
    """Additive Gaussian mutation applied elementwise with a probability per ge

    Args:
        mutation_rate: probability that any given gene is mutated.
        sigma: standard deviation of additive noise.
    """

    def __init__(self, mutation_rate: float = 0.01, sigma: float = 0.1, device: torch.device = DEVICE):
        self.mutation_rate = float(mutation_rate)
        self.sigma = float(sigma)
        self.device = device

    def __call__(self, population: torch.Tensor, fitness: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # population: (P, D)
        P, D = population.shape
        mask = torch.rand((P, D), device=self.device) < self.mutation_rate
        noise = torch.randn((P, D), device=self.device) * self.sigma
        out = population.clone()
        out[mask] += noise[mask]
        return out


class GlobalMutation(Operator):
    """Replace entire genome with noise scaled by sigma with probability mutation_prob per individual.

    Args:
        mutation_prob: probability an individual is fully replaced (exploration).
        sigma: noise std
    """

    def __init__(self, mutation_prob: float = 0.05, sigma: float = 1.0, device: torch.device = DEVICE):
        self.mutation_prob = float(mutation_prob)
        self.sigma = float(sigma)
        self.device = device

    def __call__(self, population: torch.Tensor, fitness: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        P, D = population.shape
        out = population.clone()
        mask = torch.rand(P, device=self.device) < self.mutation_prob
        if mask.any():
            noise = torch.randn((mask.sum().item(), D), device=self.device) * self.sigma
            out[mask] = out[mask] + noise
        return out


#########################
# Crossovers
#########################

class XPointCrossover(Operator):
    """X-point crossover on flattened genomes.

    Args:
        x: number of crossover points (>=1)
        prob: probability of crossover between a pair
    """

    def __init__(self, x: int = 1, prob: float = 0.9, device: torch.device = DEVICE):
        assert x >= 1
        self.x = x
        self.prob = float(prob)
        self.device = device

    def __call__(self, population: torch.Tensor, fitness: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        # population (P, D)
        P, D = population.shape
        out = population.clone()
        # Shuffle indices to create random pairs
        idx = torch.randperm(P, device=self.device)
        for i in range(0, P - 1, 2):
            if random.random() > self.prob:
                continue
            a = idx[i].item()
            b = idx[i + 1].item()
            points = sorted(random.sample(range(1, D), k=min(self.x, D - 1)))
            # perform x-point crossover
            last = 0
            src_a = out[a].clone()
            src_b = out[b].clone()
            take_from_a = True
            for p in points + [D]:
                if take_from_a:
                    out[a, last:p] = src_a[last:p]
                    out[b, last:p] = src_b[last:p]
                else:
                    out[a, last:p] = src_b[last:p]
                    out[b, last:p] = src_a[last:p]
                take_from_a = not take_from_a
                last = p
        return out


#########################
# Selections
#########################

class BestSelection(Operator):
    """Select top-k individuals (elitism).

    Args:
        k: number of individuals to keep.
    """

    def __init__(self, k: int = 2, device: torch.device = DEVICE):
        self.k = int(k)
        self.device = device

    def __call__(self, population: torch.Tensor, fitness: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        assert fitness is not None
        P = population.shape[0]
        k = min(self.k, P)
        _, idx = torch.topk(fitness, k=k, largest=True)
        return population[idx]


class RandomSelection(Operator):
    """Randomly select n individuals."""

    def __init__(self, n: int = 1, device: torch.device = DEVICE):
        self.n = int(n)
        self.device = device

    def __call__(self, population: torch.Tensor, fitness: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        P = population.shape[0]
        n = min(self.n, P)
        idx = torch.randperm(P, device=self.device)[:n]
        return population[idx]


class TournamentSelection(Operator):
    """Tournament selection: pick n winners by running tournaments of size t."""

    def __init__(self, n: int = 1, t: int = 3, device: torch.device = DEVICE):
        self.n = int(n)
        self.t = int(t)
        self.device = device

    def __call__(self, population: torch.Tensor, fitness: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        assert fitness is not None
        P = population.shape[0]
        winners = []
        for _ in range(self.n):
            participants = torch.randint(0, P, (self.t,), device=self.device)
            best = participants[torch.argmax(fitness[participants])]
            winners.append(best.item())
        return population[torch.tensor(winners, device=self.device)]


class RouletteSelection(Operator):
    """Roulette wheel (fitness-proportionate) selection of n individuals."""

    def __init__(self, n: int = 1, device: torch.device = DEVICE):
        self.n = int(n)
        self.device = device

    def __call__(self, population: torch.Tensor, fitness: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        assert fitness is not None
        # Shift fitness to be positive
        f = fitness.clone()
        minf = float(f.min())
        if minf <= 0:
            f = f - minf + 1e-6
        probs = f / f.sum()
        idx = torch.multinomial(probs, self.n, replacement=True)
        return population[idx]


#########################
# Neuroevolution core
#########################

class Neuroevolution:
    """Main neuroevolution optimizer.

    This class keeps a flattened population of genomes and applies selection, crossover, and mutation
    operators to evolve solutions for a given environment.
    """

    def __init__(
        self,
        env_maker: Callable[[], Any],
        population_size: int = 128,
        net_arch: Sequence[int] = (4, 16, 2),
        device: torch.device = DEVICE,
        operator_groups: Optional[List[Tuple[Operator, Operator, Operator, Any]]] = None,
        elite_fraction: float = 0.02,
        seed: Optional[int] = None,
    ):
        """Initialize the optimizer.

        Args:
            env_maker: callable that returns a fresh environment instance.
            population_size: number of individuals.
            net_arch: network layer sizes (input,...,output)
            operator_groups: list of tuples (selection, mutation, crossover, group_size) defining how
                             offspring are produced. group_size can be int or fraction.
            elite_fraction: fraction of population preserved as elites.
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.env_maker = env_maker
        self.population_size = int(population_size)
        self.device = device
        self.net_arch = list(net_arch)

        # single example net to know parameter shapes
        tmp = SimpleNet(net_arch, device=self.device)
        self.genome_dim = tmp.num_params()
        del tmp

        # initialize flat population (P, D)
        self.population = torch.randn((self.population_size, self.genome_dim), device=self.device) * 0.1

        # operator groups
        if operator_groups is None:
            # default: tournament selection + additive mutation + 1-point crossover
            operator_groups = [
                (TournamentSelection(n=population_size // 2, t=3, device=device), AdditiveMutation(0.02, 0.05, device=device), XPointCrossover(1, 0.9, device=device), population_size // 2),
            ]
        self.operator_groups = operator_groups

        self.elite_fraction = float(elite_fraction)
        self.num_elites = max(1, int(math.ceil(self.elite_fraction * self.population_size)))

    def _make_vector_env(self, n_envs: int):
        if gym is None:
            raise RuntimeError("gymnasium not available. Provide env_maker that returns an env-like object")
        # create list of env maker functions
        fns = [self.env_maker for _ in range(n_envs)]
        # prefer AsyncVectorEnv if available (subprocess) for CPU-bound envs
        try:
            return AsyncVectorEnv(fns)
        except Exception:
            return SyncVectorEnv(fns)
        
    def _evaluate_population_vectorized(self, population: torch.Tensor, episodes: int = 1) -> torch.Tensor:
        """
        Evaluate population with parallel envs and batched forward.
        One episode per individual (repeat 'episodes' if you wish; here we average over them).
        """
        P = population.shape[0]
        fitness = torch.zeros(P, device=self.device)

        # You can tune this; larger is faster until env-CPU becomes bottleneck
        max_batch = min(64, P)

        batched = BatchedNet(self.net_arch, device=self.device)

        for start in range(0, P, max_batch):
            end = min(P, start + max_batch)
            batch_genomes = population[start:end]                      # (B, D)
            B = batch_genomes.size(0)
            Ws, Bs = batched.split_params(batch_genomes)               # per-layer (B, ...)

            # ---- Run 'episodes' times and average
            batch_returns = torch.zeros(B, device=self.device)
            for _ in range(episodes):
                # Try VectorEnv first
                use_vector = (gym is not None) and (SyncVectorEnv is not None)
                if use_vector:
                    try:
                        vec = self._make_vector_env(B)  # AsyncVectorEnv preferred by your helper
                        obs, _ = vec.reset()
                        # Convert to torch
                        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)  # (B, obs_dim)
                        done = torch.zeros(B, dtype=torch.bool, device=self.device)
                        ep_ret = torch.zeros(B, device=self.device)

                        while not bool(done.all()):
                            # Compute actions for *all* envs, but mask-out finished ones
                            logits = batched.forward_batch(obs_t, Ws, Bs)  # (B, act_dim or 1)

                            if hasattr(vec.single_action_space, "n"):
                                # Discrete: pick argmax
                                acts = torch.argmax(logits, dim=-1).to(torch.int64)  # (B,)
                                # Replace actions for finished envs with a valid dummy (e.g., 0)
                                if done.any():
                                    acts = acts.masked_fill(done, 0)
                                actions_np = acts.detach().cpu().numpy()
                            else:
                                # Continuous: tanh to [-1,1], then scale to box if needed
                                cont = torch.tanh(logits)
                                high = torch.as_tensor(vec.single_action_space.high, device=self.device, dtype=torch.float32)
                                low  = torch.as_tensor(vec.single_action_space.low,  device=self.device, dtype=torch.float32)
                                scaled = low + (cont + 1.0) * 0.5 * (high - low)     # (B, act_dim)
                                if done.any():
                                    # keep finished envs frozen with zeros
                                    scaled = torch.where(done.unsqueeze(-1), torch.zeros_like(scaled), scaled)
                                actions_np = scaled.detach().cpu().numpy()

                            # Step vector env
                            next_obs, rews, terms, truncs, infos = vec.step(actions_np)

                            # Update trackers
                            ep_ret += torch.as_tensor(rews, device=self.device, dtype=torch.float32)
                            new_done = torch.as_tensor(terms, device=self.device, dtype=torch.bool) | \
                                      torch.as_tensor(truncs, device=self.device, dtype=torch.bool)
                            done = done | new_done

                            # For envs that finished, their next_obs is undefined until reset; freeze them
                            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
                            # Optional: if your vec env requires manual reset for finished envs before next step,
                            # do it here. Many Gymnasium VectorEnvs allow stepping again after done only after reset;
                            # if you see errors, replace the loop body with:
                            #   idx = torch.where(~done)[0].cpu().numpy().tolist()
                            #   compute actions only for idx, and call vec.step() on those envs via a manual list fallback.

                        vec.close()
                        batch_returns += ep_ret
                    except Exception:
                        # Fallback to manual list of envs if vector env semantics differ
                        use_vector = False

                if not use_vector:
                    # ---- Fallback: list of envs; still one batched forward per step
                    envs = [self.env_maker() for _ in range(B)]
                    obs = []
                    for e in envs:
                        o, _ = e.reset(seed=42)
                        obs.append(o)
                    obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32, device=self.device)
                    done = torch.zeros(B, dtype=torch.bool, device=self.device)
                    ep_ret = torch.zeros(B, device=self.device)

                    while not bool(done.all()):
                        logits = batched.forward_batch(obs_t, Ws, Bs)
                        if hasattr(envs[0].action_space, "n"):
                            acts = torch.argmax(logits, dim=-1).to(torch.int64).detach().cpu().numpy()
                        else:
                            cont = torch.tanh(logits).detach().cpu().numpy()

                        next_obs_list = []
                        for i, env in enumerate(envs):
                            if done[i]:
                                next_obs_list.append(obs_t[i].detach().cpu().numpy())
                                continue
                            if hasattr(env.action_space, "n"):
                                a = int(acts[i])
                            else:
                                # scale if needed
                                a = cont[i]
                                if hasattr(env.action_space, "high"):
                                    high = np.array(env.action_space.high, dtype=np.float32)
                                    low  = np.array(env.action_space.low,  dtype=np.float32)
                                    a = low + (a + 1.0) * 0.5 * (high - low)
                            o, r, term, trunc, _ = env.step(a)
                            ep_ret[i] += float(r)
                            done[i] = bool(term) or bool(trunc)
                            next_obs_list.append(o)
                        obs_t = torch.as_tensor(np.array(next_obs_list), dtype=torch.float32, device=self.device)

                    for e in envs:
                        e.close()
                    batch_returns += ep_ret

            # average over episodes
            fitness[start:end] = batch_returns / float(episodes)

        return fitness
    
    def _evaluate_population(self, population: torch.Tensor, episodes: int = 1, render: bool = False) -> torch.Tensor:
        """Evaluate each individual in the population on the environment and return fitness scores.

        This implementation runs evaluations sequentially by default but can be vectorized externally.
        """
        P = population.shape[0]
        fitnesses = torch.zeros(P, device=self.device)

        # We'll evaluate individuals using a vectorized env in batches for speed
        batch_size = min(8, P)  # configurable

        net = SimpleNet(self.net_arch, device=self.device)

        for start in range(0, P, batch_size):
            end = min(P, start + batch_size)
            batch = population[start:end]
            bsize = end - start
            # for each in batch evaluate episodes times
            batch_f = torch.zeros(bsize, device=self.device)
            for i in range(bsize):
                genome = batch[i]
                net.set_flat_params(genome)
                total_reward = 0.0
                for ep in range(episodes):
                    env = self.env_maker()
                    obs, _ = env.reset()
                    done = False
                    ep_reward = 0.0
                    while True:
                        x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                        out = net.forward(x).squeeze(0)
                        # map out to action depending on env action space
                        if hasattr(env, "action_space"):
                            a_space = env.action_space
                            if hasattr(a_space, "n"):
                                # discrete
                                action = int(torch.argmax(out).item())
                            else:
                                # continuous
                                # assume action in [-1,1]
                                action = out.cpu().numpy()
                                # scale if necessary
                                if hasattr(a_space, "high"):
                                    high = np.array(a_space.high, dtype=np.float32)
                                    low = np.array(a_space.low, dtype=np.float32)
                                    # tanh to -1..1
                                    action = np.tanh(action)
                                    action = low + (action + 1.0) * 0.5 * (high - low)
                                else:
                                    action = np.tanh(action)
                        else:
                            # fallback: use continuous output
                            action = out.cpu().numpy()

                        obs, reward, terminated, truncated, info = env.step(action)
                        ep_reward += float(reward)
                        if terminated or truncated:
                            break
                    total_reward += ep_reward
                    env.close()
                batch_f[i] = float(total_reward / episodes)
            fitnesses[start:end] = batch_f
        return fitnesses

    def step(self, generations: int = 1, episodes: int = 1, verbose: bool = True):
        """Run evolution for given number of generations."""
        for gen in range(1, generations + 1):
            fitness = self._evaluate_population_vectorized(self.population, episodes=episodes)
            # sort population by fitness
            sorted_fitness, idx = torch.sort(fitness, descending=True)
            pop_sorted = self.population[idx]

            # elites carried over
            elites = pop_sorted[: self.num_elites]

            # build new population
            new_pop = [elites]
            remaining = self.population_size - self.num_elites

            # allocate group sizes
            group_allocs = []
            total_frac = 0.0
            for sel, mut, cross, group_size in self.operator_groups:
                if isinstance(group_size, float):
                    alloc = int(group_size * remaining)
                else:
                    alloc = int(group_size)
                group_allocs.append((sel, mut, cross, alloc))
            # adjust rounding
            total_alloc = sum(g[3] for g in group_allocs)
            if total_alloc < remaining:
                group_allocs[0] = (group_allocs[0][0], group_allocs[0][1], group_allocs[0][2], group_allocs[0][3] + (remaining - total_alloc))

            for sel, mut, cross, alloc in group_allocs:
                if alloc <= 0:
                    continue
                # selection returns selected parents (M, D)
                parents = sel(pop_sorted, sorted_fitness)
                # if parents less than alloc, sample with replacement
                if parents.shape[0] == 0:
                    parents = pop_sorted[torch.randperm(pop_sorted.shape[0], device=self.device)[:max(1, alloc)]]
                # expand parents to have size alloc (simple sampling)
                idxs = torch.randint(0, parents.shape[0], (alloc,), device=self.device)
                children = parents[idxs]
                # crossover
                if cross is not None:
                    children = cross(children)
                # mutation
                if mut is not None:
                    children = mut(children)
                new_pop.append(children)

            new_pop = torch.cat(new_pop, dim=0)
            # safety: trim or pad
            if new_pop.shape[0] > self.population_size:
                new_pop = new_pop[: self.population_size]
            elif new_pop.shape[0] < self.population_size:
                pad = torch.randn((self.population_size - new_pop.shape[0], self.genome_dim), device=self.device) * 0.01
                new_pop = torch.cat([new_pop, pad], dim=0)

            self.population = new_pop

            if verbose:
                best = float(sorted_fitness[0].item())
                mean = float(fitness.mean().item())
                print(f"Gen {gen}: best={best:.3f}, mean={mean:.3f}")

    def get_best(self) -> SimpleNet:
        #fitness = self._evaluate_population(self.population, episodes=1)
        fitness = self._evaluate_population_vectorized(self.population, episodes=1)
        best_idx = int(torch.argmax(fitness).item())
        genome = self.population[best_idx]
        net = SimpleNet(self.net_arch, device=self.device)
        net.set_flat_params(genome)
        return net


#########################
# Examples
#########################

if __name__ == "__main__":
    # Example 1: CartPole-v1 (discrete action)
    if gym is not None:
        def make_cartpole():
            return gym.make("CartPole-v1")

        evo = Neuroevolution(
            env_maker=make_cartpole,
            population_size=64,
            net_arch=(4, 32, 2),
            operator_groups=[
                (TournamentSelection(n=32, t=3), AdditiveMutation(0.02, 0.1), XPointCrossover(1, 0.9), 32),
                (RandomSelection(n=32), GlobalMutation(0.02, 0.5), XPointCrossover(2, 0.7), 32),
            ],
            elite_fraction=0.05,
        )

        print("Evolving on CartPole-v1... (this may take time)")
        evo.step(generations=10, episodes=1)
        best_net = evo.get_best()
        print("Got best policy for CartPole. Test rollout:")
        env = make_cartpole()
        obs, _ = env.reset()
        done = False
        tot = 0.0
        while True:
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            out = best_net.forward(x).squeeze(0)
            action = int(torch.argmax(out).item())
            obs, reward, terminated, truncated, info = env.step(action)
            tot += reward
            if terminated or truncated:
                break
        print(f"Test reward: {tot}")

        env.close()

    # Example 2: Pendulum-v1 (continuous)
    if gym is not None:
        def make_pendulum():
            return gym.make("Pendulum-v1")

        evo_c = Neuroevolution(
            env_maker=make_pendulum,
            population_size=64,
            net_arch=(3, 32, 1),
            operator_groups=[
                (TournamentSelection(n=32, t=3), AdditiveMutation(0.05, 0.2), XPointCrossover(2, 0.8), 32),
                (RandomSelection(n=32), GlobalMutation(0.05, 1.0), XPointCrossover(3, 0.6), 32),
            ],
            elite_fraction=0.05,
        )

        print("Evolving on Pendulum-v1... (this may take time)")
        evo_c.step(generations=8, episodes=1)
        best_pend = evo_c.get_best()
        print("Got best policy for Pendulum. Test rollout:")
        env = make_pendulum()
        obs, _ = env.reset()
        tot = 0.0
        while True:
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            out = best_pend.forward(x).squeeze(0)
            action = np.tanh(out.cpu().numpy())
            obs, reward, terminated, truncated, info = env.step(action)
            tot += reward
            if terminated or truncated:
                break
        print(f"Test reward (Pendulum): {tot}")
        env.close()

    print("Done")
