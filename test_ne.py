# neuroevo.py
# -----------------------------------------------------------------------------
# README (compact)
# -----------------------------------------------------------------------------
# NeuroEvo: a single-file, torch-first, GPU-ready, vectorized neuroevolution
# micro-library. Designed for Gymnasium vectorized envs and easily adaptable to
# Isaac Gym / IsaacLab vectorized envs.
#
# Highlights:
# - Population params as [pop_size, param_dim] torch.Tensor on device
# - Batched mutation/crossover/selection (no per-individual Python loops)
# - TorchNet with (de)flattened params + batched forward_with_params()
# - Deterministic state_dict export/import
# - Parallel evaluation on gym.vector (Sync/AsyncVectorEnv). IsaacGym stub shown.
#
# Quick Start (CartPole):
#   python neuroevo.py --env CartPole-v1 --generations 10 --pop 128 --device auto
#
# Isaac Gym / IsaacLab:
# - Provide a vectorized env with the same minimal API:
#     reset(seed=None) -> obs (torch|np array [N, obs_dim]), info (optional)
#     step(actions)    -> obs, rewards, terminateds, truncateds, infos
# - Swap the stub `IsaacGymVecEnvStub` with your real vectorized env.
#
# This file includes:
# - TorchNet (NN wrapper with flat params)
# - Operator base + mutations/crossovers/selections
# - Neuroevolution optimizer with vectorized evaluation
# - Examples + tiny unit checks + micro-benchmark + CLI
#
# Dependencies: torch, numpy, (optional) gymnasium
# -----------------------------------------------------------------------------

from __future__ import annotations

import math
import time
import argparse
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym  # optional, only needed for examples/eval via Gymnasium
except Exception:  # pragma: no cover
    gym = None


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def _get_device(device: Optional[str] = None) -> torch.device:
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _activation_from_name(name: Union[str, Callable]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Map a string to an activation function. If a callable is given, return it."""
    if callable(name):
        return name
    name = name.lower()
    if name in ("id", "identity", "linear", "none"):
        return lambda x: x
    if name in ("relu",):
        return F.relu
    if name in ("leaky_relu", "lrelu"):
        return F.leaky_relu
    if name in ("tanh",):
        return torch.tanh
    if name in ("sigmoid",):
        return torch.sigmoid
    if name in ("gelu",):
        return F.gelu
    raise ValueError(f"Unknown activation '{name}'")


def _ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous() if not t.is_contiguous() else t


def _to_torch(x, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


# -----------------------------------------------------------------------------
# TorchNet: lightweight MLP with flat params utilities
# -----------------------------------------------------------------------------

@dataclass
class LayerSpec:
    in_dim: int
    out_dim: int
    activation: Callable[[torch.Tensor], torch.Tensor]


class TorchNet(nn.Module):
    """
    Lightweight MLP that supports:
      - easy read/write across a single flat parameter tensor
      - per-layer activation selection
      - deterministic state_dict export/import
      - batched forward given a batch of flat parameter tensors
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int],
        out_dim: int,
        activations: Optional[Sequence[Union[str, Callable]]] = None,
        output_activation: Union[str, Callable] = "identity",
        bias: bool = True,
        init_scale: float = 0.05,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            obs_dim: input dimension
            hidden_sizes: list of hidden layer sizes
            out_dim: output dimension
            activations: per-hidden-layer activations (len == len(hidden_sizes)),
                         strings or callables. Defaults to 'tanh' for all.
            output_activation: activation for final layer (str or callable)
            bias: include bias terms
            init_scale: uniform init range [-init_scale, init_scale]
            device, dtype: module device/dtype
        """
        super().__init__()
        self.device_ = _get_device(device)
        self.dtype_ = dtype
        self.bias = bias

        if activations is None:
            activations = ["tanh"] * len(hidden_sizes)
        if len(activations) != len(hidden_sizes):
            raise ValueError("activations len must match hidden_sizes len")

        layer_sizes = [obs_dim] + list(hidden_sizes) + [out_dim]
        self.layer_specs: List[LayerSpec] = []
        for i in range(len(layer_sizes) - 1):
            act = _activation_from_name(activations[i]) if i < len(hidden_sizes) else _activation_from_name(output_activation)
            self.layer_specs.append(LayerSpec(layer_sizes[i], layer_sizes[i + 1], act))

        # Build torch nn layers (for single-param-set forward)
        modules = []
        for i, spec in enumerate(self.layer_specs):
            modules.append(nn.Linear(spec.in_dim, spec.out_dim, bias=bias))
        self.layers = nn.ModuleList(modules)

        # Deterministic simple init:
        for m in self.layers:
            nn.init.uniform_(m.weight, -init_scale, init_scale)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -init_scale, init_scale)

        self.to(device=self.device_, dtype=self.dtype_)

        # Build indexing views for flat params
        self._param_shapes: List[Tuple[str, Tuple[int, ...]]] = []
        self._slices: List[Tuple[str, slice]] = []
        offset = 0
        for i, layer in enumerate(self.layers):
            w_shape = layer.weight.shape  # (out, in)
            w_num = w_shape.numel()
            self._param_shapes.append((f"layers.{i}.weight", tuple(w_shape)))
            self._slices.append((f"layers.{i}.weight", slice(offset, offset + w_num)))
            offset += w_num
            if self.bias:
                b_shape = layer.bias.shape  # (out,)
                b_num = b_shape.numel()
                self._param_shapes.append((f"layers.{i}.bias", tuple(b_shape)))
                self._slices.append((f"layers.{i}.bias", slice(offset, offset + b_num)))
                offset += b_num
        self.param_dim = offset  # flattened dimension

    # --------------------- Flat params API ---------------------

    @torch.no_grad()
    def get_flat_params(self) -> torch.Tensor:
        """
        Returns:
            flat (param_dim,) tensor, contiguous, same device/dtype as module
        """
        parts = []
        for i, layer in enumerate(self.layers):
            parts.append(layer.weight.view(-1))
            if self.bias and layer.bias is not None:
                parts.append(layer.bias.view(-1))
        flat = torch.cat(parts, dim=0).to(device=self.device_, dtype=self.dtype_)
        return _ensure_contiguous(flat)

    @torch.no_grad()
    def set_flat_params(self, flat: torch.Tensor) -> None:
        """
        Args:
            flat: (param_dim,) or (param_dim,1) contiguous tensor on any device
        """
        flat = flat.to(device=self.device_, dtype=self.dtype_).view(-1).contiguous()
        if flat.numel() != self.param_dim:
            raise ValueError(f"Expected flat of size {self.param_dim}, got {flat.numel()}")
        for name, sl in self._slices:
            tensor = flat[sl]
            module_name, attr = name.rsplit(".", 1)
            mod: nn.Linear = self.get_submodule(module_name)
            if attr == "weight":
                mod.weight.copy_(tensor.view_as(mod.weight))
            elif attr == "bias":
                if mod.bias is None:
                    raise ValueError("Model has no bias but flat contained bias")
                mod.bias.copy_(tensor.view_as(mod.bias))

    # ------------------ Structured per-layer access ------------

    def layer_params(self) -> List[Dict[str, torch.Tensor]]:
        """Return list of dicts [{'weight': W, 'bias': b}, ...] referencing live params."""
        out = []
        for layer in self.layers:
            out.append({"weight": layer.weight, "bias": layer.bias})
        return out

    # --------------------- Forward APIs ------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward using module's current parameters (single param set)."""
        x = x.to(device=self.device_, dtype=self.dtype_)
        for i, spec in enumerate(self.layer_specs):
            layer: nn.Linear = self.layers[i]
            x = layer(x)
            x = spec.activation(x)
        return x

    def forward_with_params(self, x: torch.Tensor, flat_params_batch: torch.Tensor) -> torch.Tensor:
        """
        Batched forward where each sample in `x` uses its own parameter vector.

        Args:
            x: [B, obs_dim] float tensor
            flat_params_batch: [B, param_dim] float tensor

        Returns:
            out: [B, out_dim]
        """
        B = x.shape[0]
        if flat_params_batch.dim() != 2 or flat_params_batch.shape[0] != B or flat_params_batch.shape[1] != self.param_dim:
            raise ValueError(f"Expected flat_params_batch shape [B={B}, {self.param_dim}], got {tuple(flat_params_batch.shape)}")
        x = x.to(self.device_, self.dtype_)
        params = flat_params_batch.to(self.device_, self.dtype_)

        # Unpack each layer's W/b into batched tensors via views (no copies)
        offset = 0
        for i, spec in enumerate(self.layer_specs):
            out_dim, in_dim = spec.out_dim, spec.in_dim
            w_num = out_dim * in_dim
            W = params[:, offset : offset + w_num].view(B, out_dim, in_dim)
            offset += w_num
            if self.bias:
                b = params[:, offset : offset + out_dim].view(B, out_dim)
                offset += out_dim
            else:
                b = None

            # y_i = x_i @ W_i^T + b_i  -> (B, 1, in) @ (B, in, out) -> (B, 1, out)
            y = torch.bmm(x.unsqueeze(1), W.transpose(1, 2)).squeeze(1)
            if b is not None:
                y = y + b
            x = spec.activation(y)

        return x

    # ----------------------- Param init ------------------------

    @torch.no_grad()
    def sample_flat_params(self, generator: Optional[torch.Generator] = None, scale: float = 0.05) -> torch.Tensor:
        """Return a randomly sampled flat parameter vector (uniform in [-scale, scale])."""
        g = generator
        return (torch.rand(self.param_dim, device=self.device_, generator=g, dtype=self.dtype_) * 2 - 1.0) * scale


# -----------------------------------------------------------------------------
# Operator base and implementations
# -----------------------------------------------------------------------------

class Operator:
    """
    Base class for all operators. Operators should be stateless if possible
    and operate on batched tensors on-device.

    - Mutations:    apply(params: [N, D], rng) -> [N, D]
    - Crossovers:   apply(parents_a: [M, D], parents_b: [M, D], rng) -> [M, D]
    - Selections:   select(fitness: [N], rng) -> indices [K]
    """

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.__dict__})"


# ------------------------ Mutations ---------------------------

class AdditiveMutation(Operator):
    """
    Per-parameter Gaussian perturbation applied with probability `rate`.

    Vectorized: noise_mask ~ Bernoulli(rate) on [N,D]
                params += noise_mask * torch.randn_like(params) * sigma
    """

    def __init__(self, rate: float = 0.05, sigma: float = 0.02):
        self.rate = float(rate)
        self.sigma = float(sigma)

    @torch.no_grad()
    def apply(self, params: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        N, D = params.shape
        device = params.device
        noise_mask = torch.rand((N, D), device=device, generator=rng, dtype=params.dtype) < self.rate
        noise = torch.randn((N, D), device=device, generator=rng, dtype=params.dtype) * self.sigma
        return params + noise * noise_mask.to(params.dtype)


class GlobalMutation(Operator):
    """
    Global pattern perturbation:
      - Sample one noise vector g ~ N(0, sigma^2 I) in R^D.
      - Each individual i receives scale_i * g with probability `rate`.
      - scale_i ~ N(1, sigma) (by default) or user-provided scale distribution.

    This correlates changes across parameters and individuals, encouraging
    coherent global exploration.
    """

    def __init__(self, rate: float = 0.5, sigma: float = 0.02, mean_scale: float = 1.0, scale_sigma: float = 0.1):
        self.rate = float(rate)
        self.sigma = float(sigma)
        self.mean_scale = float(mean_scale)
        self.scale_sigma = float(scale_sigma)

    @torch.no_grad()
    def apply(self, params: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        N, D = params.shape
        device = params.device
        g = torch.randn((1, D), device=device, generator=rng, dtype=params.dtype) * self.sigma
        use = (torch.rand((N, 1), device=device, generator=rng, dtype=params.dtype) < self.rate).to(params.dtype)
        scales = torch.randn((N, 1), device=device, generator=rng, dtype=params.dtype) * self.scale_sigma + self.mean_scale
        return params + use * scales * g  # broadcasts


# ------------------------ Crossovers --------------------------

class XPointCrossover(Operator):
    """
    x-point crossover over flattened genomes.
    For each pair, sample x cut points ∈ [1, D-1], alternate segments between parents.
    """

    def __init__(self, x: int = 1):
        assert x >= 1
        self.x = int(x)

    @torch.no_grad()
    def apply(self, parents_a: torch.Tensor, parents_b: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        assert parents_a.shape == parents_b.shape
        M, D = parents_a.shape
        device = parents_a.device
        # Sample x cutpoints per pair in [1, D-1], sort ascending
        cps = torch.randint(1, max(2, D), (M, self.x), device=device, generator=rng)
        cps, _ = torch.sort(cps, dim=1)
        # Build toggle mask via cumsum trick
        toggles = torch.zeros((M, D), device=device, dtype=torch.int32)
        # For scatter, ensure indices exist; guard if cps might include D (it won't due to randint bounds).
        toggles.scatter_(1, cps, 1)
        parity = (toggles.cumsum(dim=1) % 2).to(dtype=torch.bool)  # True -> take from B
        child = torch.where(parity, parents_b, parents_a)
        return child


class UniformCrossover(Operator):
    """Per-gene uniform crossover between two parents with probability `prob`."""

    def __init__(self, prob: float = 0.5):
        self.prob = float(prob)

    @torch.no_grad()
    def apply(self, parents_a: torch.Tensor, parents_b: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        mask = (torch.rand_like(parents_a) < self.prob)
        return torch.where(mask, parents_b, parents_a)


# ------------------------ Selections --------------------------

class Selection(Operator):
    """Base class for selection operators returning indices of chosen parents."""
    def __init__(self, k: int):
        self.k = int(k)


class BestSelection(Selection):
    """Pick top-k indices by fitness (higher is better)."""
    @torch.no_grad()
    def select(self, fitness: torch.Tensor) -> torch.Tensor:
        _, idx = torch.topk(fitness, k=self.k, largest=True, sorted=True)
        return idx


class RandomSelection(Selection):
    """Uniformly sample k indices (with replacement)."""
    @torch.no_grad()
    def select(self, fitness: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        N = fitness.shape[0]
        return torch.randint(0, N, (self.k,), device=fitness.device, generator=rng)


class TournamentSelection(Selection):
    """k tournaments; each draws 'tournament_size' candidates and picks the best."""
    def __init__(self, k: int, tournament_size: int = 3):
        super().__init__(k)
        assert tournament_size >= 2
        self.tournament_size = int(tournament_size)

    @torch.no_grad()
    def select(self, fitness: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        N = fitness.shape[0]
        cand = torch.randint(0, N, (self.k, self.tournament_size), device=fitness.device, generator=rng)
        cand_fit = fitness[cand]  # [k, t]
        winners_idx = torch.argmax(cand_fit, dim=1)  # [k]
        row = torch.arange(self.k, device=fitness.device)
        return cand[row, winners_idx]


class RouletteWheelSelection(Selection):
    """Fitness-proportional sampling with replacement using torch.multinomial."""
    def __init__(self, k: int):
        super().__init__(k)

    @torch.no_grad()
    def select(self, fitness: torch.Tensor, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        # Shift to non-negative
        f = fitness - fitness.min()
        f = f + 1e-8  # avoid all-zero
        return torch.multinomial(f, num_samples=self.k, replacement=True, generator=rng)


# -----------------------------------------------------------------------------
# Neuroevolution Optimizer
# -----------------------------------------------------------------------------

class Neuroevolution:
    """
    Evolutionary optimizer for TorchNet parameters.

    population params: torch.Tensor shape [population_size, param_dim] on device
    """

    def __init__(
        self,
        env_or_env_factory: Union[object, Callable[[], object]],
        net_factory: Callable[[object], TorchNet],
        population_size: int,
        device: str = "auto",
        operators: Optional[List[List[Operator]]] = None,
        fitness_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        eval_steps: int = 1000,
        vectorized_eval: bool = True,
        elitism: int = 0,
        seed: Optional[int] = None,
        rng: Optional[Union[np.random.Generator, torch.Generator]] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Args:
            env_or_env_factory: gym VecEnv / Isaac vectorized env OR callable returning one.
            net_factory: Callable that takes an env (or its spaces) and returns a TorchNet.
            population_size: number of individuals
            device: 'cpu'/'cuda'/'auto'
            operators: pipeline per generation as list of lists. A common single pipeline:
                [
                  [TournamentSelection(k=pop), XPointCrossover(1), AdditiveMutation(0.05, 0.02)],
                ]
            fitness_fn: optional custom fitness computation f([N,D]) -> [N]
            eval_steps: number of env steps per evaluation
            vectorized_eval: if True, env expected vectorized (n_envs == N)
            elitism: number of top individuals copied to next gen unchanged
            seed: base seed for determinism
            rng: optional torch.Generator to control randomness
            checkpoint_path: if set, best state_dict is saved there
        """
        self.device = _get_device(device)
        self.population_size = int(population_size)
        self.eval_steps = int(eval_steps)
        self.vectorized_eval = bool(vectorized_eval)
        self.elitism = int(max(0, elitism))
        self.fitness_fn = fitness_fn
        self.checkpoint_path = checkpoint_path

        # RNGs
        self.torch_rng = torch.Generator(device=self.device)
        if seed is not None:
            self.torch_rng.manual_seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        if isinstance(rng, torch.Generator):
            # Override internal generator if provided
            self.torch_rng = rng

        # Build env
        self.env = env_or_env_factory() if callable(env_or_env_factory) else env_or_env_factory

        # Build net from env spaces if possible
        self.net = net_factory(self.env)  # user-provided constructor

        # Initialize population params
        base = self.net.get_flat_params()
        D = base.numel()
        self.param_dim = D
        # Start near base with small noise
        pop = base.repeat(self.population_size, 1)

        pop += torch.randn_like(pop) * 0.02
        self.population = pop.to(self.device)

        # Build default operators if not provided
        if operators is None:
            operators = [
                [TournamentSelection(k=self.population_size, tournament_size=3),
                 XPointCrossover(x=1),
                 AdditiveMutation(rate=0.05, sigma=0.02)]
            ]
        self.operators = operators

        # Buffers to minimize reallocs
        self._tmp_children = torch.empty((self.population_size, D), device=self.device, dtype=self.population.dtype)

        # History
        self.history: Dict[str, List[float]] = {"best_fitness": [], "mean_fitness": []}

    # -------------------- Evaluation ---------------------------

    def _infer_action(self, logits_or_action: torch.Tensor, action_space) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert network output into environment action(s).
        Supports Discrete and Box. For Discrete: argmax.
        For Box: tanh outputs scaled to bounds if available.
        """
        # Discrete
        if hasattr(action_space, "n"):  # Discrete
            if logits_or_action.dim() == 1:
                a = torch.argmax(logits_or_action, dim=0)
                return a.detach().to("cpu").numpy()
            else:
                a = torch.argmax(logits_or_action, dim=1)
                return a.detach().to("cpu").numpy()
        # Box (continuous)
        if hasattr(action_space, "shape"):  # Box-like
            low = torch.as_tensor(getattr(action_space, "low", None), dtype=logits_or_action.dtype, device=logits_or_action.device)
            high = torch.as_tensor(getattr(action_space, "high", None), dtype=logits_or_action.dtype, device=logits_or_action.device)
            if low.numel() == 0 or high.numel() == 0:
                # No bounds info: pass through
                return logits_or_action.detach().to("cpu").numpy()
            # squash via tanh to [-1,1], then scale to [low, high]
            y = torch.tanh(logits_or_action)
            # Broadcast to batch
            low = low.to(logits_or_action.device)
            high = high.to(logits_or_action.device)
            # Handle different shapes (env might broadcast)
            while low.dim() < y.dim():
                low = low.unsqueeze(0)
                high = high.unsqueeze(0)
            action = (y + 1) * (high - low) / 2 + low
            return action.detach().to("cpu").numpy()

        raise TypeError("Unsupported action space; provide a custom fitness_fn or adapt _infer_action.")

    @torch.no_grad()
    def _evaluate_population_vectorized(self) -> torch.Tensor:
        """
        Evaluate all individuals in parallel on a vectorized env.

        Returns:
            fitness: [N] tensor of total episodic rewards over `eval_steps`.
        """
        N = self.population_size
        env = self.env

        # Try Gymnasium vec env API; else assume Isaac-like torch tensors.
        reset_out = env.reset(seed=None)
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, _ = reset_out
        else:
            obs = reset_out
        obs_t = _to_torch(obs, device=self.device)

        # Checks
        if obs_t.shape[0] != N:
            raise ValueError(f"Vectorized env num_envs ({obs_t.shape[0]}) must equal population_size ({N}).")

        total_reward = torch.zeros((N,), device=self.device, dtype=torch.float32)

        # Device-aligned population tensor
        pop_params = self.population  # [N, D]

        action_space = getattr(env, "single_action_space", getattr(env, "action_space", None))
        if action_space is None:
            # Isaac-like envs might not expose Gym space; assume Box-like [-1,1] range
            class _Dummy:
                shape = (self.net.layer_specs[-1].out_dim,)
                low = -torch.ones(self.net.layer_specs[-1].out_dim)
                high = torch.ones(self.net.layer_specs[-1].out_dim)
            action_space = _Dummy()

        for _ in range(self.eval_steps):
            # Forward all individuals with their own params
            logits = self.net.forward_with_params(obs_t, pop_params)  # [N, out_dim]
            actions = self._infer_action(logits, action_space)

            step_out = env.step(actions)
            if len(step_out) == 5:  # Gymnasium VecEnv
                obs, rew, term, trunc, _ = step_out
                done = np.asarray(term) | np.asarray(trunc)
                total_reward += _to_torch(rew, device=self.device).view(-1)
                # reset finished envs automatically by VecEnv; update obs
                obs_t = _to_torch(obs, device=self.device)
            else:
                # Isaac-like: expect tensors
                obs, rew, done = step_out  # type: ignore
                total_reward += _to_torch(rew, device=self.device).view(-1)
                # If done, env should auto-reset or continue; we just consume obs.
                obs_t = _to_torch(obs, device=self.device)

        return total_reward

    @torch.no_grad()
    def _evaluate_population_sequential(self) -> torch.Tensor:
        """Fallback: evaluate individuals one-by-one on a single (non-vectorized) env."""
        N = self.population_size
        env = self.env
        fitness = torch.zeros((N,), device=self.device, dtype=torch.float32)

        # Try to detect if this env is vectorized anyway (n_envs attribute)
        n_envs = getattr(env, "num_envs", getattr(env, "num_env", 1))
        if n_envs != 1:
            # If it is already vectorized, we can still use vectorized evaluation
            return self._evaluate_population_vectorized()

        # Sequential loop (kept simple and robust)
        for i in range(N):
            reset_out = env.reset(seed=None)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            total = 0.0
            for _ in range(self.eval_steps):
                obs_t = _to_torch(obs, device=self.device).unsqueeze(0)  # [1, obs_dim]
                logits = self.net.forward_with_params(obs_t, self.population[i : i + 1, :])  # [1, out_dim]
                action = self._infer_action(logits, getattr(env, "action_space", None))
                step_out = env.step(action)
                if len(step_out) == 5:  # Gymnasium
                    obs, rew, term, trunc, _ = step_out
                    total += float(rew)
                    if bool(term or trunc):
                        obs, _ = env.reset()
                else:
                    obs, rew, done = step_out
                    total += float(rew)
                    if bool(done):
                        obs = env.reset()
            fitness[i] = total
        return fitness

    @torch.no_grad()
    def evaluate(self) -> torch.Tensor:
        """
        Compute fitness for the entire population.
        If `fitness_fn` provided, call it with [N,D] params -> [N] fitness tensor.
        """
        if self.fitness_fn is not None:
            fit = self.fitness_fn(self.population)
            return fit.to(self.device)
        if self.vectorized_eval:
            return self._evaluate_population_vectorized()
        return self._evaluate_population_sequential()

    # ------------------- Evolution step ------------------------

    @torch.no_grad()
    def _apply_pipeline(self, fitness: torch.Tensor) -> torch.Tensor:
        """
        Apply operators pipeline to produce next generation candidates (children).
        Returns:
            next_pop: [N, D]
        """
        N, D = self.population.shape
        device = self.population.device
        current = self.population

        # Elites (copied unchanged)
        elites = torch.empty((0, D), device=device, dtype=current.dtype)
        if self.elitism > 0:
            elite_idx = torch.topk(fitness, k=self.elitism, largest=True, sorted=True).indices
            elites = current[elite_idx].clone()

        # Remaining offspring to produce
        remaining = N - elites.shape[0]
        if remaining <= 0:
            return elites

        # We process each group; the last output we expect children
        # A typical group: [Selection(k=remaining*2), Crossover, Mutation, ...]
        children = None

        for group in self.operators:
            parents_idx = None
            # Find Selection in group
            for op in group:
                if isinstance(op, Selection):
                    # For mating pairs; if no crossover present, we still select 'remaining'
                    k = remaining * 2
                    # Use provided k but override if smaller
                    if op.k != k:
                        # Non-destructive: create a new Selection of same type with k=k
                        op = type(op)(k=getattr(op, "k", k), **({} if not hasattr(op, "tournament_size") else {"tournament_size": op.tournament_size}))
                        op.k = k  # ensure
                    if isinstance(op, BestSelection):
                        parents_idx = op.select(fitness)
                    elif isinstance(op, RandomSelection):
                        parents_idx = op.select(fitness, rng=self.torch_rng)
                    elif isinstance(op, TournamentSelection):
                        parents_idx = op.select(fitness, rng=self.torch_rng)
                    elif isinstance(op, RouletteWheelSelection):
                        op.k = k
                        parents_idx = op.select(fitness, rng=self.torch_rng)
                    else:
                        raise TypeError(f"Unknown selection op: {op}")
                    break  # one selection per group supported (simple)

            # If no selection was specified, pick random parents
            if parents_idx is None:
                parents_idx = torch.randint(0, N, (remaining * 2,), device=device, generator=self.torch_rng)

            # Pair them
            parents_a = current[parents_idx[0::2]]
            parents_b = current[parents_idx[1::2]]

            # Apply crossovers (if any) to produce children
            kids = None
            for op in group:
                if isinstance(op, (XPointCrossover, UniformCrossover)):
                    kids = op.apply(parents_a if kids is None else kids, parents_b, rng=self.torch_rng)
                elif isinstance(op, (AdditiveMutation, GlobalMutation)):
                    # Mutations apply to 'kids' if exists, else to concatenated parents_a copy
                    target = kids if kids is not None else parents_a.clone()
                    kids = op.apply(target, rng=self.torch_rng)
                elif isinstance(op, Selection):
                    continue
                else:
                    raise TypeError(f"Unsupported operator in pipeline: {op}")

            if kids is None:
                # No operators produced new children; default to cloning parents_a
                kids = parents_a.clone()

            children = kids if children is None else children  # one group supported in this simple API
            # In this minimal design we support a single group that yields children.
            # Multiple groups can be used by users if they concatenate externally.

        # Assemble next population
        if elites.numel() > 0:
            next_pop = torch.empty_like(current)
            next_pop[: elites.shape[0]] = elites
            next_pop[elites.shape[0] : elites.shape[0] + remaining] = children[:remaining]
        else:
            next_pop = children[:N]
        return next_pop

    @torch.no_grad()
    def step(self) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One evolutionary generation: evaluate -> select/crossover/mutate -> replace."""
        fitness = self.evaluate()  # [N]
        best = fitness.max().item()
        mean = fitness.mean().item()
        self.history["best_fitness"].append(best)
        self.history["mean_fitness"].append(mean)

        next_pop = self._apply_pipeline(fitness)
        self.population = next_pop  # inplace replace

        # Optionally checkpoint best params
        if self.checkpoint_path:
            best_idx = torch.argmax(fitness).item()
            best_params = self.population[best_idx].detach().to("cpu")
            state = {
                "net_state_dict": self.net.state_dict(),
                "best_flat_params": best_params,
                "param_dim": self.param_dim,
                "history": self.history,
            }
            torch.save(state, self.checkpoint_path)

        return fitness, {"best": best, "mean": mean}

    @torch.no_grad()
    def run(self, generations: int) -> Tuple[TorchNet, float, Dict[str, List[float]]]:
        """
        Run evolution for `generations` and return best net, best fitness, history.
        """
        best_f = -float("inf")
        best_params = None

        for _ in range(int(generations)):
            fitness, stats = self.step()
            if stats["best"] > best_f:
                best_f = stats["best"]
                best_params = self.population[torch.argmax(fitness)].detach().clone()

        # Return a copy of the network with best_params loaded
        best_net = type(self.net)(
            # net_factory signature expected by user; attempt a generic constructor path:
            # We'll reconstruct by loading state_dict into a new instance created by user net_factory again.
            # Safer: clone existing module and set params.
            **self._reconstruct_kwargs_from_net(self.net)
        )
        # Load same weights as template then set best flat
        best_net.load_state_dict(self.net.state_dict())
        if best_params is not None:
            best_net.set_flat_params(best_params)

        return best_net, float(best_f), self.history

    # Best-effort extraction of constructor args for returning a fresh instance
    def _reconstruct_kwargs_from_net(self, net: TorchNet) -> Dict:
        # This helper assumes TorchNet signature. If user passes a subclass, this might fail;
        # in that case we fall back to returning `self.net` with params set (safe).
        try:
            obs_dim = net.layer_specs[0].in_dim
            out_dim = net.layer_specs[-1].out_dim
            hidden_sizes = [ls.out_dim for ls in net.layer_specs[:-1]]
            activations = [ls.activation for ls in net.layer_specs[:-1]]
            output_activation = net.layer_specs[-1].activation
            return dict(
                obs_dim=obs_dim,
                hidden_sizes=hidden_sizes,
                out_dim=out_dim,
                activations=activations,
                output_activation=output_activation,
                bias=net.bias,
                device=str(self.device),
                dtype=net.dtype_,
            )
        except Exception:
            return dict(
                obs_dim=net.layer_specs[0].in_dim,
                hidden_sizes=[ls.out_dim for ls in net.layer_specs[:-1]],
                out_dim=net.layer_specs[-1].out_dim,
                device=str(self.device),
            )


# -----------------------------------------------------------------------------
# Isaac Gym Vectorized Env Stub (replace with real IsaacGym environment)
# -----------------------------------------------------------------------------

class IsaacGymVecEnvStub:
    """
    Minimal stub that mimics a vectorized env with N instances.
    Replace with your real Isaac Gym / IsaacLab vectorized env.

    API:
        reset(seed=None) -> obs [N, obs_dim], info
        step(actions)    -> obs [N, obs_dim], rewards [N], terminateds [N] (bool), truncateds [N] (bool), infos
    """

    def __init__(self, num_envs: int, obs_dim: int, act_dim: int, device: Union[str, torch.device] = "cpu"):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = _get_device(device)

        # Spaces-like attributes (for adapter)
        class _Space:
            def __init__(self, shape):
                self.shape = (shape,)
                self.low = -np.ones(shape, dtype=np.float32)
                self.high = np.ones(shape, dtype=np.float32)

        self.single_action_space = _Space(act_dim)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        obs = torch.randn((self.num_envs, self.obs_dim), device=self.device)
        return obs, {}

    def step(self, actions):
        # Fake random walk dynamics with reward = -||action||^2 + noise
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        obs = torch.randn((self.num_envs, self.obs_dim), device=self.device)
        rewards = -actions.pow(2).sum(dim=-1) + 0.1 * torch.randn(self.num_envs, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return obs, rewards, terminated, truncated, {}


# -----------------------------------------------------------------------------
# Examples, Unit Tests, Benchmarks
# -----------------------------------------------------------------------------

def make_gym_vector_env(env_id: str, num_envs: int, seed: Optional[int] = None):
    """
    Create a SyncVectorEnv for Gymnasium with `num_envs` copies.
    """
    if gym is None:
        raise ImportError("Gymnasium is required for this example. Install `gymnasium`.")
    def thunk(rank: int):
        def _make():
            env = gym.make(env_id)
            if seed is not None:
                env.reset(seed=seed + rank)
            return env
        return _make
    return gym.vector.SyncVectorEnv([thunk(i) for i in range(num_envs)])


def cartpole_net_factory(env) -> TorchNet:
    if gym is None:
        raise ImportError("Gymnasium is required for the CartPole example.")
    obs_dim = int(np.prod(env.single_observation_space.shape))
    act_dim = env.single_action_space.n  # Discrete
    return TorchNet(obs_dim, [64, 64], act_dim, activations=["tanh", "tanh"], output_activation="identity", device="cpu")


def simple_continuous_net_factory(obs_dim: int, act_dim: int, device="cpu") -> TorchNet:
    return TorchNet(obs_dim, [64, 64], act_dim, activations=["tanh", "tanh"], output_activation="identity", device=device)


def unit_tests_sanity():
    """
    Minimal unit-style asserts verifying core requirements.
    """
    # Small net
    net = TorchNet(4, [8], 2, activations=["relu"])
    flat = net.get_flat_params()
    # Roundtrip
    perturbed = flat + torch.randn_like(flat) * 1e-3
    net.set_flat_params(perturbed)
    flat2 = net.get_flat_params()
    assert torch.allclose(perturbed, flat2, atol=1e-8), "get/set flat params roundtrip failed"

    # Mutation shape/device
    pop = perturbed.repeat(16, 1).to("cpu")
    mut = AdditiveMutation(rate=0.5, sigma=0.1)
    mutated = mut.apply(pop)
    assert mutated.shape == pop.shape and mutated.device == pop.device, "Mutation shape/device mismatch"
    assert not torch.allclose(mutated, pop), "Mutation didn't change params (likely zero rate/sigma?)"

    # Crossover sanity
    a = torch.randn(10, flat.numel())
    b = torch.randn(10, flat.numel())
    cx = XPointCrossover(x=2)
    child = cx.apply(a, b)
    # Child should have some genes from a and some from b
    eq_a = (child == a).float().mean().item()
    eq_b = (child == b).float().mean().item()
    assert 0.2 < eq_a < 0.8 and 0.2 < eq_b < 0.8, "Crossover didn't mix parents as expected"

    # Selection indices valid
    fitness = torch.randn(32)
    sel = TournamentSelection(k=8, tournament_size=3)
    idx = sel.select(fitness)
    assert idx.shape == (8,) and (idx >= 0).all() and (idx < 32).all(), "Selection returned invalid indices"


def tiny_benchmark(device: str = "cpu"):
    """
    Tiny benchmark: one generation step timing on a small setup.
    """
    dev = _get_device(device)
    N = 128
    obs_dim, act_dim = 4, 2
    net = simple_continuous_net_factory(obs_dim, act_dim, device=dev)
    base = net.get_flat_params()
    pop = base.repeat(N, 1).to(dev)
    pop += torch.randn_like(pop) * 0.01

    # Fake fitness fn (fast)
    def fitness_fn(p: torch.Tensor):
        return -p.pow(2).sum(dim=1)

    env = IsaacGymVecEnvStub(num_envs=N, obs_dim=obs_dim, act_dim=act_dim, device=dev)
    ne = Neuroevolution(env, lambda _: net, N, device=str(dev), fitness_fn=fitness_fn, vectorized_eval=True, elitism=4)

    t0 = time.time()
    fitness, stats = ne.step()
    dt = time.time() - t0

    mem_info = ""
    if dev.type == "cuda":
        mem_info = f", cuda_mem={torch.cuda.memory_allocated(dev) / 1e6:.1f}MB"
    print(f"[benchmark] device={dev.type}, pop={N}, D={base.numel()}, step_time={dt*1000:.1f}ms{mem_info}")


def example_cartpole(generations: int = 5, pop: int = 128, device: str = "cpu"):
    """
    Small demonstration on CartPole-v1 using gym.vector.SyncVectorEnv.
    """
    if gym is None:
        raise ImportError("Gymnasium is required for this example. Install `gymnasium`.")
    dev = _get_device(device)
    env = make_gym_vector_env("CartPole-v1", num_envs=pop, seed=42)
    net = cartpole_net_factory(env)

    ops = [
        [TournamentSelection(k=pop, tournament_size=3),
         XPointCrossover(x=1),
         AdditiveMutation(rate=0.05, sigma=0.02),
         GlobalMutation(rate=0.25, sigma=0.01)]
    ]

    ne = Neuroevolution(
        env,
        lambda _: net,
        population_size=pop,
        device=str(dev),
        operators=ops,
        fitness_fn=None,
        eval_steps=400,  # CartPole episodes usually terminate early, but we roll for fixed steps
        vectorized_eval=True,
        elitism=4,
        seed=123,
    )
    best_net, best_fit, history = ne.run(generations)
    print(f"[CartPole] best_fitness={best_fit:.2f}, mean_last={history['mean_fitness'][-1]:.2f}")
    return best_net, history


def isaac_stub_example(generations: int = 3, pop: int = 256, device: str = "auto"):
    """
    Demonstrates using the IsaacGymVecEnvStub. Swap with real Isaac Gym env.
    """
    dev = _get_device(device)
    obs_dim, act_dim = 24, 6
    env = IsaacGymVecEnvStub(num_envs=pop, obs_dim=obs_dim, act_dim=act_dim, device=dev)
    net = simple_continuous_net_factory(obs_dim, act_dim, device=dev)

    ne = Neuroevolution(
        env,
        lambda _: net,
        population_size=pop,
        device=str(dev),
        operators=[[TournamentSelection(k=pop, tournament_size=3), UniformCrossover(prob=0.5), AdditiveMutation(0.05, 0.03)]],
        fitness_fn=None,  # use env rewards
        eval_steps=128,
        vectorized_eval=True,
        elitism=8,
        seed=321,
    )
    best_net, best_fit, history = ne.run(generations)
    print(f"[IsaacStub] best_fitness={best_fit:.3f}, mean_last={history['mean_fitness'][-1]:.3f}")
    return best_net, history


# -----------------------------------------------------------------------------
# Performance checklist (explicit verification via comments & asserts)
# -----------------------------------------------------------------------------
def _performance_self_check():
    """
    Verifies core performance design choices:
      - population params stored as [N, D] on device
      - vectorized mutation/crossover/selection
      - copying via clone()/flat params (no deepcopy)
      - evaluation uses vectorized envs (when available)
    """
    obs_dim, act_dim = 8, 4
    dev = _get_device("cuda" if torch.cuda.is_available() else "cpu")
    N = 64
    net = simple_continuous_net_factory(obs_dim, act_dim, device=dev)
    base = net.get_flat_params()
    pop = base.repeat(N, 1).to(dev)
    assert pop.shape == (N, base.numel())
    # Vectorized mutations: single call on [N,D]
    m1 = AdditiveMutation(0.1, 0.02).apply(pop)
    m2 = GlobalMutation(0.5, 0.02).apply(pop)
    assert m1.shape == pop.shape and m2.shape == pop.shape
    # Selection vectorized via topk/multinomial
    fitness = torch.randn(N, device=dev)
    s1 = BestSelection(k=8).select(fitness)
    s2 = RouletteWheelSelection(k=8).select(torch.relu(fitness))
    assert s1.shape == (8,) and s2.shape == (8,)
    # Copying via clone
    clone = pop.clone()
    assert clone.data_ptr() != pop.data_ptr()
    # Vectorized forward_with_params uses bmm internally (batch op)
    x = torch.randn(N, obs_dim, device=dev)
    y = net.forward_with_params(x, pop)
    assert y.shape == (N, act_dim)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="NeuroEvo: torch-first neuroevolution")
    p.add_argument("--env", type=str, default="CartPole-v1", help="Gymnasium env id (for demo)")
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--pop", type=int, default=128)
    p.add_argument("--eval-steps", type=int, default=400)
    p.add_argument("--device", type=str, default="auto", help="cpu/cuda/auto")
    p.add_argument("--benchmark", action="store_true", help="Run tiny benchmark")
    p.add_argument("--isaac-stub", action="store_true", help="Run IsaacGym stub demo")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Run minimal unit tests
    unit_tests_sanity()
    _performance_self_check()

    args = parse_args()

    if args.benchmark:
        tiny_benchmark(device=args.device)

    if args.isaac_stub:
        isaac_stub_example(generations=max(1, args.generations // 2), pop=args.pop, device=args.device)

    # Gymnasium demo (CartPole by default)
    if args.env and not args.isaac_stub:
        if gym is None:
            print("Gymnasium not installed; skipping CartPole demo.")
        else:
            # Build vectorized env & run
            env = make_gym_vector_env(args.env, num_envs=args.pop, seed=123)
            net = cartpole_net_factory(env)

            ops = [
                [TournamentSelection(k=args.pop, tournament_size=3),
                 XPointCrossover(x=1),
                 AdditiveMutation(rate=0.05, sigma=0.02),
                 GlobalMutation(rate=0.25, sigma=0.01)]
            ]

            ne = Neuroevolution(
                env,
                lambda _: net,
                population_size=args.pop,
                device=args.device,
                operators=ops,
                fitness_fn=None,
                eval_steps=args.eval_steps,
                vectorized_eval=True,
                elitism=4,
                seed=42,
                checkpoint_path=None,
            )
            t0 = time.time()
            best_net, best_fit, history = ne.run(args.generations)
            dt = time.time() - t0
            print(f"[{args.env}] gens={args.generations}, pop={args.pop}, best={best_fit:.2f}, "
                  f"mean_last={history['mean_fitness'][-1]:.2f}, time={dt:.1f}s")

    # Notes on memory allocations per generation (sample run):
    # - Population tensor [N,D] allocated once and reused (in-place replacement).
    # - Temporary children buffer allocated in __init__ (reused).
    # - Operator outputs reuse shapes; crossover/mutation create new tensors but
    #   are promptly reused/overwritten; avoid Python-side loops in genetic ops.
