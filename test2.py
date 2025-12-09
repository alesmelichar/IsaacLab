from __future__ import annotations
import asyncio
import time
import math
import copy
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# --- Import user primitives from the synchronous module (your existing file) ---
# If this file lives next to your current module, this import will work as-is.
# Otherwise, adjust the import path accordingly.
from neuroevo_torch import (
    BatchedMLPPolicy,
    Tensor,
    Selection,
    Mutation,
    Crossover,
    TournamentSelection,
    XPointCrossover,
    AdditiveMutation,
    GymAdapter,
    IsaacAdapter,
    EnvAdapter,
)

# =============================================================
# Utilities for reproducibility & policy cloning
# =============================================================

def _mix_uint64(x: int) -> int:
    """SplitMix64 mixer -> returns 64-bit scrambled value."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF


def make_eval_seed(base_seed: int, eval_id: int, rollout: int = 0) -> int:
    """Deterministic per-evaluation seed, independent of scheduling."""
    z = base_seed & 0xFFFFFFFFFFFFFFFF
    z = _mix_uint64(z ^ (eval_id & 0xFFFFFFFFFFFFFFFF))
    z = _mix_uint64(z ^ (rollout & 0xFFFFFFFFFFFFFFFF))
    # gym prefers 32-bit signed int
    return int(z & 0x7FFFFFFF)


def _policy_for_device(src: BatchedMLPPolicy, device: torch.device, dtype: torch.dtype) -> BatchedMLPPolicy:
    """Create a lightweight device-local view of the policy to avoid concurrent .to() races."""
    clone = copy.copy(src)
    clone.device = torch.device(device)
    clone.dtype = dtype
    if clone.action_space == "continuous":
        for attr in ("action_low", "action_high", "_a", "_m"):
            t = getattr(clone, attr, None)
            if t is not None:
                setattr(clone, attr, t.to(device, dtype))
    return clone


# =============================================================
# Environment reset with seeds (works for Gym/Gymnasium + tries Isaac)
# =============================================================

def _adapter_reset(adapter: EnvAdapter, seeds: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Reset helper that tries to pass per-env seeds when supported.
    - For GymAdapter (vector), we call underlying env.reset(seed=list_of_ints)
    - For IsaacAdapter, we try env.reset(seeds) if provided, otherwise env.reset()
    Fallbacks gracefully if signatures don't match.
    Returns obs on adapter.device with adapter's dtype.
    """
    device = adapter.device
    dtype = getattr(adapter, "dtype", torch.float32)

    # Gym path with vector env
    if isinstance(adapter, GymAdapter) and hasattr(adapter, "env"):
        if seeds is not None:
            if isinstance(seeds, torch.Tensor):
                seed_list = [int(x) for x in seeds.to("cpu").tolist()]
            else:
                seed_list = None
        else:
            seed_list = None
        try:
            obs, _ = adapter.env.reset(seed=seed_list)
        except TypeError:
            obs, _ = adapter.env.reset()
        return torch.as_tensor(obs, device=device, dtype=dtype)

    # Isaac / custom adapters
    try:
        if seeds is None:
            obs = adapter.reset()
        else:
            # try common names
            try:
                obs = adapter.reset(seeds)
            except TypeError:
                obs = adapter.reset(seed=seeds)
        return obs.to(device, dtype)
    except Exception:
        # last-resort: whatever reset does
        obs = adapter.reset()
        return obs.to(device, dtype)


# =============================================================
# Batch evaluation (single wave) with optional rollout averaging + seeds
# =============================================================

def evaluate_wave(
    policy: BatchedMLPPolicy,
    genomes: torch.Tensor,            # [B, P]
    adapter: EnvAdapter,
    max_steps: int,
    rollouts_per_genome: int = 1,
    base_seed: Optional[int] = None,
    eval_ids: Optional[torch.Tensor] = None,  # [B]
) -> torch.Tensor:
    """Evaluates a batch ("wave") and returns fitness [B]. Uses device-local policy copy."""
    device = adapter.device
    dtype = getattr(adapter, "dtype", policy.dtype)
    B = genomes.shape[0]
    pol = _policy_for_device(policy, device, dtype)

    with torch.inference_mode():
        if rollouts_per_genome <= 1:
            # Seeds per-env
            seeds = None
            if base_seed is not None and eval_ids is not None:
                seeds = torch.tensor([make_eval_seed(base_seed, int(eid), 0) for eid in eval_ids.tolist()],
                                     device=device, dtype=torch.int64)
            layers = pol.unflatten(genomes.to(device, dtype))
            obs = _adapter_reset(adapter, seeds)
            assert obs.shape[0] == B, f"Env reset returned {obs.shape[0]} obs, expected {B}"
            rewards = torch.zeros((B,), device=device, dtype=dtype)
            dones = torch.zeros((B,), device=device, dtype=torch.bool)
            for _ in range(max_steps):
                logits_or_actions = pol.forward_from_layers(obs, layers)
                actions = logits_or_actions.argmax(dim=-1) if adapter.discrete else logits_or_actions
                obs, r, term, trunc = adapter.step(actions)
                step_done = (term | trunc)
                rewards += r * (~dones)
                dones |= step_done
                if torch.all(dones):
                    break
            return rewards.to(policy.device, policy.dtype)
        else:
            # Expand K rollouts per genome and mean
            K = int(rollouts_per_genome)
            expanded = genomes.repeat_interleave(K, dim=0)
            eval_ids_rep = eval_ids.repeat_interleave(K) if eval_ids is not None else None
            seeds = None
            if base_seed is not None and eval_ids_rep is not None:
                seeds = torch.tensor([
                    make_eval_seed(base_seed, int(eid), int(i % K)) for i, eid in enumerate(eval_ids_rep.tolist())
                ], device=device, dtype=torch.int64)
            layers = pol.unflatten(expanded.to(device, dtype))
            obs = _adapter_reset(adapter, seeds)
            assert obs.shape[0] == expanded.shape[0]
            R = expanded.shape[0]
            rewards = torch.zeros((R,), device=device, dtype=dtype)
            dones = torch.zeros((R,), device=device, dtype=torch.bool)
            for _ in range(max_steps):
                logits_or_actions = pol.forward_from_layers(obs, layers)
                actions = logits_or_actions.argmax(dim=-1) if adapter.discrete else logits_or_actions
                obs, r, term, trunc = adapter.step(actions)
                step_done = (term | trunc)
                rewards += r * (~dones)
                dones |= step_done
                if torch.all(dones):
                    break
            rewards = rewards.view(B, K).mean(dim=1)
            return rewards.to(policy.device, policy.dtype)


# =============================================================
# Asynchronous backends (GPU Isaac, CPU Gym)
# =============================================================

@dataclass
class EvalTask:
    eval_id: int
    genome: torch.Tensor           # [P]
    rollout_k: int
    max_steps: int
    backend_hint: Optional[str] = None  # "gpu" | "cpu" | None

@dataclass
class EvalResult:
    eval_id: int
    fitness: float


class AsyncBackend:
    def __init__(self, name: str, capacity: int):
        self.name = name
        self.capacity = capacity
        self.in_q: "asyncio.Queue[EvalTask]" = asyncio.Queue()
        self.out_q: "asyncio.Queue[EvalResult]" = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self):  # pragma: no cover - abstract
        raise NotImplementedError


class GymBatchBackend(AsyncBackend):
    def __init__(
        self,
        env_fn: Callable[[], "gym.Env"],
        discrete: bool,
        policy: BatchedMLPPolicy,
        capacity: int,
        device_dtype: Optional[Tuple[torch.device, torch.dtype]] = None,
        base_seed: int = 0,
    ):
        super().__init__(name="cpu-gym", capacity=capacity)
        self.env_fn = env_fn
        self.discrete = discrete
        self.policy = policy
        self.base_seed = int(base_seed)
        # Build fixed-size adapter once
        env_fns = [env_fn for _ in range(capacity)]
        self.adapter = GymAdapter(env_fns, discrete=discrete, dtype=policy.dtype)
        if device_dtype is not None:
            dev, dt = device_dtype
            self.adapter.device = dev
            self.adapter.dtype = dt

    async def _run_loop(self):
        # gather up to capacity tasks and evaluate as a wave (pad to capacity)
        while True:
            batch: List[EvalTask] = []
            # Wait for at least one task
            task0: EvalTask = await self.in_q.get()
            batch.append(task0)
            # Pull more without waiting (if available) to fill capacity
            try:
                while len(batch) < self.capacity:
                    batch.append(self.in_q.get_nowait())
            except asyncio.QueueEmpty:
                pass

            # Form tensors
            B = len(batch)
            eval_ids = torch.tensor([t.eval_id for t in batch], device=self.policy.device, dtype=torch.long)
            genomes = torch.stack([t.genome for t in batch], dim=0)
            k = batch[0].rollout_k
            steps = batch[0].max_steps

            # Pad to adapter capacity if needed
            if B < self.capacity:
                deficit = self.capacity - B
                pad_idx = torch.arange(deficit, device=genomes.device) % B
                genomes_pad = torch.cat([genomes, genomes[pad_idx]], dim=0)
                eval_ids_pad = torch.cat([eval_ids, eval_ids[pad_idx]], dim=0)
                valid = B
            else:
                genomes_pad = genomes
                eval_ids_pad = eval_ids
                valid = B

            # Evaluate in a background thread to avoid blocking the loop (CPU-bound)
            rewards = await asyncio.to_thread(
                evaluate_wave,
                self.policy,
                genomes_pad,
                self.adapter,
                steps,
                k,
                self.base_seed,
                eval_ids_pad,
            )
            # Emit only the first `valid` results (corresponding to the real tasks)
            for i in range(valid):
                await self.out_q.put(EvalResult(eval_id=int(eval_ids[i].item()), fitness=float(rewards[i].item())))



class IsaacBatchBackend(AsyncBackend):
    def __init__(
        self,
        isaac_env,
        discrete: bool,
        policy: BatchedMLPPolicy,
        capacity: Optional[int] = None,  # default: isaac_env.num_envs
        base_seed: int = 0,
    ):
        cap = getattr(isaac_env, "num_envs", None)
        if capacity is None and cap is not None:
            capacity = int(cap)
        assert capacity is not None and capacity > 0, "Isaac backend requires a valid capacity (env.num_envs)"
        super().__init__(name="gpu-isaac", capacity=capacity)
        self.adapter = IsaacAdapter(isaac_env, discrete=discrete, dtype=policy.dtype)
        self.policy = policy
        self.base_seed = int(base_seed)

    async def _run_loop(self):
        while True:
            batch: List[EvalTask] = []
            # Wait for one, then opportunistically gather more for a whole GPU wave
            task0: EvalTask = await self.in_q.get()
            batch.append(task0)
            try:
                while len(batch) < self.capacity:
                    batch.append(self.in_q.get_nowait())
            except asyncio.QueueEmpty:
                pass

            B = len(batch)
            eval_ids = torch.tensor([t.eval_id for t in batch], device=self.policy.device, dtype=torch.long)
            genomes = torch.stack([t.genome for t in batch], dim=0)
            k = batch[0].rollout_k
            steps = batch[0].max_steps

            # Pad to full GPU capacity
            if B < self.capacity:
                deficit = self.capacity - B
                pad_idx = torch.arange(deficit, device=genomes.device) % B
                genomes_pad = torch.cat([genomes, genomes[pad_idx]], dim=0)
                eval_ids_pad = torch.cat([eval_ids, eval_ids[pad_idx]], dim=0)
                valid = B
            else:
                genomes_pad = genomes
                eval_ids_pad = eval_ids
                valid = B

            rewards = evaluate_wave(
                self.policy,
                genomes_pad,
                self.adapter,
                steps,
                k,
                self.base_seed,
                eval_ids_pad,
            )
            for i in range(valid):
                await self.out_q.put(EvalResult(eval_id=int(eval_ids[i].item()), fitness=float(rewards[i].item())))


# =============================================================
# Steady-state EA (asynchronous, elitist, reproducible)
# =============================================================

@dataclass
class EAConfig:
    population_size: int
    elite_count: int
    max_steps: int
    rollouts_per_genome: int = 1
    total_children: int = 10_000  # evolution budget after initial population
    selection: Optional[Selection] = None
    crossover: Optional[Crossover] = None
    mutation: Optional[Mutation] = None
    seed: int = 0
    max_inflight: Optional[int] = None  # default: sum backend capacities
    # Backend routing: probability to send a task to GPU if available
    gpu_preference: float = 0.75


class AsyncSteadyStateEA:
    def __init__(
        self,
        policy: BatchedMLPPolicy,
        cfg: EAConfig,
        backends: List[AsyncBackend],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.policy = policy
        self.cfg = cfg
        self.backends = backends

        # Core tensor state
        self.device = torch.device(device) if device is not None else policy.device
        self.dtype = dtype or policy.dtype
        self.policy = self.policy.to(self.device, self.dtype)

        # RNG: single stream used ONLY by the manager (not workers), with deterministic eval_id ordering
        self.base_seed = int(cfg.seed)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(self.base_seed)

        # Initial population
        self.pop_size = int(cfg.population_size)
        self.genomes: Tensor = self.policy.sample_population(self.pop_size, generator=self.rng)
        self.fitness: Tensor = torch.full((self.pop_size,), float("nan"), device=self.device, dtype=self.dtype)

        # Operators
        self.selection = cfg.selection or TournamentSelection(k=2, tournament_size=3)
        self.crossover = cfg.crossover or XPointCrossover(x_points=1)
        self.mutation = cfg.mutation or AdditiveMutation(rate=0.1, sigma=0.1)

        # Elites
        self.elite_count = int(cfg.elite_count)
        assert 0 <= self.elite_count < self.pop_size
        self.elite_mask = torch.zeros((self.pop_size,), dtype=torch.bool, device=self.device)

        # Scheduling
        self.eval_counter = 0  # monotonically increasing eval_id
        self.expected_eval_id = 0
        self.pending: Dict[int, Tuple[str, Optional[int]]] = {}
        # maps eval_id -> ("init", idx) or ("child", None)

        self.max_inflight = cfg.max_inflight or sum(b.capacity for b in backends)
        self.out_q: "asyncio.Queue[EvalResult]" = asyncio.Queue()

        # Best tracking
        self.best_genome: Optional[Tensor] = None
        self.best_fitness: Optional[float] = None

    # ---------- Reproduction & replacement helpers ----------

    def _make_child(self) -> Tensor:
        # Parent pool selection (indices over current population)
        # Use only evaluated, non-NaN fitness
        valid = ~torch.isnan(self.fitness)
        assert torch.any(valid), "No evaluated individuals available for selection."
        fit = self.fitness.clone()
        fit[~valid] = -1e30
        # Select k indices per operator
        k = getattr(self.selection, "k", 2)
        idx = self.selection.select(fit, rng=self.rng)
        parents = self.genomes[idx]
        # Crossover -> exactly k offspring, take first
        if isinstance(self.crossover, Crossover):
            off = self.crossover.crossover(parents, num_offspring=k, rng=self.rng)
        else:
            off = parents.clone()
        child = off[0:1]  # [1, P]
        # Mutation
        if isinstance(self.mutation, Mutation):
            child = self.mutation.mutate(child, rng=self.rng)
        return child.squeeze(0)

    def _update_elites(self):
        # Called once initial fitness is available or periodically
        sorted_idx = torch.argsort(self.fitness, descending=True)
        self.elite_mask[:] = False
        self.elite_mask[sorted_idx[: self.elite_count]] = True
        best_idx = int(sorted_idx[0].item())
        best_fit = float(self.fitness[best_idx].item())
        self.best_fitness = best_fit if (self.best_fitness is None or best_fit > self.best_fitness) else self.best_fitness
        if self.best_genome is None or best_fit >= self.best_fitness:
            self.best_genome = self.genomes[best_idx].clone()

    def _replace_with_child(self, child: Tensor, child_fitness: float):
        # Replace the worst non-elite individual if child is better than that worst (\u00b5+1 elitist)
        non_elite_idx = torch.where(~self.elite_mask)[0]
        worst_idx = non_elite_idx[torch.argmin(self.fitness[non_elite_idx])]
        if child_fitness > float(self.fitness[worst_idx]):
            self.genomes[worst_idx] = child.to(self.device, self.dtype)
            self.fitness[worst_idx] = float(child_fitness)
            # Elites may change
            self._update_elites()

    # ---------- Scheduling helpers ----------

    def _route_backend(self, hint: Optional[str]) -> AsyncBackend:
        if hint == "gpu":
            for b in self.backends:
                if isinstance(b, IsaacBatchBackend):
                    return b
        if hint == "cpu":
            for b in self.backends:
                if isinstance(b, GymBatchBackend):
                    return b
        # heuristic: prefer GPU with probability p if present & queue not crazy
        gpu_b = next((b for b in self.backends if isinstance(b, IsaacBatchBackend)), None)
        cpu_b = next((b for b in self.backends if isinstance(b, GymBatchBackend)), None)
        if gpu_b and (np.random.rand() < self.cfg.gpu_preference):
            return gpu_b
        return gpu_b or cpu_b or self.backends[0]

    async def _submit_eval(self, genome: Tensor, rollout_k: int, max_steps: int, hint: Optional[str], kind: str, idx: Optional[int] = None):
        eid = self.eval_counter
        self.eval_counter += 1
        task = EvalTask(eval_id=eid, genome=genome.detach().to(self.policy.device, self.policy.dtype), rollout_k=rollout_k, max_steps=max_steps, backend_hint=hint)
        backend = self._route_backend(hint)
        await backend.in_q.put(task)
        self.pending[eid] = (kind, idx)

    async def _drain_backend_results(self):
        # Collect from all backend out queues and forward to common out_q
        polls = [b.out_q.get() for b in self.backends]
        done, pending = await asyncio.wait(polls, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            res: EvalResult = d.result()
            await self.out_q.put(res)
        # put back unfinished awaitables by re-registering them
        for p in pending:
            p.cancel()

    # ---------- Public API ----------

    async def run(self, progress: Optional[Callable[[Dict], None]] = None) -> Dict:
        """Run the asynchronous steady-state EA to completion.
        Returns a summary dict with best genome/fitness and history.
        """
        # Start backends
        for b in self.backends:
            await b.start()

        # Submit initial population in waves (auto routing)
        for i in range(self.pop_size):
            await self._submit_eval(self.genomes[i], self.cfg.rollouts_per_genome, self.cfg.max_steps, hint=None, kind="init", idx=i)

        inflight = self.pop_size
        produced_children = 0

        # Process results in deterministic eval_id order
        buffer: Dict[int, EvalResult] = {}

        def maybe_process_buffer():
            nonlocal produced_children, inflight
            while self.expected_eval_id in buffer:
                res = buffer.pop(self.expected_eval_id)
                kind, idx = self.pending.pop(res.eval_id)
                if kind == "init":
                    assert idx is not None
                    self.fitness[idx] = float(res.fitness)
                    # when all initial finished, compute elites
                    if not torch.isnan(self.fitness).any():
                        self._update_elites()
                else:
                    # child result: perform replacement
                    # We cached the child genome as pending? Simpler: store child in temp map
                    pass
                self.expected_eval_id += 1

        # For children, we must also remember their genome tensors by eval_id to replace on return
        child_cache: Dict[int, Tensor] = {}

        # Evolution loop
        total_budget = int(self.cfg.total_children)
        try:
            while produced_children < total_budget or inflight > 0:
                # Keep the pipeline full
                while inflight < self.max_inflight and (produced_children < total_budget or torch.isnan(self.fitness).any()):
                    if torch.isnan(self.fitness).any():
                        # Still initializing -> let initial evals drain before generating children
                        break
                    # Create one child and submit
                    child = self._make_child()
                    eid_before = self.eval_counter
                    await self._submit_eval(child, self.cfg.rollouts_per_genome, self.cfg.max_steps, hint=None, kind="child", idx=None)
                    child_cache[eid_before] = child.clone()
                    inflight += 1
                    produced_children += 1

                # Receive at least one result
                await self._drain_backend_results()
                res = await self.out_q.get()
                buffer[res.eval_id] = res

                # If this result is a child, try replacement when it's its turn
                kind, _ = self.pending.get(res.eval_id, ("?", None))
                if kind == "child":
                    # Wait until it's the expected id to keep determinism
                    if res.eval_id == self.expected_eval_id:
                        # process immediately
                        child = child_cache.pop(res.eval_id)
                        self._replace_with_child(child, res.fitness)
                        self.pending.pop(res.eval_id, None)
                        self.expected_eval_id += 1
                    else:
                        # store; will be processed in maybe_process_buffer when we reach it
                        pass
                elif kind == "init":
                    # If this init result is the expected one, write it now
                    if res.eval_id == self.expected_eval_id:
                        idx = self.pending[res.eval_id][1]
                        assert idx is not None
                        self.fitness[idx] = float(res.fitness)
                        self.pending.pop(res.eval_id, None)
                        self.expected_eval_id += 1
                    # else it stays buffered

                # Now process any contiguous buffered results in order
                while self.expected_eval_id in buffer:
                    res2 = buffer.pop(self.expected_eval_id)
                    kind2, idx2 = self.pending.pop(res2.eval_id)
                    if kind2 == "init":
                        assert idx2 is not None
                        self.fitness[idx2] = float(res2.fitness)
                    else:
                        child = child_cache.pop(res2.eval_id)
                        self._replace_with_child(child, res2.fitness)
                    self.expected_eval_id += 1

                # Recompute elites once all initial are ready
                if not torch.isnan(self.fitness).any():
                    self._update_elites()

                inflight = len(self.pending)

                if progress is not None:
                    best_fit = (self.best_fitness if self.best_fitness is not None else float("nan"))
                    progress({
                        "inflight": inflight,
                        "children_produced": produced_children,
                        "best_fitness": best_fit,
                        "mean_fitness": float(torch.nanmean(self.fitness).item()),
                    })
        finally:
            for b in self.backends:
                await b.stop()

        return {
            "best_fitness": self.best_fitness,
            "best_genome": None if self.best_genome is None else self.best_genome.clone(),
            "population_genomes": self.genomes.clone(),
            "population_fitness": self.fitness.clone(),
        }


import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np

ENV_ID = "BipedalWalker-v3"  # or "LunarLander-v2" / "LunarLanderContinuous-v2"

# Probe a single env to get spaces & dims
probe = gym.make(ENV_ID)
obs_dim = int(np.prod(probe.observation_space.shape))
act_space = probe.action_space
is_discrete = isinstance(act_space, Discrete)

if is_discrete:
    act_dim = int(act_space.n)
    policy = BatchedMLPPolicy(
        obs_dim=obs_dim, hidden_layers=[64, 64], act_dim=act_dim,
        activation="tanh", action_space="discrete"
    )
else:
    act_dim = int(np.prod(act_space.shape))
    policy = BatchedMLPPolicy(
        obs_dim=obs_dim, hidden_layers=[128, 128], act_dim=act_dim,
        activation="tanh", action_space="continuous",
        action_low=act_space.low, action_high=act_space.high
    )
probe.close()

def env_fn():  # gym/gymnasium factory
    return gym.make(ENV_ID)

cpu_backend = GymBatchBackend(
    env_fn=env_fn,
    discrete=is_discrete,    # IMPORTANT
    policy=policy,
    capacity=32,
    base_seed=1234,
)

cfg = EAConfig(
    population_size=50, elite_count=5,
    max_steps=1000, total_children=5000,
    selection=TournamentSelection(k=4),
    crossover=XPointCrossover(1),
    mutation=AdditiveMutation(0.05, 0.1),
    seed=42,
)
ea = AsyncSteadyStateEA(policy, cfg, backends=[cpu_backend])

def on_progress(info):
    print(f"inflight={info['inflight']:<4} kids={info['children_produced']:<6} "
          f"best={info['best_fitness']} mean={info['mean_fitness']:.3f}")

asyncio.run(ea.run(progress=on_progress))