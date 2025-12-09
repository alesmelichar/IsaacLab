
"""
neuroevo_demo.py
A compact, efficient PyTorch-first neuroevolution library with:
 - Neural network class with easy access to parameter vectors
 - Operator classes for selection / mutation / crossover
 - Neuroevolution orchestrator (vectorized where possible)
 - Small demo (uses gymnasium if available; otherwise synthetic demo)

Designed to be GPU-friendly: parameters are represented as a single tensor of shape (pop, n_params)
and mutations/crossovers operate directly on that tensor for speed.
"""

from typing import Callable, List, Optional, Tuple, Sequence
import math
import random
import copy
import sys

# Try to import torch; fall back to numpy implementation if unavailable.
try:
    import torch
    import torch.nn as nn
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
    TORCH_AVAILABLE = True
except Exception as e:
    TORCH_AVAILABLE = False
    import numpy as np

# ----------------------------- Neural network wrapper -----------------------------
if TORCH_AVAILABLE:
    class TorchMLP(nn.Module):
        """
        Simple MLP with easy access to flattened parameter vector.
        layers: list of int sizes (including input and output)
        activations: list/str specifying activation for hidden layers. Output uses Identity by default.
        """
        ACTIVATIONS = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'identity': nn.Identity,
            'leaky_relu': nn.LeakyReLU,
        }

        def __init__(self, layers: Sequence[int], activation: str = 'tanh', device: Optional[torch.device] = None):
            super().__init__()
            assert len(layers) >= 2, "Provide at least input and output size."
            self.device = device or torch.device('cpu')
            self.layers_sizes = list(layers)
            modules = []
            for i in range(len(layers)-1):
                modules.append(nn.Linear(layers[i], layers[i+1]))
                # Use activation for all but final layer
                act = activation if i < len(layers)-2 else 'identity'
                modules.append(self.ACTIVATIONS[act]())
            self.net = nn.Sequential(*modules)
            self.to(self.device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x.to(self.device))

        def get_param_vector(self) -> torch.Tensor:
            """Return a 1-D tensor with all parameters concatenated (on correct device)."""
            vec = parameters_to_vector(self.parameters()).detach().to(self.device).clone()
            return vec

        def set_param_vector(self, vec: torch.Tensor) -> None:
            """Set module parameters from flattened vector (in-place)."""
            vector_to_parameters(vec.to(self.device), self.parameters())

        @property
        def n_params(self) -> int:
            return sum(p.numel() for p in self.parameters())
else:
    # Minimal numpy MLP for fallback (no autograd; only for functionality tests)
    class NumpyMLP:
        def __init__(self, layers: Sequence[int], activation: str = 'tanh'):
            import numpy as _np
            self._np = _np
            self.layers = list(layers)
            self.weights = [ _np.random.randn(self.layers[i+1], self.layers[i]).astype(_np.float32) for i in range(len(self.layers)-1) ]
            self.biases  = [ _np.random.randn(self.layers[i+1]).astype(_np.float32) for i in range(len(self.layers)-1) ]
            self.act = activation

        def forward(self, x):
            a = x
            for i,(W,b) in enumerate(zip(self.weights, self.biases)):
                a = a.dot(W.T) + b
                if i < len(self.weights)-1:
                    if self.act == 'tanh':
                        a = self._np.tanh(a)
                    elif self.act == 'relu':
                        a = self._np.maximum(0,a)
            return a

        def get_param_vector(self):
            flat = [w.ravel() for w in self.weights] + [b.ravel() for b in self.biases]
            return self._np.concatenate(flat).astype(self._np.float32)

        def set_param_vector(self, vec):
            idx = 0
            for i in range(len(self.weights)):
                w = self.weights[i]
                s = w.size
                self.weights[i] = vec[idx:idx+s].reshape(w.shape)
                idx += s
            for i in range(len(self.biases)):
                b = self.biases[i]
                s = b.size
                self.biases[i] = vec[idx:idx+s]
                idx += s

        @property
        def n_params(self):
            return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)


# ----------------------------- Operators (mutation / crossover / selection) -----------------------------
if TORCH_AVAILABLE:
    # Vectorized mutations that operate on (pop, n_params) tensors
    class AdditiveMutation:
        def __init__(self, mutation_rate: float = 0.1, sigma: float = 0.02, device: Optional[torch.device] = None):
            self.mutation_rate = mutation_rate
            self.sigma = sigma
            self.device = device or torch.device('cpu')

        def __call__(self, param_matrix: torch.Tensor) -> torch.Tensor:
            # param_matrix: (pop, n_params)
            with torch.no_grad():
                mask = (torch.rand_like(param_matrix) < self.mutation_rate).to(param_matrix.device)
                noise = torch.randn_like(param_matrix) * self.sigma
                out = param_matrix + mask * noise
            return out

    class GlobalMutation:
        def __init__(self, mutation_rate: float = 0.05, magnitude: float = 0.1):
            """
            GlobalMutation occasionally replaces entire parameter entries with random values
            drawn from a normal distribution scaled by magnitude.
            """
            self.mutation_rate = mutation_rate
            self.magnitude = magnitude

        def __call__(self, param_matrix: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                mask = (torch.rand_like(param_matrix) < self.mutation_rate).to(param_matrix.device)
                # replace with random normals scaled by magnitude of param std
                rnd = torch.randn_like(param_matrix) * self.magnitude * (param_matrix.std(dim=0, keepdim=True) + 1e-6)
                out = torch.where(mask, rnd, param_matrix)
            return out

    class XPointCrossover:
        def __init__(self, x_points: int = 1):
            assert x_points >= 1
            self.x_points = x_points

        def __call__(self, parents: torch.Tensor) -> torch.Tensor:
            """
            parents: (pop, n_params) where pop is even and consecutive pairs are parents
            returns: children tensor of same shape
            Implementation: for each pair (2i, 2i+1), perform x-point crossover on vector elements.
            """
            pop, n = parents.shape
            assert pop % 2 == 0, "Population for pairwise crossover should be even."
            children = parents.clone()
            for i in range(0, pop, 2):
                p1 = parents[i]
                p2 = parents[i+1]
                # choose x cut points
                points = sorted(random.sample(range(1, n), min(self.x_points, n-1)))
                mask = torch.zeros(n, dtype=torch.bool, device=parents.device)
                take_from_p1 = True
                last = 0
                for pt in points + [n]:
                    if take_from_p1:
                        mask[last:pt] = True
                    else:
                        mask[last:pt] = False
                    take_from_p1 = not take_from_p1
                    last = pt
                child1 = torch.where(mask, p1, p2)
                child2 = torch.where(mask, p2, p1)
                children[i]   = child1
                children[i+1] = child2
            return children

    # Selection functions produce indices of parents to keep / mate
    class BestSelection:
        def __init__(self, k: int):
            self.k = k

        def __call__(self, fitness: torch.Tensor) -> torch.Tensor:
            # fitness: (pop,) higher = better
            _, idx = torch.topk(fitness, self.k)
            return idx

    class RandomSelection:
        def __init__(self, k: int):
            self.k = k

        def __call__(self, fitness: torch.Tensor) -> torch.Tensor:
            pop = fitness.shape[0]
            idx = torch.randint(0, pop, (self.k,), device=fitness.device)
            return idx

    class TournamentSelection:
        def __init__(self, k: int, tournament_size: int = 3):
            self.k = k
            self.tournament_size = tournament_size

        def __call__(self, fitness: torch.Tensor) -> torch.Tensor:
            pop = fitness.shape[0]
            winners = []
            for _ in range(self.k):
                competitors = torch.randint(0, pop, (self.tournament_size,), device=fitness.device)
                best = competitors[torch.argmax(fitness[competitors])]
                winners.append(best.item())
            return torch.tensor(winners, device=fitness.device, dtype=torch.long)

    class RouletteWheelSelection:
        def __init__(self, k: int):
            self.k = k

        def __call__(self, fitness: torch.Tensor) -> torch.Tensor:
            # Shift fitness to be positive
            minf = float(torch.min(fitness).item())
            shifted = fitness - minf + 1e-8
            probs = shifted / torch.sum(shifted)
            idx = torch.multinomial(probs, self.k, replacement=True)
            return idx
else:
    # Numpy versions for fallback
    class AdditiveMutation:
        def __init__(self, mutation_rate=0.1, sigma=0.02):
            self.mutation_rate = mutation_rate
            self.sigma = sigma

        def __call__(self, param_matrix):
            mask = (np.random.rand(*param_matrix.shape) < self.mutation_rate)
            noise = np.random.randn(*param_matrix.shape) * self.sigma
            return param_matrix + mask * noise

    class GlobalMutation:
        def __init__(self, mutation_rate=0.05, magnitude=0.1):
            self.mutation_rate = mutation_rate
            self.magnitude = magnitude

        def __call__(self, param_matrix):
            mask = (np.random.rand(*param_matrix.shape) < self.mutation_rate)
            rnd = np.random.randn(*param_matrix.shape) * self.magnitude * (param_matrix.std(axis=0, keepdims=True) + 1e-6)
            return np.where(mask, rnd, param_matrix)

    class XPointCrossover:
        def __init__(self, x_points=1):
            self.x_points = x_points

        def __call__(self, parents):
            pop, n = parents.shape
            children = parents.copy()
            for i in range(0, pop, 2):
                p1 = parents[i].copy()
                p2 = parents[i+1].copy()
                points = sorted(random.sample(range(1, n), min(self.x_points, n-1)))
                mask = np.zeros(n, dtype=bool)
                take_from_p1 = True
                last = 0
                for pt in points + [n]:
                    if take_from_p1:
                        mask[last:pt] = True
                    take_from_p1 = not take_from_p1
                    last = pt
                child1 = np.where(mask, p1, p2)
                child2 = np.where(mask, p2, p1)
                children[i] = child1
                children[i+1] = child2
            return children

    class BestSelection:
        def __init__(self, k): self.k = k
        def __call__(self, fitness):
            return np.argsort(fitness)[-self.k:]

    class RandomSelection:
        def __init__(self, k): self.k = k
        def __call__(self, fitness):
            pop = fitness.shape[0]
            return np.random.randint(0, pop, size=(self.k,))

    class TournamentSelection:
        def __init__(self, k, tournament_size=3): self.k=k; self.tournament_size=tournament_size
        def __call__(self, fitness):
            pop = fitness.shape[0]
            winners = []
            for _ in range(self.k):
                competitors = np.random.randint(0,pop,size=(self.tournament_size,))
                best = competitors[np.argmax(fitness[competitors])]
                winners.append(best)
            return np.array(winners, dtype=int)

    class RouletteWheelSelection:
        def __init__(self,k): self.k=k
        def __call__(self, fitness):
            minf = fitness.min()
            shifted = fitness - minf + 1e-8
            probs = shifted / shifted.sum()
            return np.random.choice(len(fitness), size=(self.k,), p=probs, replace=True)


# ----------------------------- Neuroevolution orchestrator -----------------------------
if TORCH_AVAILABLE:
    class Neuroevolution:
        def __init__(self,
                     individual_constructor: Callable[[], object],
                     population_size: int = 64,
                     device: Optional[torch.device] = None,
                     elitism: int = 1,
                     seed: Optional[int] = None):
            """
            individual_constructor: function that returns a fresh neural network instance (TorchMLP)
            population_size: number of agents
            device: torch.device()
            elitism: number of top individuals preserved every generation
            """
            if seed is not None:
                random.seed(seed)
                torch.manual_seed(seed)
            self.device = device or torch.device('cpu')
            self.pop_size = population_size
            self.individual_template = individual_constructor()
            self.n_params = self.individual_template.n_params
            self.elitism = elitism
            # Initialize population parameter matrix (pop, n_params)
            self.population = torch.stack([ self.individual_template.get_param_vector() for _ in range(self.pop_size) ]).to(self.device)
            # Keep a template network to clone weights back when needed
            self.constructor = individual_constructor

        def get_individual(self, idx: int):
            net = self.constructor()
            vec = self.population[idx].detach().cpu().clone()
            net.set_param_vector(vec.to(net.get_param_vector().device if TORCH_AVAILABLE else 'cpu'))
            return net

        def evaluate_population(self, eval_fn: Callable[[object], float], batch: Optional[Sequence[int]] = None) -> torch.Tensor:
            """
            eval_fn: function taking an individual network instance (or param vector) and returning scalar fitness.
            For speed, eval_fn should accept param vectors if you want vectorized evaluation.
            Returns tensor of shape (pop,)
            """
            fitness = torch.zeros(self.pop_size, device=self.device)
            for i in range(self.pop_size):
                ind = self.get_individual(i)
                fitness[i] = float(eval_fn(ind))
            return fitness

        def step(self,
                 eval_fn: Callable[[object], float],
                 selection_fn: Callable = None,
                 mutation_ops: List[Callable] = None,
                 crossover_op: Optional[Callable] = None):
            """
            One generation step:
            - evaluate population
            - select parents (indices) via selection_fn (expects fitness Tensor)
            - form parent matrix, pair them, apply crossover to produce children
            - apply mutations
            - form next generation with elitism
            """
            if selection_fn is None:
                selection_fn = BestSelection(self.pop_size//2)
            if mutation_ops is None:
                mutation_ops = [AdditiveMutation(0.1, 0.02)]
            # 1) evaluate
            fitness = self.evaluate_population(eval_fn)
            # 2) elitism
            _, order = torch.sort(fitness, descending=True)
            elites = self.population[order[:self.elitism]].clone() if self.elitism > 0 else torch.empty(0, self.n_params, device=self.device)
            # 3) selection to pick parents (we need an even number to pair)
            parents_idx = selection_fn(fitness)
            # Ensure even parent count for pairing
            if len(parents_idx) % 2 == 1:
                parents_idx = torch.cat([parents_idx, parents_idx[:1]])
            parents = self.population[parents_idx]
            # If not enough parents to fill population, repeat parents to reach pop_size
            # Pair parents sequentially: (0,1), (2,3), ...
            if parents.shape[0] < self.pop_size:
                # tile parents to fill
                times = math.ceil(self.pop_size / parents.shape[0])
                parents = parents.repeat(times, 1)[:self.pop_size]
            # 4) crossover
            children = parents.clone()
            if crossover_op is not None:
                # Ensure even count for pairwise crossover
                if children.shape[0] % 2 == 1:
                    children = torch.cat([children, children[:1]], dim=0)
                children = crossover_op(children)
            # 5) mutations (apply sequentially for now)
            for mut in mutation_ops:
                children = mut(children)
            # 6) assemble new population: elites + children (trimmed)
            new_pop = children[:max(0, self.pop_size - self.elitism)]
            if self.elitism > 0:
                self.population = torch.cat([elites, new_pop], dim=0)[:self.pop_size]
            else:
                self.population = new_pop[:self.pop_size]
            return fitness, order[0].item(), fitness.max().item()

else:
    class Neuroevolution:
        def __init__(self, individual_constructor: Callable[[], object], population_size: int = 64, seed: Optional[int] = None):
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            self.pop_size = population_size
            self.individual_template = individual_constructor()
            self.n_params = self.individual_template.n_params
            self.population = np.stack([ self.individual_template.get_param_vector() for _ in range(self.pop_size) ])

        def get_individual(self, idx):
            net = copy.deepcopy(self.individual_template)
            net.set_param_vector(self.population[idx].copy())
            return net

        def evaluate_population(self, eval_fn):
            fitness = np.zeros(self.pop_size, dtype=np.float32)
            for i in range(self.pop_size):
                ind = self.get_individual(i)
                fitness[i] = float(eval_fn(ind))
            return fitness

        def step(self, eval_fn, selection_fn=None, mutation_ops=None, crossover_op=None):
            if selection_fn is None:
                selection_fn = BestSelection(self.pop_size//2)
            if mutation_ops is None:
                mutation_ops = [AdditiveMutation(0.1, 0.02)]
            fitness = self.evaluate_population(eval_fn)
            order = np.argsort(fitness)[::-1]
            parents_idx = selection_fn(fitness)
            parents = self.population[parents_idx]
            if parents.shape[0] < self.pop_size:
                times = math.ceil(self.pop_size / parents.shape[0])
                parents = np.tile(parents, (times,1))[:self.pop_size]
            children = parents.copy()
            if crossover_op is not None:
                if children.shape[0] % 2 == 1:
                    children = np.vstack([children, children[:1]])
                children = crossover_op(children)
            for mut in mutation_ops:
                children = mut(children)
            self.population = children[:self.pop_size]
            return fitness, order[0], fitness.max()

# ----------------------------- Demo / Usage example -----------------------------
def _demo_synthetic():
    print("Running synthetic demo (no gymnasium available).")
    # create an individual constructor for a small MLP: input 4, hidden 16, output 2
    if TORCH_AVAILABLE:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        constructor = lambda: TorchMLP([4, 16, 2], activation='tanh', device=device)
        evo = Neuroevolution(constructor, population_size=20, device=device, elitism=2, seed=42)
    else:
        constructor = lambda: NumpyMLP([4,16,2], activation='tanh')
        evo = Neuroevolution(constructor, population_size=20, seed=42)
    # define a synthetic eval function: prefer parameter vectors with larger sum of squares (arbitrary)
    def eval_fn(individual):
        # individual is a network instance; compute simple metric using a fixed input
        import numpy as _np
        if TORCH_AVAILABLE:
            x = torch.randn(10,4, device=individual.device)
            with torch.no_grad():
                out = individual(x).cpu().numpy()
            return float((out.mean(axis=0)**2).sum())
        else:
            x = _np.random.randn(10,4).astype(_np.float32)
            out = individual.forward(x)
            return float((out.mean(axis=0)**2).sum())

    # create operators
    sel = TournamentSelection(k=10, tournament_size=3) if TORCH_AVAILABLE else TournamentSelection(10,3)
    mut1 = AdditiveMutation(mutation_rate=0.15, sigma=0.05)
    mut2 = GlobalMutation(mutation_rate=0.03, magnitude=0.2)
    cross = XPointCrossover(x_points=2)

    for gen in range(5):
        fitness, best_idx, best_val = evo.step(eval_fn, selection_fn=sel, mutation_ops=[mut1, mut2], crossover_op=cross)
        print(f"Gen {gen:2d} best fitness: {best_val:.6f} best idx: {int(best_idx)} mean fitness: {float(fitness.mean()):.6f}")

def _demo_gymnasium_env(env_name='CartPole-v1', generations=5, pop_size=50):
    try:
        import gymnasium as gym
    except Exception as e:
        print("gymnasium not available:", e)
        return _demo_synthetic()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obs_space = gym.make(env_name).observation_space.shape[0]
    action_space = gym.make(env_name).action_space.n
    constructor = lambda: TorchMLP([obs_space, 32, action_space], activation='tanh', device=device)
    evo = Neuroevolution(constructor, population_size=pop_size, device=device, elitism=2, seed=123)

    def eval_fn(individual):
        # run deterministic episodes and accumulate reward
        env = gym.make(env_name)
        total = 0.0
        obs, _ = env.reset(seed=0)
        done = False
        steps = 0
        while not done and steps < 500:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=individual.device).unsqueeze(0)
            with torch.no_grad():
                logits = individual(obs_t).cpu().squeeze(0).numpy()
            # pick action greedily
            action = int(logits.argmax())
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total += reward
            steps += 1
        env.close()
        return float(total)

    sel = RouletteWheelSelection(k=pop_size//2)
    mut = AdditiveMutation(0.1, 0.02)
    cross = XPointCrossover(1)
    for gen in range(generations):
        fitness, best_idx, best_val = evo.step(eval_fn, selection_fn=sel, mutation_ops=[mut], crossover_op=cross)
        print(f"Gen {gen:2d} best fitness: {best_val:.2f} mean fitness: {float(fitness.mean()):.2f}")

# When run as script, attempt gym demo; fallback to synthetic
if __name__ == '__main__':
    print("Neuroevolution demo module. Trying gymnasium demo (CartPole). If gym not available, running synthetic demo.")
    try:
        _demo_gymnasium_env('CartPole-v1', generations=50, pop_size=20)
    except Exception as e:
        print('Gym demo failed (or gym absent):', e)
        _demo_synthetic()
