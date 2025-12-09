import gymnasium as gym
from neuroevo_torch import (
    BatchedMLPPolicy, Neuroevolution,
    BestSelection, RouletteWheelSelection,
    AdditiveMutation, XPointCrossover, GlobalMutation, RandomSelection
)
from neuroevo_torch import *
import neuroevo_torch as ne


env_id = "BipedalWalker-v3"
tmp = gym.make(env_id)
obs_dim = tmp.observation_space.shape[0]
act_dim = tmp.action_space.shape[0]
low, high = tmp.action_space.low, tmp.action_space.high
tmp.close()

policy = BatchedMLPPolicy(
    obs_dim=obs_dim, hidden_layers=[64, 64], act_dim=act_dim,
    activation="tanh", action_space="continuous",
    action_low=low, action_high=high
)

def make_env(): return gym.make(env_id)

# A single steady-state pipeline: Selection -> (Crossover, Mutation ...)
ops = [
    TournamentSelection(k=8, tournament_size=3, minimize=False),
    XPointCrossover(x_points=1),
    AdditiveMutation(rate=0.1, sigma=0.05),
]

async_ea = AsyncNeuroevolution(
    policy=policy,
    population_size=512,
    operators=ops,
    seed=42,
    tb_log_dir="runs/exp_async"         # <--- enable TensorBoard logging
)

# 2) Bootstrap the population so selection has signal
evaluator = make_gym_vector_evaluator_async_bootstrap(
    env_fn=make_env,              # your callable that returns a single env
    policy=policy,
    max_steps=512,
    discrete=False,
    vec_size=64,                  # run in waves of 64 for bootstrap
    rollouts_per_genome=1,
)
async_ea.bootstrap_population(evaluator)

# 3) Create an adapter (Gym or Isaac) and run async steady-state
adapter = GymAdapter([make_env for _ in range(64)], discrete=False, dtype=policy.dtype)

stats = async_ea.run_async(env_adapter=adapter, max_steps=512, total_evaluations=5000)

print("Final best:", stats.best_fitness)
