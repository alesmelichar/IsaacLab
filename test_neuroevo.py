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
policy.to('cpu')


evaluator = Neuroevolution.make_gym_vector_evaluator(
    env_fn=make_env, 
    policy=policy, 
    max_steps=200, 
    discrete=False, 
)

pop_size = 256
operators = [
    [BestSelection(20)],
    [RandomSelection(156), AdditiveMutation(rate=0.1, sigma=0.02)],
    [RouletteWheelSelection(80), XPointCrossover(1), AdditiveMutation(rate=0.05, sigma=0.1)],
]

evo = Neuroevolution(policy, pop_size, operators, evaluator, seed=123)

for gen in range(100):
    stats = evo.step()
    print(f"[Pendulum ] Gen {gen:03d} | best={stats.best_fitness:.2f} | mean={stats.mean_fitness:.2f}")
