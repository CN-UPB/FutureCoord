from copy import deepcopy
from itertools import combinations_with_replacement
from collections import defaultdict

import torch as th
import numpy as np
import numpy.ma as ma
import networkx as nx
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy

from coordination.environment.traffic import Request


class RandomPolicy:
    def __init__(self, seed=None, **kwargs):
        np.random.seed(seed)

    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        """Samples a valid action from all valid actions."""
        valid_nodes = np.asarray([node for node in env.valid_routes])
        return np.random.choice(valid_nodes)


class AllCombinations:
    def __init__(self, **kwargs):
        self.actions = []

    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        if not self.actions:
            self.actions = self._predict(env, **kwargs)
            self.actions = list(self.actions)

        action = self.actions.pop(0)
        return action

    def _predict(self, env, **kwargs):
        position = env.request.ingress
        vtypes = [vnf for vnf in env.request.vtypes]
        nodes = env.net.nodes()        

        # insert upcoming request after to-be-deployed requests, so that environment 
        # does not progress in time immediately after finalized deployment 
        env = deepcopy(env)
        arrival = env.request.arrival + np.nextafter(0, 1)
        stub = Request(arrival, env.request.duration, 0.0, float('inf'), (0, 0), 0)
        env.trace = iter([stub]) 

        compute = sum(env.computing.values())
        memory = sum(env.memory.values())
        capacity = sum(env.datarate.values())
        min_placements = None
        min_score = float('inf')

        # set score coefficients proportional to inverse of available capcaity
        c_max = max(c for _, c in env.net.nodes('compute'))
        c_avail = compute / sum(c for _, c in env.net.nodes('compute'))
        c_alpha = 1 / (c_avail * c_max)

        m_max = max(m for _, m in env.net.nodes('memory'))
        m_util = memory / sum(m for _, m in env.net.nodes('memory'))
        m_alpha = 1 / (m_util * m_max)

        d_max = max(data['datarate'] for _, _, data in env.net.edges(data=True))
        d_util = capacity / sum(data['datarate'] for _, _, data in env.net.edges(data=True))
        d_alpha = 1 / (d_util * d_max)

        for placements in combinations_with_replacement(nodes, len(vtypes)):
            sim_env = deepcopy(env)
            accepts = sum(info.accepts for info in sim_env.info)

            # simulate placement actions on environment
            for num, placement in enumerate(placements):
                if not placement in sim_env.valid_routes.keys():
                    break

                sim_env.step(placement)

            # update assessment only for accepted requests
            if accepts >= sum(info.accepts for info in sim_env.info):
                continue
                
            delta_compute = compute - sum(sim_env.computing.values())
            delta_memory = memory - sum(sim_env.memory.values())
            delta_capacity = capacity - sum(sim_env.datarate.values())

            score = c_alpha * delta_compute + m_alpha * delta_memory + d_alpha * delta_capacity
            if score < min_score:
                min_score = score
                min_placements = placements

        # case: no valid placement action is available: choose REJECT_ACTION
        if min_placements is None:
            return [env.REJECT_ACTION]

        if not isinstance(min_placements, tuple):
            min_placements = (min_placements)

        if min_score < float('inf'):
            return min_placements


class GreedyHeuristic:

    def __init__(self, **kwargs):
        pass

    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        valid_actions = env.valid_routes.keys()
        _, pnode = env.routes_bidict[env.request][-1]

        # compute delays (including processing delays) associated with actions
        lengths, _ = nx.single_source_dijkstra(env.net, source=pnode, weight=env.get_weights, cutoff=env.request.resd_lat)
        
        # choose (valid) VNF placement with min. latency increase
        action = min(valid_actions, key=lengths.get)

        return action

class MaskedPPO(PPO):
    def predict(self, observation: th.Tensor, deterministic: bool = False, env=None, **kwargs):
        if deterministic and not env is None:
            observation = np.asarray(observation).reshape((-1,) + self.env.observation_space.shape)
            observation = th.as_tensor(observation).to('cpu')
            
            # get action mask of valid choices from environment 
            valid_actions = np.full(env.ACTION_DIM, False)
            valid = list(env.valid_routes.keys())
            valid_actions[valid] = True

            latent_pi, _, latent_sde = self.policy._get_latent(observation)
            distribution = self.policy._get_action_dist_from_latent(latent_pi, latent_sde)
            valid, = np.where(valid_actions)

            actions = th.arange(valid_actions.size)
            log_prob = distribution.log_prob(actions).detach().cpu().numpy()

            action = ma.masked_array(log_prob, ~valid_actions, fill_value=np.NINF).argmax()
            return action
        
        action, _ = super(PPO, self).predict(observation, deterministic)
        return action

    def load(self, path, device='auto'):
        # when loading a pre-trained policy from `path`, do nothing upon call to `learn`
        self.policy = MlpPolicy.load(path, device)

        def stub(*args, **kwargs):
            pass

        self.learn = stub