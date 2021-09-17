import numpy as np
from munch import unmunchify
from tensorboardX import SummaryWriter
from stable_baselines3.common.monitor import Monitor


class CoordMonitor(Monitor):
    REQUEST_KEYS = ['accepts', 'requests', 'num_invalid', 'num_rejects', 'no_egress_route', 'no_extension', 'skipped_on_arrival']
    ACCEPTED_KEYS = ['cum_service_length', 'cum_route_hops', 'cum_datarate', 'cum_max_latency', 'cum_resd_latency']
    ACCEPTED_VALS = ['mean_service_len', 'mean_hops', 'mean_datarate', 'mean_latency', 'mean_resd_latency']

    def __init__(self, episode, tag, env, filename=None, allow_early_resets=True, reset_keywords=(), infor_keywords=()):
        super().__init__(env, None, allow_early_resets, reset_keywords, infor_keywords)
        self.writer = SummaryWriter(filename)
        self.episode = episode
        self.tag = tag

        self.reset()

    def close(self):
        self.writer.flush()
        self.writer.close()
        super().close()

    def reset(self, **kwargs):
        self.c_util, self.m_util, self.d_util = [], [], []

        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        for service in range(len(self.env.services)):
            logs = unmunchify(self.env.info[service])

            for key in self.REQUEST_KEYS:
                scalar = logs[key] / self.env.num_requests
                tag = f'{self.tag}/{service}/{key}'
                self.writer.add_scalar(tag, scalar, self.episode)
            
            accepts = logs['accepts'] if logs['accepts'] > 0 else np.inf
            for key in self.ACCEPTED_KEYS:
                scalar = logs[key] / accepts
                tag = f'{self.tag}/{service}/{key}'
                self.writer.add_scalar(tag, scalar, self.episode)
        
        self.update_utilization()

        return obs, reward, done, info

    def update_utilization(self):
        nodes = self.env.net.nodes
        cutil = [1 - self.env.computing[n] / self.env.net.nodes[n]['compute'] for n in nodes]
        mutil = [1 - self.env.memory[n] / self.env.net.nodes[n]['memory'] for n in nodes]
        cutil = np.mean(cutil)
        mutil = np.mean(mutil)

        edges = self.env.net.edges
        max_cap = [self.env.net.edges[e]['datarate'] for e in edges]
        edges = [frozenset({*e}) for e in edges]
        cap = [self.env.datarate[e] for e in edges]
        dutil = 1 - np.asarray(cap) / np.asarray(max_cap)
        dutil = np.mean(dutil)

        self.c_util.append(cutil)
        self.m_util.append(mutil)
        self.d_util.append(dutil)

    def get_episode_results(self):
        ep = {}
        info = self.env.info
        info = [unmunchify(slogs) for slogs in info]

        # aggregate results over all provided services
        num_requests = max(1, self.env.num_requests)
        total_accepts = max(1, sum(slogs['accepts'] for slogs in info))
        ep['accept_rate'] = total_accepts / num_requests
        ep['balanced_accept_rate'] = np.prod([logs['accepts'] / logs['requests'] for logs in info])     
        for key, val in zip(self.ACCEPTED_KEYS, self.ACCEPTED_VALS):
            aggr = sum(info[service][key] for service in range(len(info)))
            ep[val] = aggr / total_accepts

        # log service specific results
        for service, logs in enumerate(info):
            for key in self.REQUEST_KEYS:
                ep[f'serivce_{service}_{key}'] = logs[key]

            for key, val in zip(self.ACCEPTED_KEYS, self.ACCEPTED_VALS):
                ep[f'serivce_{service}_{val}'] = logs[key]

        # log utilization of substrate network
        ep['mean_cutil'] = np.mean(self.c_util)
        ep['mean_mutil'] = np.mean(self.m_util)
        ep['mean_dutil'] = np.mean(self.d_util)

        # update information from stable baselines 3 monitor
        ep['ep_return'] = self.get_episode_rewards()[0]
        ep['ep_length'] = self.get_episode_lengths()[0]
        ep['ep_time'] = self.get_episode_times()[0]

        return ep 

