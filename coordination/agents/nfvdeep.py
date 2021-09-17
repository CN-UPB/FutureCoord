from itertools import chain
from operator import itemgetter
from collections import defaultdict

import numpy as np
from gym import spaces

from coordination.environment.deployment import ServiceCoordination

class NFVdeepCoordination(ServiceCoordination):
    COMPUTE_UNIT_COST = 0.2
    MEMORY_UNIT_COST = 0.2
    DATARATE_UNIT_COST = 6.0 * 1e-4
    # worked best in our experiments; set similar to threshold in MAVEN-S
    REVENUE = 5.0 
    
    def __init__(self, net_path, process, vnfs, services):
        super().__init__(net_path, process, vnfs, services)

        # observation space of NFVdeep simulation environment
        self.OBS_SIZE = 3 * len(self.net.nodes) + 6
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.OBS_SIZE,), dtype=np.float16)

    def compute_state(self) -> np.ndarray:
        if self.done:
            return np.zeros(self.OBS_SIZE)

        # (1) encode remaining compute resources
        computing = [c / self.MAX_COMPUTE for c in self.computing.values()] 
        
        # (2) encode remaining memory resources
        memory = [m / self.MAX_MEMORY for m in self.memory.values()]

        # (3) encode remaining output datarate
        MAX_OUTPUT = self.MAX_DEGREE * max(data['datarate'] for _, _, data in self.net.edges(data=True))
        output_rates = defaultdict(float)
        for src in self.net.nodes:
            for trg in self.net.nodes:
                if frozenset({src, trg}) in self.datarate:
                    output_rates[src] += self.datarate[frozenset({src, trg})]

        output_rates = list(itemgetter(*self.net.nodes)(output_rates))
        output_rates = [rate / MAX_OUTPUT for rate in output_rates]

        # (4) encode request specific properties
        rate = self.request.datarate / self.MAX_LINKRATE
        resd_lat = self.request.resd_lat / 100.0
        num_components = (len(self.request.vtypes) - len(self.vtype_bidict.mirror[self.request])) / max(len(s) for s in self.services)

        # resource consumption depend on placement decisions; use the mean resource demand 
        cdemands, mdemands = [], []
        vnum = len(self.vtype_bidict.mirror[self.request])
        vtype = self.request.vtypes[vnum]
        config = self.vnfs[vtype]
        
        for node in self.net.nodes:
            supplied_rate = sum([service.datarate for service in self.vtype_bidict[(node, vtype)]])
            after_cdem, after_mdem = self.score(supplied_rate + self.request.datarate, config)
            prev_cdem, prev_mdem = self.score(supplied_rate, config)
            cdemand = np.clip((after_cdem - prev_cdem) / self.MAX_COMPUTE, a_min=0.0, a_max=1.0)
            mdemand = np.clip((after_mdem - prev_mdem) / self.MAX_MEMORY, a_min=0.0, a_max=1.0)

            cdemands.append(cdemand)
            mdemands.append(mdemand)

        cdemand = np.mean(cdemands) / self.MAX_COMPUTE
        mdemand = np.mean(mdemands) / self.MAX_MEMORY
        duration = self.request.duration / 100
        request = [rate, resd_lat, num_components, cdemand, mdemand, duration]


        state = chain(computing, memory, output_rates, request)
        return np.asarray(list(state))

    def compute_reward(self, finalized, deployed, req) -> float:
        if deployed:
            cresources = np.asarray([data['compute'] for node, data in self.net.nodes(data=True)])
            cavailable = np.asarray([self.computing[node] for node in self.net.nodes])
            ccost = np.sum(((cresources - cavailable) > 0) * cresources) * self.COMPUTE_UNIT_COST / self.MAX_COMPUTE

            mresources = np.asarray([data['memory'] for node, data in self.net.nodes(data=True)])
            mavailable = np.asarray([self.memory[node] for node in self.net.nodes])
            mcost = np.sum(((mresources - mavailable) > 0) * mresources) * self.MEMORY_UNIT_COST / self.MAX_MEMORY

            dresources = np.asarray([data['datarate'] for src, trg, data in self.net.edges(data=True)])
            davailable = np.asarray([self.datarate[frozenset({src, trg})] for src, trg in self.net.edges])
            dcost = np.sum(((dresources - davailable) > 0) * dresources) * self.DATARATE_UNIT_COST / self.MAX_LINKRATE

            # in our setting, the revenue is the same for any request 
            return self.REVENUE - (ccost + mcost + dcost)

        return 0.0
