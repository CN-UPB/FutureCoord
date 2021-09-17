import numpy as np


class GRC:
    def __init__(self, damping, alpha, **kwargs):
        self.damping = damping
        self.alpha = alpha

        self.sgrc, self.rgrc =  [], []

    def learn(self, **kwargs):
        pass

    def predict(self, env, **kwargs):
        if not self.rgrc:
            self.rgrc = self.request_grc(env)

        vgrc = self.rgrc.pop(0)

        # in contrast to the original paper, we recompute the substrate GRC vector after every placement decision
        # since their setting assumes (1) resource demands irrespective of placements
        # and (2) more than one VNF instance may be served by the same node   
        self.sgrc = self.substrate_grc(env)

        argsort = sorted(range(len(self.sgrc)), key=self.sgrc.__getitem__)
        argsort.reverse()

        action = next(node for node in argsort if node in env.valid_routes)
        return action

    def substrate_grc(self, env):
        num_nodes = len(env.net.nodes())

        # compute (normalized) remaining computing and memory resources
        compute = np.asarray(list(env.computing.values()))
        max_compute = np.asarray(list(c for _, c in env.net.nodes('compute')))
        compute = compute / np.sum(max_compute)

        memory = np.asarray(list(env.memory.values()))
        max_memory = np.asarray(list(m for _, m in env.net.nodes('memory')))
        memory = memory / np.sum(max_memory)

        # compute aggregated resource vector (accounts for multiple resources)
        resources = self.alpha * compute + (1 - self.alpha) * memory

        # determine datarate transition matrix
        datarate = np.zeros(shape=(num_nodes, num_nodes))
        for u, v, data in env.net.edges(data=True):
            datarate[u, v] = data['datarate']
            datarate[v, u] = data['datarate']
        
        # determince grc vector for substrate network
        total_datarate = np.sum(datarate, axis=0)
        datarate = datarate / total_datarate[:, np.newaxis]
        substrate_grc = (1 - self.damping) * np.linalg.inv(np.eye(num_nodes) - self.damping * datarate) @ resources
        
        return list(substrate_grc)

    def request_grc(self, env):
        num_vnfs = len(env.request.vtypes)

        # in our scenario, requested resources depend on the placement, i.e. consider an aggregation of resource demands
        resources = np.asarray([self._mean_resource_demand(env, env.request, vtype) for vtype in env.request.vtypes])
        resources = resources / np.sum(resources)

        # normalized transition matrix for linear chain of VNFs is the identity matrix
        datarate = np.eye(num_vnfs)
        request_grc = (1 - self.damping) * np.linalg.inv(np.eye(num_vnfs) - self.damping * datarate) @ resources

        return list(request_grc)     

    def _mean_resource_demand(self, env, req, vtype):
        config = env.vnfs[vtype]
        demand = []

        for node in env.net.nodes():
            # compute resource demand after placing VNF of `node`
            supplied_rate = sum([service.datarate for service in env.vtype_bidict[(node, vtype)]])
            after_cdem, after_mdem = env.score(supplied_rate + req.datarate, config)
            prev_cdem, prev_mdem = env.score(supplied_rate, config)

            cincr = (after_cdem - prev_cdem) / env.net.nodes[node]['compute']
            mincr = (after_mdem - prev_mdem) / env.net.nodes[node]['memory']

            # compute aggregated increase (accounts for multiple rather than single resource type)
            incr = self.alpha * cincr + (1 - self.alpha) * mincr

            # filter invalid placements (infinite resource demands)
            if incr >= 0.0:
                demand.append(incr)

        return np.mean(demand)