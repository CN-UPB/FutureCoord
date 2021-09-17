from typing import List, Dict
from functools import cmp_to_key

import numpy as np
import scipy.stats as stats
from numpy.random import default_rng, BitGenerator
from tick.base import TimeFunction
from tick.hawkes import SimuInhomogeneousPoisson


class Request:
    def __init__(self, arrival: float, duration: float, datarate: float, max_latency: float, endpoints: tuple, service: int):
        self.arrival = arrival
        self.duration = duration
        self.datarate = datarate
        self.max_latency = max_latency

        self.ingress, self.egress = endpoints
        self.ingress = int(self.ingress)
        self.egress = int(self.egress)
        self.service: int = int(service)
        self.vtypes: List[int] = None
        self.resd_lat: float = None

    def __str__(self):
        attrs = [round(self.duration, 2), round(self.datarate, 2), round(self.resd_lat, 2), round(self.max_latency, 2)]
        attrs = [self.ingress, self.egress, *attrs, self.service]

        return 'Route: ({}-{}); Duration: {}; Rate: {}; Resd. Lat.: {}; Lat.: {}; Service: {}'.format(*attrs)


class ServiceTraffic:
    def __init__(self, rng: BitGenerator, service: int, horizon: float, process: Dict, datarates: Dict, latencies: Dict, endpoints: np.ndarray, rates: np.ndarray, spaths: Dict):
        self.rng = rng
        self.MAX_SEED = 2**30 - 1

        self.service = service
        self.horizon = horizon
        self.process = process
        self.datarates = datarates
        self.latencies = latencies
        self.endpoints = endpoints
        self.spaths = spaths

        # create time function for inhomogenous poisson process
        T = np.linspace(0.0, horizon - 1, horizon)
        rates = np.ascontiguousarray(rates)
        self.rate_function = TimeFunction((T, rates))

    def sample_arrival(self, horizon):
        poi_seed = self.rng.integers(0, self.MAX_SEED)
        poi_seed = int(poi_seed)

        in_poisson = SimuInhomogeneousPoisson(
            [self.rate_function], end_time=horizon, verbose=False, seed=poi_seed)
        in_poisson.track_intensity()
        in_poisson.simulate()
        arrivals = in_poisson.timestamps[0]
        return arrivals

    def sample_duration(self, size):
        mduration = self.process['mduration']
        duration = self.rng.exponential(scale=mduration, size=size)

        return duration

    def sample_datarates(self, size):
        mean = self.datarates['loc']
        scale = self.datarates['scale']
        a, b = self.datarates['a'], self.datarates['b']

        a, b = (a - mean) / scale, (b - mean) / scale
        datarates = stats.truncnorm.rvs(a, b, mean, scale, size=size, random_state=self.rng)
        
        return datarates

    def sample_latencies(self, propagation: np.ndarray):
        mean = self.latencies['loc']
        scale = self.latencies['scale']
        a, b = self.latencies['a'], self.latencies['b']

        a, b = (a - mean) / scale, (b - mean) / scale
        lat = stats.truncnorm.rvs(a, b, mean, scale, size=propagation.size, random_state=self.rng)
        # scale maximum end-to-end latencies (given by shortest path propagation delay) with sampled factor 
        lat = lat * propagation

        return lat

    def sample_endpoints(self, arrivals):
        ingresses, egresses = [], []

        for arrival in arrivals:
            # get endpoint probability matrix for respective timestep
            timestep = int(np.floor(arrival))
            prob = self.endpoints[timestep]

            # sample ingress / egress from probability matrix
            flatten = prob.ravel()
            index = np.arange(flatten.size)
            ingress, egress = np.unravel_index(
                self.rng.choice(index, p=flatten), prob.shape)
            ingresses.append(ingress)
            egresses.append(egress)

        return ingresses, egresses

    def sample(self):
        # sample parameters for each service from distribution functions
        arrival = self.sample_arrival(self.horizon)
        duration = self.sample_duration(len(arrival))
        ingresses, egresses = self.sample_endpoints(arrival)

        # use arrival time to index the endpoint probability matrix and traffic matrix
        rates = self.sample_datarates(size=len(arrival))
        propagation = np.asarray([self.spaths[ingr][egr] for ingr, egr in zip(ingresses, egresses)])
        latencies = self.sample_latencies(propagation)

        # build request objects and append them to the traffic trace
        requests = []
        for arr, dr, rate, lat, ingr, egr in zip(arrival, duration, rates, latencies, ingresses, egresses):
            req = Request(arr, dr, rate, lat, (ingr, egr), self.service)
            requests.append(req)

        return requests


class Traffic:
    def __init__(self, processes):
        self.processes = processes

    def sample(self):
        # generate requests for each type of service from respective processes
        requests = [process.sample() for process in self.processes]
        requests = [req for srequests in requests for req in srequests]

        # sort to-be-simulated service requests according to their arrival time
        requests = sorted(requests, key=cmp_to_key(
            lambda r1, r2: r1.arrival - r2.arrival))
        return requests

    def __iter__(self):
        trace = self.sample()
        return iter(trace)


class TrafficStub:
    def __init__(self, trace):
        self.trace = trace

    def sample(self):
        return self.trace

    def __iter__(self):
        return iter(self.trace)
