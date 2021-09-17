import os
import re
import yaml
import argparse
import subprocess
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool

import networkx as nx
from munch import munchify, unmunchify


def spawn(ns):
    # call `script.py` to evaluate agent configuration in subprocess
    command = f'python script.py --experiment {ns.experiment} --agent {ns.agent} --episodes {ns.episodes} --logdir {ns.logdir} --seed {ns.seed}'
    subprocess.run([command], shell=True) 


parser = argparse.ArgumentParser()
parser.add_argument('--compute', nargs='+', type=float)
parser.add_argument('--datarate', nargs='+', type=float)

# define parameters of to-be-evaluated experiment
parser.add_argument('--experiment', type=str, default='./data/experiments/abilene/trace/trace.yml')
parser.add_argument('--agent', type=str, default='./data/configurations/random.yml')
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--logdir', type=str, default='./results/')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--pool', type=int, default=1)



if __name__ == '__main__':
    args = parser.parse_args()
    logdir = Path(args.logdir)

    # load base experiment configuration from file
    with open(args.experiment, 'r') as file:
        experiment = munchify(yaml.safe_load(file))

    # load to-be-evaluated agent configuration from file
    with open(args.agent, 'r') as file:
        agent = munchify(yaml.safe_load(file))

    # load network configuration from file
    overlay = nx.read_gpickle(experiment.overlay)

    # modify namespace for experiment
    namespace = deepcopy(args)
    del namespace.compute
    del namespace.datarate
    del namespace.pool 

    for compute, datarate in zip(args.compute, args.datarate):
        path = logdir / f'compute_{compute}_datarate_{datarate}'
        path.mkdir()

        exp = deepcopy(experiment)
        exp.overlay = str(path / 'overlay.gpickle')

        with open(path / 'experiment.yml', 'w') as file:
            yaml.dump(unmunchify(exp), file)    

        with open(path / 'agent.yml', 'w') as file:
            yaml.dump(unmunchify(agent), file)    

        network = deepcopy(overlay)
        for node in network.nodes:
            network.nodes[node]['compute'] = compute * network.nodes[node]['compute']

        for link in network.edges:
            network.edges[link]['datarate'] = datarate * network.edges[link]['datarate']

        nx.write_gpickle(network, path / 'overlay.gpickle')

        namespace.logdir = str(path)
        namespace.experiment = str(path / 'experiment.yml')
        namespace.agent = str(path / 'agent.yml')

        with open(path / 'args.yml', 'w') as file:
            yaml.dump(vars(namespace), file)    

    runs = []
    for run in os.listdir(str(logdir)):
        with open(logdir / run / 'args.yml', 'r') as file:
            runs.append(munchify(yaml.safe_load(file)))

    with Pool(processes=args.pool) as pool:
        pool.map(spawn, runs)

