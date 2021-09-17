import os
import yaml
import argparse
import tempfile
import subprocess
from pathlib import Path
from copy import deepcopy
from itertools import product
from multiprocessing import Pool

from munch import munchify, unmunchify


parser = argparse.ArgumentParser()
# define parameters of to-be-evaluated experiment
parser.add_argument('--experiment', type=str, default='./data/experiments/abilene/traffic_trace/trace.yml')
parser.add_argument('--agent', type=str, default='./data/configurations/coord_random.yml')

# arguments to specify properties of simulation process
parser.add_argument('--oracle', dest='oracle', action='store_true')

# define arguments of evaluation 
parser.add_argument('--logdir', type=str, default='./results/')
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--pool', type=int, default=1)

# arguments to specify the to-be-evaluated parameter space
parser.add_argument('--max_searches', nargs='+', type=int)
parser.add_argument('--max_requests', nargs='+', type=int)
parser.add_argument('--sim_discounts', nargs='+', type=float)


def spawn(config, args, logdir):
    policy = config.policy
    logdir = logdir / f'{policy.max_searches}_{policy.max_requests}_{policy.sim_discount}_{args.oracle}'
    logdir.mkdir()

    # write to-be-evaluated agent configuration to temporary file
    tmp = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    yaml.dump(config, tmp)    
    tmp.flush()

    # call `script.py` to evaluate agent configuration in subprocess
    command = f'python script.py --experiment {args.experiment} --agent {tmp.name} --episodes {args.episodes} --logdir {str(logdir)} --seed {args.seed}'
    if args.oracle:
        command += ' --oracle' 
    subprocess.run([command], shell=True)

    # unlink from temporary configuration file
    tmp.close()
    os.unlink(tmp.name)


if __name__ == '__main__':
    args = parser.parse_args()
    logdir = Path(args.logdir)

    # load to-be-evaluated agent configuration from file
    with open(args.agent, 'r') as file:
        agent = munchify(yaml.safe_load(file))

    configs = []
    for max_searches, max_requests, discount in product(args.max_searches, args.max_requests, args.sim_discounts):
        config = deepcopy(agent)
        config.policy.max_searches = max_searches
        config.policy.max_requests = max_requests
        config.policy.sim_discount = discount
        configs.append(config)

    with Pool(processes=args.pool) as pool:
        pool.starmap(spawn, [(config, args, logdir) for config in configs])

