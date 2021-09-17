import os
import sys
import yaml
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from munch import munchify, unmunchify
from stable_baselines3.common.monitor import Monitor

import coordination.evaluation.utils as utils
from coordination.environment.traffic import TrafficStub
from coordination.evaluation.monitor import CoordMonitor
from coordination.environment.deployment import ServiceCoordination


NUM_DAYS = 167


parser = argparse.ArgumentParser()
# arguments to specify the experiment, agent & evaluation 
parser.add_argument('--experiment', type=str, default='./data/experiments/abilene/trace.yml')
parser.add_argument('--agent', type=str, default='./data/configurations/random.yml')
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--logdir', type=str, default='./results/')
parser.add_argument('--seed', type=int, default=0)

# arguments to specify properties of simulation process
parser.add_argument('--oracle', dest='oracle', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    # load agent configuration from file
    with open(args.agent, 'r') as file:
        config = yaml.safe_load(file)
        config = munchify(config)

    # setup folders for logging & results
    logdir = Path(args.logdir)
    log_train, log_eval, log_results = logdir / 'train', logdir / 'evaluate', logdir  / 'results'
    log_train.mkdir(parents=True)
    log_eval.mkdir(parents=True)
    log_results.mkdir(parents=True)

    # load experiment configuration from file
    with open(Path(args.experiment)) as file:
        exp = yaml.safe_load(file)
        exp = munchify(exp)

    # load VNF configurations from file 
    with open(exp.vnfs) as file:
        vnfs = pd.read_csv(file)
        vnfs = [config.to_dict() for _, config in vnfs.iterrows()] 

    # load service configurations from files
    services = []
    for service in exp.services:
        with open(service) as file:
            service = yaml.safe_load(file)
            service = munchify(service)
            services.append(service)
    
    # save arguments, experiment configuration and agent configuration in result folder
    with open(logdir / 'summary.yml', 'w') as file:
        evaluation = {'experiment': unmunchify(exp), 'agent': unmunchify(config), 'args': vars(args)}
        yaml.dump(evaluation, file)    

    for ep in range(args.episodes):
        train_rng = np.random.default_rng(seed=sys.maxsize - ep)
        sim_rng = np.random.default_rng(seed=sys.maxsize - ep)
        eval_rng = np.random.default_rng(seed=ep)

        # determine to-be-used day for endpoint probability matrix 
        eday = rng.integers(0, NUM_DAYS) 

        # determine to-be used days for arrival rates of each service; setup service traffic
        sdays = rng.integers(0, NUM_DAYS, size=len(services) + 1)
        train_process = utils.setup_process(train_rng, exp, services, eday, sdays, exp.load, exp.datarate, exp.latency)        
        eval_process = utils.setup_process(eval_rng, exp, services, eday, sdays, exp.load, exp.datarate, exp.latency)
        eval_process = TrafficStub(eval_process.sample())
        sim_process = utils.setup_sim_process(rng, sim_rng, exp, args, eval_process, services, eday, sdays, exp.sim_load, exp.sim_datarate, exp.sim_latency)


        # setup training environment where traffic is seeded with `train_rng` random number generator
        chains = [service.vnfs for service in services]
        env = utils.setup_simulation(config, exp.overlay, train_process, vnfs, chains)
        env.logger.disabled = True
        monitor = Monitor(env, str(log_train))

        # setup and train agent on environmnet according to configuration
        config.policy.tensorboard_log = log_train
        agent = utils.setup_agent(config, monitor, seed=ep)
        agent.learn(**unmunchify(config.train))

        # setup evaluation environment with traffic seeded by `eval_rng`
        env.replace_process(eval_process)
        monitor = CoordMonitor(ep, config.name, env, log_eval)

        # evaluate agent on evaluation environment
        ep_results = utils.evaluate_episode(agent, monitor, sim_process)
        ep_results = {ep: ep_results}
        utils.save_ep_results(ep_results, log_results)



        







    
        

        

            

        
