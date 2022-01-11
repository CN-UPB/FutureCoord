[![CI](https://github.com/CN-UPB/FutureCoord/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/CN-UPB/FutureCoord/actions/workflows/python-package-conda.yml)

# Use What You Know: Network and Service Coordination Beyond Certainty

This repository holds the implementation of FutureCoord, presented in our paper "Use What You Know: Network and Service Coordination Beyond Certainty" ([author version](https://ris.uni-paderborn.de/download/29220/29222/author_version.pdf)) accepted at [2022 IEEE/IFIP Network Operations and Management Symposium](https://noms2022.ieee-noms.org/). 
FutureCoord combines Monte Carlo Tree Search with Traffic Forecasts for Online Orchestration of Network Services.

## Citation

If you use this code, please cite our paper ([author version](https://ris.uni-paderborn.de/download/29220/29222/author_version.pdf)):

```
@inproceedings{werner2022futurecoord,
	title={Use What You Know: Network and Service Coordination Beyond Certainty},
	author={Werner, Stefan and Schneider, Stefan and Karl, Holger},
	booktitle={IEEE/IFIP Network Operations and Management Symposium (NOMS)},
	year={2022},
	publisher={IEEE/IFIP}
}
```

## Setup
Assuming an Anaconda distribution has already been installed, the environment can simply be cloned via ``conda env create -f environment.yml``. We tested this setup on an Ubuntu 18.04 machine with Intel Xeon E5-2695v4@2.1GHz CPUs and 64GB RAM.

## Execution
The ``script.py`` file serves as an interface to running any experiment. It can be used to specify what experiment to run, what service coordinator to use, how many episodes to evaluate and where evaluation files are saved. The interface is as follows:
```console
python script.py
  --experiment data/experiments/<topology>/trace.yml
 --agent data/configurations/<coordinator>.yml 
 --logdir <path to logdir>
 --episodes <#evaluation episodes>
 --seed <random seed>
```

The experiment and algorithm configurations are saves as `*.yml` files under `data/experiments` and `data/configurations`, respectively. Each experiment logs summary files of its evaluation, i.e. experiment, algorithm, etc.,  as `summary.yml` in the `<logdir>`. Monitoring information including the obtained completion ratio as well as resource utilizations are logged as `results.csv`.

## Scenarios
In our evaluation, we vary the expected data rate, max. delay bound and arrival rates for all kinds of flows via the `scenarios/flows.py` file. It can be called as follows:

```console
python scenarios/flows.py 
  --experiment data/experiments/<topology>/trace.yml
  --agent data/configurations/<coordinator>.yml
  --logdir <path to logdir>
  --episodes <#evaluation episodes>
  --seed <random seed>
  --property <datarate / latency / load>
  --factors <varies mean of distributions>
  --sim_factors <varies mean of distributions for forecasts>
  --traffic <accurate / erroneous> 
  --pool <number of jobs>
```
The parameter ``--factors`` varies the flow distributions' mean values proportionally either in terms of their latency, data rates or arrival rates (``--property``).  Similarly, parameter ``--sim_factors`` varies the forecast distributions' mean values. Whether or not the forecast flows follow the correct pattern of arrival rates (or another episode's) is decided by the ``--traffic`` parameter. Similarly, the files `scenarios/network.py` and `scenarios/searches.py` define interfaces to vary either the compute and link capacities or the number of search iterations performed by FutureCoord.
