# Evaluation Details

This file specifies some of the details of our experimental evaluation. 

## Components
We define six different components. Each component defines how its consumed compute and memory capacity changes given the total throughput it processes. We derive fourth-order polynomials to define components' compute capacity demands from benchmarking data ([SNDZoo](https://sndzoo.github.io/)). The coefficients are stored in the `data/experiments.vnfs.csv` file. We also assign a fixed amount of memory that components require upon instantiation; it remains unchanged when scaled up. 

## Services
We define four different network services composed of 2-4 components. Their configuration is defined in ``data/services/<service>.yml`` files. Each file specifies the service's components and by what kind of flows it is requested. This includes the parameters of the exponentially distributed flow duration times, the flow arrival interarrival times which follow an inhomogeneous Poisson process and the distribution of flow data rates and max. end-to-end delays. At which rates flows enter and depart is derived from traces for the Abilene network using [SNDlib](http://sndlib.zib.de/home.action). Data rates and max. delay bounds are assumed normally distributed with parameters again specified in ``data/services/<service>.yml``. Their values are clipped in between bounds to prevent, e.g., that the max. end-to-end delay is lower than the shortest path's propagation delay. The distribution of flow data rates (mean value & standard deviation) is parameterized proportional to the minimum maximum throughput of its components. 

# Ingresses & Egresses
Where flows enter and depart at varies over time and is also defined using Abilene traces. Specifically, we derive probability matrices from the total amount of data send among nodes within 5 minute periods. Episode then simulate 43 discrete periods each lasting 5 minutes. The arrival rates as well as ingress & egress node probabilities change at each period.