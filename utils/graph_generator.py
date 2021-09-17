import random
import argparse

import numpy as np
import networkx as nx
from geopy.distance import geodesic


def graphml_reader(seed, compute, bandwidth, inputfile, outputfile):
    '''Creates a gpickle graph from a graphml file. The node capacity (compute)
    and the link capacity (bandwidth) are created randomly within the given bounds,
    while the latency is calculated by the distance of the nodes'''

    SPEED_OF_LIGHT = 299792458  # meter per second
    PROPAGATION_FACTOR = 0.77  # https://en.wikipedia.org/wiki/Propagation_delay

    random.seed(seed)
    # setting ranged for random values of the nodes

    file = inputfile
    if not file.endswith(".graphml"):
        raise ValueError("{} is not a GraphML file".format(file))
    network = nx.read_graphml(file, node_type=int)

    newnetwork = nx.Graph()
    mapping = dict()

    num = 0
    for (node, data) in network.nodes(data=True):
        # some nodes are not actually nodes in a sense that a position ect. is given.
        if data['Internal'] == 1:
            mapping[node] = num
            newnetwork.add_node(num, compute=random.uniform(*compute))
            num += 1

    for e in network.edges():
        n1 = network.nodes(data=True)[e[0]]
        n2 = network.nodes(data=True)[e[1]]
        if n1['Internal'] == 0 or n2['Internal'] == 0:
            continue
        n1_lat, n1_long = n1.get("Latitude"), n1.get("Longitude")
        n2_lat, n2_long = n2.get("Latitude"), n2.get("Longitude")
        distance = geodesic((n1_lat, n1_long),
                            (n2_lat, n2_long)).meters  # in meters
        delay = (distance / SPEED_OF_LIGHT * 1000) * \
            PROPAGATION_FACTOR  # in milliseconds

        # This is not normalized
        newnetwork.add_edge(mapping[e[0]], mapping[e[1]], latency=float(
            delay), bandwidth=random.uniform(*bandwidth))

    nx.write_gpickle(newnetwork, outputfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''Creates a gpickle graph from a graphml or gml file.
            The node capacity (compute) and the link capacity (bandwidth)
            are created randomly, while the latency
            is calculated by the distance of the nodes''')

    parser.add_argument('--seed', type=int,  nargs='?',
                        default=0)
    parser.add_argument('--inputfile', type=str, nargs='?',
                        const=1)
    parser.add_argument('--outputfile', type=str, nargs='?',
                        const=1, default=r'./data/network.gpickle')
    args = parser.parse_args()

    # bounds for the resources - should be normalized between 0 and 1
    compute = (0.0, 1.0)
    bandwidth = (0.0, 1.0)

    if args.inputfile.endswith(".graphml"):
        graphml_reader(args.seed, compute, bandwidth,
                       args.inputfile, args.outputfile)
    else:
        raise ValueError("Input not supported. It should be a graphml file")
