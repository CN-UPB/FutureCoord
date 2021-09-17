import os
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--output', type=str, default='./results')
    args = parser.parse_args()

    index_mapping = {'agent': 'Agent', 'episode': 'Episode'}

    measure_mapping = {'reward': 'Reward',
                       'acceptance_rate': 'Acceptance Rate'}

    results = pd.DataFrame()

    dirs = [directory for directory in os.listdir(args.logdir)]
    tables = [Path(args.logdir) / directory / 'results' /
              'results.csv' for directory in dirs]
    tables = [table for table in tables if table.exists()]
    for table in tables:
        data = pd.read_csv(table)
        results = pd.concat((results, data))

    results = results.rename(columns={**index_mapping, **measure_mapping})
    results = results.reset_index()

    sns.set_style("whitegrid")
    for measure in measure_mapping.values():
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.boxplot(x='Agent', y=measure, data=results, ax=ax)
        sns.despine()
        fig.savefig(Path(args.output) / f'{measure}.pdf')