import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from experiment_setup import ROOT_DIR
from pathlib import Path
import matplotlib

matplotlib.rcParams['savefig.dpi'] = 600
# matplotlib.rcParams['savefig.dpi'] = 2 * matplotlib.rcParams['savefig.dpi']

#
# dataset = 'MNIST'
# dataset = 'CIFAR-10'
dataset = 'SVHN'

file = Path(ROOT_DIR) / 'experiments' / 'data' / f'mnist_beauty_simple_conv_100_20.csv'
df = pd.read_csv(file)
df = df.replace('mc_dropout', 'MC dropout')
df = df.replace('decorrelating_sc', 'decorrelation')
print(df)
df = df[df['Method'] != 'k_dpp_noisereg']
df2 = df[df.Method.isin(['random', 'error_oracle', 'max_entropy'])]
df3 = df[~df.Method.isin(['random', 'error_oracle', 'max_entropy'])]
df4 = pd.concat((df3, df2))

fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(left=0.1, right=0.95)

with sns.axes_style('whitegrid'):
    sns.lineplot('Step', 'Accuracy', hue='Method', data=df4)
plt.title(f'Active learning on MNIST')

ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.savefig(f'experiments/data/results/al.png')
plt.show()