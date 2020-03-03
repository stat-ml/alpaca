import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from experiment_setup import ROOT_DIR
from pathlib import Path
import matplotlib

matplotlib.rcParams['savefig.dpi'] = 150

#
# dataset = 'MNIST'
# dataset = 'CIFAR-10'
# dataset = 'SVHN'

file = Path(ROOT_DIR) / 'experiments'/ 'data'/ f'var_ratio_4_boxplot_{dataset}_50000_150.csv'
df = pd.read_csv(file)
df = df.replace('mc_dropout', 'MC dropout')
df = df.replace('decorrelating_sc', 'decorrelation')
df = df[df['Estimator type'] != 'k_dpp_noisereg']
print(df)
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(right=.95)

with sns.axes_style('whitegrid'):
    sns.boxplot(data=df, y='ROC-AUC score', x='Estimator type', ax=ax)
plt.title(f'{dataset} wrong prediction ROC-AUC')

ax.yaxis.grid(True)
ax.xaxis.grid(True)


plt.savefig(f'experiments/data/results/rocauc_{dataset}_4.png')
plt.show()

