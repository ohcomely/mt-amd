#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import re
import seaborn as sns
import sys

dataset = [
  'pre2',
  'nd24k',
  'ldoor',
  'dielFilterV3real',
  'Flan_1565',
  'HV15R',
  'Queen_4147',
  'stokes',
  'nlpkkt240',
]

def scaling(scratch_dir):
  def parse(fp):
    with open(fp, 'r') as f:
      log = f.read()
      fill = int(re.findall(r'Fill = (.*)', log)[0])
      time = float(re.findall(r'Time = (.*) seconds', log)[0])
      return fill, time

  df = pd.DataFrame()
  seed = [1, 2, 3, 4, 5]
  for matrix in dataset:
    data = {}
    
    
    famd = np.median([parse(f'{scratch_dir}/dataset/{matrix}/{matrix}.amd.scaling.s{s}.log')[0] for s in seed])
    tamd = np.median([parse(f'{scratch_dir}/dataset/{matrix}/{matrix}.amd.scaling.s{s}.log')[1] for s in seed])
    f1 = np.median([parse(f'{scratch_dir}/dataset/{matrix}/{matrix}.paramd.scaling.s{s}.t1.log')[0] for s in seed])
    t1 = np.median([parse(f'{scratch_dir}/dataset/{matrix}/{matrix}.paramd.scaling.s{s}.t1.log')[1] for s in seed])
    f64 = np.median([parse(f'{scratch_dir}/dataset/{matrix}/{matrix}.paramd.scaling.s{s}.t64.log')[0] for s in seed])
    t64 = np.median([parse(f'{scratch_dir}/dataset/{matrix}/{matrix}.paramd.scaling.s{s}.t64.log')[1] for s in seed])
    df[matrix] = {
      'SuiteSparse time (sec)': f'{tamd:.2f}',
      'ParAMD 1t time (sec)': f'{t1:.2f}',
      'ParAMD 64t time (sec)': f'{t64:.2f}',
      'ParAMD 1t Speedup over SuiteSparse': f'{tamd/t1:.2f}x',
      'ParAMD 64t Speedup over SuiteSparse': f'{tamd/t64:.2f}x',
      r'SuiteSparse #Fill-ins': f'{famd:.2e}',
      r'ParAMD 1t #Fill-ins': f'{f1:.2e}',
      r'ParAMD 64t #Fill-ins': f'{f64:.2e}',
      'ParAMD 1t Fill-in Ratio over SuiteSparse': f'{f1/famd:.2f}x',
      'ParAMD 64t Fill-in Ratio over SuiteSparse': f'{f64/famd:.2f}x',
    }
  df = df.T
  savepath = scratch_dir / 'plots' / 'scaling.csv'
  print(f'Saving scaling table to: {savepath}')
  df.to_csv(savepath)
  savepath = scratch_dir / 'plots' / 'scaling.html'
  print(f'Saving scaling table to: {savepath}')
  df.to_html(savepath)

def breakdown(scratch_dir):
  def parse(fp):
    with open(fp, 'r') as f:
      log = f.read()
      aat = float(re.findall(r'A \+ AT: (.*) seconds', log)[0])
      core = float(re.findall(r'Core: (.*) seconds', log)[0])
      dist2 = float(re.findall(r'Distance-2 Independent Sets: (.*) seconds', log)[0])
      other = float(re.findall(r'Other: (.*) seconds', log)[0])
    return [aat, core, dist2, other]
  label = ['A+A^T', 'Core', 'Distance-2 Independent Sets', 'Other']
  color = sns.color_palette('colorblind')
  fig, ax = plt.subplots(3, 3, figsize = (32, 24))

  nt = [1, 2, 4, 8, 16, 32, 64]
  xticks = [1, 2, 3, 4, 5, 6, 7]
  for ind, matrix in enumerate(dataset):
    i  = ind // 3
    j = ind % 3
    bottom = np.zeros(7)
    res = [parse(f'{scratch_dir}/dataset/{matrix}/{matrix}.paramd.breakdown.s1.t{t}.log') for t in nt]
    res = np.array(res).T
    for k in range(res.shape[0]):
      ax[i, j].bar(xticks, res[k], 1, lw = 3, edgecolor = 'black', 
                   label = label[k], color = color[k], bottom = bottom)
      bottom += res[k]
    ax[i, j].set_xticks(xticks, nt)
    ax[i, j].set_title(matrix, y = 1.02, size = 26)
    ax[i, j].set_xlim([0, 8])
    ax[i, j].set_xlabel('Number of Threads', fontsize = 26)
    ax[i, j].set_ylabel('Time (sec)', fontsize = 26)
    ax[i, j].tick_params(axis = 'both', labelsize = 26)
    ax[i, j].grid(axis = 'y')
 
  handles, labels = ax[0, 0].get_legend_handles_labels()
  fig.legend(handles, labels, loc = 'lower center', ncols = 4, bbox_to_anchor=(0.5, -0.03), fontsize = 26)
  savepath = scratch_dir / 'plots' / 'breakdown.png'
  print(f'Saving breakdown plot to: {savepath}')
  plt.tight_layout()
  plt.savefig(savepath, bbox_inches="tight")

def distribution(scratch_dir):
  def parse(fp):
    with open(fp, 'r') as f:
      log = f.read()
      return [int(x) for x in re.findall(r'Size of distance-2 independent sets: \[(.*), \]', log)[0].split(',')]
  
  data = [parse(f'{scratch_dir}/dataset/{matrix}/{matrix}.paramd.distribution.s1.t64.log') for matrix in dataset]

  fig, ax = plt.subplots(figsize=(32, 8))
  sns.violinplot(data=data, ax=ax, log_scale=True, palette = "colorblind", cut = 0)
  ax.set_ylabel('Size of Distance-2 Independent Sets', fontsize = 26)
  ax.set_xticks(range(len(dataset)), dataset, fontsize = 26)
  ax.tick_params(axis = 'y', labelsize = 26)
  
  ax.axhline(y = 64, color = 'black', linestyle='--', lw = 2)
  ax.grid(axis = 'y')
  plt.tight_layout()
  savepath = scratch_dir / 'plots' / 'distribution.png'
  print(f'Saving distribution plot to: {savepath}')
  plt.savefig(savepath)

def tuning(scratch_dir):
  def parse(fp):
    with open(fp, 'r') as f:
      log = f.read()
      core = float(re.findall(r'Core: (.*) seconds', log)[0])
      dist2 = float(re.findall(r'Distance-2 Independent Sets: (.*) seconds', log)[0])
      fill = int(re.findall(r'Fill = (.*)', log)[0])
      return core, dist2, fill

  fig, ax = plt.subplots(2, 3, figsize = (32, 16))
  mult_vals = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
  lim_vals = [8, 32, 128, 512, 2048]
  base = [10 ** 8, 10 ** 12]
  for ind, matrix in enumerate(['nd24k', 'nlpkkt240']):
    core, dist2, fill = [], [], []
    for lim in lim_vals:
      core.append([]), dist2.append([]), fill.append([])
      for mult in mult_vals:
          c, d, f = parse(f'{scratch_dir}/dataset/{matrix}/{matrix}.paramd.tuning.s1.t64.m{mult}.l{lim * 64}.log')
          core[-1].append(c), dist2[-1].append(d), fill[-1].append(f / base[ind])
    sns.heatmap(core, ax = ax[ind, 0], xticklabels = mult_vals, yticklabels = lim_vals, annot_kws = {"fontsize": 24}, 
                annot=True, fmt = f'.{3-ind}f', cmap = sns.cm.rocket_r, linewidth = 1.5)
    sns.heatmap(dist2, ax = ax[ind, 1], xticklabels = mult_vals, yticklabels = lim_vals, annot_kws = {"fontsize": 24},
                annot=True, fmt = f'.{3-ind}f', cmap = sns.cm.rocket_r, linewidth = 1.5)
    sns.heatmap(fill, ax = ax[ind, 2], xticklabels = mult_vals, yticklabels = lim_vals, annot_kws = {"fontsize": 24},
                annot=True, fmt = '.2f', cmap = sns.cm.rocket_r, linewidth = 1.5)
    for j in range(3):
      cbar = ax[ind, j].collections[0].colorbar.ax.tick_params(labelsize = 24)
      ax[ind, j].invert_yaxis()
      ax[ind, j].tick_params(axis = 'y', labelrotation=0)
      ax[ind, j].set_xlabel('mult', fontsize = 24)
      ax[ind, j].set_ylabel('lim', fontsize = 24)
      ax[ind, j].tick_params(axis = 'both', labelsize = 24)
    ax[ind, 0].set_ylabel(f'{matrix}\n\nlim', fontsize = 24)
  ax[0, 0].set_title('Time spent on core AMD (sec)', size = 24)
  ax[0, 1].set_title('Time spent on dist-2 indep sets (sec)', size = 24)
  ax[0, 2].set_title(r'#Fill-ins ($\times10^8$)', size = 24)
  ax[1, 0].set_title('Time spent on core AMD (sec)', size = 24)
  ax[1, 1].set_title('Time spent on dist-2 indep sets (sec)', size = 24)
  ax[1, 2].set_title(r'#Fill-ins ($\times10^{12}$)', size = 24)
  plt.tight_layout()
  savepath = scratch_dir / 'plots' / 'tuning.png'
  print(f'Saving tuning plot to: {savepath}')
  plt.savefig(savepath)

def main():
  if len(sys.argv) < 3:
    print('Usage: python plot.py /path/to/a/clean/scratch/dir/ {scaling, breakdown, distribution, tuning}')
    exit(1)
  scratch_dir = Path(sys.argv[1])
  if sys.argv[2] == 'scaling':
    scaling(scratch_dir)
  elif sys.argv[2] == 'breakdown':
    breakdown(scratch_dir)
  elif sys.argv[2] == 'distribution':
    distribution(scratch_dir)
  elif sys.argv[2] == 'tuning':
    tuning(scratch_dir)
  else:
    print(f'Unknown option {sys.argv[2]}')
    exit(1)

if __name__ == '__main__':
  main()