from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import pickle
import sys
from itertools import product

# Data modules
from Data.generate_synthetic import lorentz_96_model_2
from Data.data_processing import format_ts_data, normalize

# Model modules
sys.path.append('../Model')
from bigvar import run_bigvar

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nlambdas', type=int, default=50, help='number of lambda values in grid')
parser.add_argument('--lamratio', type=float, default=1000., help='ratio of largest lambda in grid to smallest')
parser.add_argument('--seed', type=int, default=12345, help='seed')
parser.add_argument('--model_lag', type=int, default=5, help='lag of BigVAR model')

parser.add_argument('--sparsity', type = float, default = 0.3, help = 'sparsity of connections')
parser.add_argument('--sd', type = float, default = 0.1, help = 'standard deviation of noise')
parser.add_argument('--dt', type = float, default = 0.1, help = 'sampling rate')
parser.add_argument('--p', type = int, default = 20, help = 'dimensionality of time series')
parser.add_argument('--T', type = int, default = 1000, help = 'length of time series')
args = parser.parse_args()

# Prepare filename
experiment_base = 'Oscillator_BigVAR'
results_dir = 'Results/' + experiment_base

experiment_name = results_dir + '/scan_expt.out'

# Create directory, if necessary
if not os.path.exists(results_dir):
     os.makedirs(results_dir)

# Verify that experiment doesn't exist
if os.path.isfile(experiment_name):
	print('Skipping experiment')
	sys.exit(0)

# Parameters to scan
T_grid = [500, 750, 1000]
sparsity_grid = [0.2, 0.3, 0.4]
dt_grid = [0.05, 0.1, 0.25, 0.5]

results_list = []

for T, sparsity, dt in product(T_grid, sparsity_grid, dt_grid):

  # generate and prepare data
  X, GC = kuramoto_model(sparsity, args.p, N = T, delta_t = dt, sd = args.sd, num_trials = None)
  X = normalize(X)

  coefs, lambdas, _ = run_bigvar(X, args.model_lag, 'HVARELEM',
                                 nlambdas=args.nlambdas,
                                 lamratio=float(args.lamratio),
                                 T1=0, T2=T-1, use_intercept=True)

  # for each lambda, estimate granger causality
  coef_tnsrs = list()
  for c in coefs:
      c_tnsr = np.empty((args.p, args.p, args.model_lag))
      for l in range(args.model_lag):
          c_tnsr[:,:,l] = c[:, l*args.p:(l+1)*args.p]
      coef_tnsrs.append(c_tnsr)
  coef_tnsrs = np.array(coef_tnsrs)

  GC_est = (np.max(np.abs(coef_tnsrs), axis=-1) > 0.).astype(int)

  experiment_params = {'seed': args.seed,
                       'lamratio': args.lamratio, 'model_lag': args.model_lag,
                       'lambdas': lambdas
                      }
  data_params = {'T': T, 'sparsity': sparsity, 'dt': dt, 
                }

  best_results = {'GC_est': GC_est, 'GC_true': GC}


  results_dict = {'experiment_params': experiment_params,
                  'data_params': data_params,
                  'best_results': best_results
                  }

  results_list.append(results_dict)

from metrics import *
for r in results_list:
  print(r['data_params'])
  GC_est = r['best_results']['GC_est']
  GC_true = r['best_results']['GC_true']
  GC_list = [GC_est[i, :, :] for i in range(GC_est.shape[0])]
  tp, fp, auc = compute_AUC(GC_true, GC_list, 0.1); print(auc)

# Save results
with open(experiment_name, 'wb') as f:
    pickle.dump(results_list, f)
