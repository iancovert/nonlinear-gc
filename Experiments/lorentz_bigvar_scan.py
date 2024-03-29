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
parser.add_argument('--nlambdas', type=int, default=50,
                    help='number of lambda values in grid')
parser.add_argument('--lamratio', type=float, default=1000.,
                    help='ratio of largest lambda in grid to smallest')
parser.add_argument('--seed', type=int, default=12345, help='seed')
parser.add_argument('--model_lag', type=int, default=5,
                    help='lag of BigVAR model')

parser.add_argument('--p', type=int, default=20,
                    help='dimensionality of time series')
parser.add_argument('--T', type=int, default=500,
                    help='length of time series')
args = parser.parse_args()

# Prepare filename
experiment_base = 'Lorentz BigVAR'
results_dir = 'Results/' + experiment_base

experiment_name = results_dir + '/scan_expt.out'

# Create directory, if necessary
if not os.path.exists(results_dir):
     os.makedirs(results_dir)

# Parameters to scan
sd_grid = [2.0]
FC_grid = [10, 40]
T_grid = [500, 750, 1000]

results_list = []

for sd, FC, T in product(sd_grid, FC_grid, T_grid):

  # generate and prepare data
  X, GC = lorentz_96_model_2(FC, args.p, T + 1, sd = sd, delta_t = 0.05)
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
  data_params = {'p': args.p, 'T': T,
                 'sd': sd, 'FC': FC
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

