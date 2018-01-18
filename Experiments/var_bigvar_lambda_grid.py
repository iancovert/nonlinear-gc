from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import pickle
import sys

# Data modules
from Data.generate_synthetic import var_model
from Data.data_processing import format_ts_data, normalize

# Model modules
sys.path.append('../Model')
from bigvar import run_bigvar

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nlambdas', type=int, default=10,
                    help='number of lambda values in grid')
parser.add_argument('--lamratio', type=float, default=10.,
                    help='ratio of largest lambda in grid to smallest')
parser.add_argument('--seed', type=int, default=12345, help='seed')
parser.add_argument('--model_lag', type=int, default=5,
                    help='lag of BigVAR model')

parser.add_argument('--sparsity', type=float, default=0.2,
                    help='sparsity of time series')
parser.add_argument('--p', type=int, default=10,
                    help='dimensionality of time series')
parser.add_argument('--T', type=int, default=1000,
                    help='length of time series')
parser.add_argument('--lag', type=int, default=2,
                    help='lag in simulated VAR model')
args = parser.parse_args()

# Prepare filename
experiment_base = 'VAR BigVAR'
results_dir = 'Results/' + experiment_base

experiment_name = results_dir + '/expt'
experiment_name += '_nlambdas=%d_lamratio=%e_seed=%d_model-lag=%d' % (args.nlambdas, args.lamratio, args.seed, args.model_lag)
experiment_name += '_spars=%e_p=%d_T=%d_lag=%d.out' % (args.sparsity, args.p, args.T, args.lag) 

# Create directory, if necessary
if not os.path.exists(results_dir):
     os.makedirs(results_dir)

# Verify that experiment doesn't exist
if os.path.isfile(experiment_name):
	print('Skipping experiment')
	sys.exit(0)

# generate and prepare data
X, _, GC = var_model(args.sparsity, args.p, 1, 1, args.T+1, args.lag)
X = normalize(X)

coefs, lambdas, _ = run_bigvar(X, args.model_lag, 'HVARELEM',
                               nlambdas=args.nlambdas,
                               lamratio=float(args.lamratio),
                               T1=0, T2=args.T-1, use_intercept=False)

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
data_params = {'sparsity': args.sparsity, 'p': args.p, 'T': args.T,
               'lag': args.lag, 'GC_true': GC
              }
best_results = {'GC_est': GC_est}


results_dict = {'experiment_params': experiment_params,
                'data_params': data_params,
                'best_results': best_results
                }

# Save results
with open(experiment_name, 'wb') as f:
	pickle.dump(results_dict, f)
