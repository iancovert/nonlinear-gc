import torch
import numpy as np
import pandas as pd
import argparse
import os
from itertools import product
import pickle
import shutil
import sys

# Data modules
from Data.generate_synthetic import standardized_var_model
from Data.data_processing import format_ts_data, normalize

# Model modules
sys.path.append('../Model')
from mlp import ParallelMLPEncoding
from experiment_line import run_experiment

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--lam', type = float, default = 0.1, help = 'lambda for weight decay')
parser.add_argument('--seed', type = int, default = 12345, help = 'seed')
parser.add_argument('--hidden', type = int, default = 10, help = 'hidden units')
parser.add_argument('--network_lag', type = int, default = 2, help = 'lag considered by MLP')

parser.add_argument('--nepoch', type = int, default = 1000, help = 'number of training epochs')
parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate')

parser.add_argument('--sparsity', type = float, default = 0.3, help = 'sparsity of time series')
parser.add_argument('--p', type = int, default = 10, help = 'dimensionality of time series')
parser.add_argument('--T', type = int, default = 500, help = 'length of time series')
parser.add_argument('--lag', type = int, default = 1, help = 'lag in VAR model')

parser.add_argument('--loss_check', type = int, default = 10, help = 'interval for checking loss')

args = parser.parse_args()

# Set a couple arguments
args.nepoch = 20000
args.loss_check = 100
args.network_lag = 5
args.lag = 3
results_list = []

# Prepare data
X, _, GC = standardized_var_model(args.sparsity, args.p, 5, 0.1, args.T, args.lag)
X = normalize(X)
X_train, Y_train, _, _ = format_ts_data(X, args.network_lag, validation = 0.0)

lam_grid = np.geomspace(0.2, 0.01, 10)

for lam in lam_grid:
	lam = float(lam)

	# Get model
	if args.seed != 0:
		torch.manual_seed(args.seed)
	model = ParallelMLPEncoding(Y_train.shape[1], Y_train.shape[1], args.network_lag, [args.hidden], args.lr, 'line', lam, 'hierarchical', nonlinearity = 'sigmoid', weight_decay = 0.01)

	# Run experiment
	train_loss, train_objective, weights, pred = run_experiment(model, X_train, Y_train, args.nepoch, predictions = True, loss_check = args.loss_check, verbose = True)

	results_dict = {
		'GC_est': [np.linalg.norm(np.reshape(w, newshape = (args.hidden * args.network_lag, args.p), order = 'F'), axis = 0) for w in weights],
		'loss': train_loss,
		'objective': train_objective,
		'weights': weights,
		'GC_true': GC
	}

	results_list.append(results_dict)
	print('Done with lam = %f' % lam)

with open('lag_selection_results.out', 'wb') as f:
	pickle.dump(results_list, f)