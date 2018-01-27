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
from Data.DREAM_preprocess import import_DREAM
from Data.data_processing import format_replicate_ts_data, normalize

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
parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')

parser.add_argument('--size', type = int, default = 50, help = 'size of Dream network')
parser.add_argument('--type', type = str, default = 'Ecoli', help = 'Dream network type')
parser.add_argument('--number', type = int, default = 1, help = 'Dream dataset number')

parser.add_argument('--loss_check', type = int, default = 10, help = 'interval for checking loss')

args = parser.parse_args()

# Prepare filename
experiment_base = 'DREAM MLP Encoding'
results_dir = 'Results/' + experiment_base

experiment_name = results_dir + '/expt'
experiment_name += '_nepoch=%d_lr=%e' % (args.nepoch, args.lr)
experiment_name += '_lam=%e_seed=%d_hidden=%d_networklag=%d' % (args.lam, args.seed, args.hidden, args.network_lag)
experiment_name += '_size=%d_type=%s_number=%d.out' % (args.size, args.type, args.number)

# Create directory, if necessary
if not os.path.exists(results_dir):
     os.makedirs(results_dir)

# Verify that experiment doesn't exist
if os.path.isfile(experiment_name):
	print('Skipping experiment')
	sys.exit(0)

# Prepare data
X, GC = import_DREAM(size = args.size, network_type = args.type, number = args.number)
X = normalize(X)
X_train, Y_train, _, _ = format_replicate_ts_data(X, lag = args.network_lag, validation = 0.0)

# Get model
if args.seed != 0:
	torch.manual_seed(args.seed)
model = ParallelMLPEncoding(Y_train.shape[1], Y_train.shape[1], args.network_lag, [args.hidden], args.lr, 'line', args.lam, 'group_lasso', nonlinearity = 'sigmoid')

# Run experiment
train_loss, train_objective, weights, pred = run_experiment(model, X_train, Y_train, args.nepoch, predictions = True, loss_check = args.loss_check)

# Format results
experiment_params = {
	'nepoch': args.nepoch,
	'lr': args.lr,
	'lam': args.lam,
	'seed': args.seed,
	'hidden': args.hidden,
	'network_lag': args.network_lag
}

data_params = {
	'size': args.size,
	'type': args.type,
	'number': args.number,
	'GC_true': GC
}

best_results = {
	'predictions_train': pred,
	'GC_est': [np.linalg.norm(np.reshape(w, newshape = (args.hidden * args.network_lag, args.size), order = 'F'), axis = 0) for w in weights]
}

results_dict = {
	'experiment_params': experiment_params,
	'data_params': data_params,
	'best_results': best_results
}

# Save results
with open(experiment_name, 'wb') as f:
	pickle.dump(results_dict, f)
