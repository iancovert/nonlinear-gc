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
from Data.generate_synthetic import var_model
from Data.data_processing import split_data, normalize

# Model modules
sys.path.append('../Model')
from lstm import ParallelLSTMEncoding
from experiment_cv import run_recurrent_experiment

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type = int, default = 1000, help = 'number of training epochs')
parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
parser.add_argument('--lam', type = float, default = 0.1, help = 'lambda for weight decay')
parser.add_argument('--seed', type = int, default = 12345, help = 'seed')
parser.add_argument('--hidden', type = int, default = 10, help = 'hidden units')

parser.add_argument('--sparsity', type = float, default = 0.2, help = 'sparsity of time series')
parser.add_argument('--p', type = int, default = 10, help = 'dimensionality of time series')
parser.add_argument('--T', type = int, default = 1000, help = 'length of time series')
parser.add_argument('--lag', type = int, default = 5, help = 'lag in long lag VAR model')

parser.add_argument('--window', type = int, default = 20, help = 'size of sliding windows for splitting training data')
parser.add_argument('--stride', type = int, default = 10, help = 'size of stride of sliding windows for splitting training data')
parser.add_argument('--truncation', type = int, default = None, help = 'length of gradient truncation')
parser.add_argument('--loss_check', type = int, default = 10, help = 'interval for checking loss')

args = parser.parse_args()

# Prepare filename
experiment_base = 'VAR LSTM Encoding'
results_dir = 'Results/' + experiment_base

experiment_name = results_dir + '/expt'
experiment_name += '_nepoch=%d_lr=%e_lam=%e_seed=%d_hidden=%d' % (args.nepoch, args.lr, args.lam, args.seed, args.hidden)
experiment_name += '_spars=%e_p=%d_T=%d_lag=%d.out' % (args.sparsity, args.p, args.T, args.lag)

# Create directory, if necessary
if not os.path.exists(results_dir):
     os.makedirs(results_dir)

# Verify that experiment doesn't exist
if os.path.isfile(experiment_name):
	print('Skipping experiment')
	sys.exit(0)

# Prepare data
X, _, GC = var_model(args.sparsity, args.p, 1, 1, args.T, args.lag)
X = normalize(X)
X_train, X_val = split_data(X, validation = 0.1)
Y_train = X_train[1:, :]
X_train = X_train[:-1, :]
Y_val = X_val[1:, :]
X_val = X_val[:-1, :]

p_in = Y_val.shape[1]
p_out = Y_val.shape[1]

# Get model
if args.seed != 0:
	torch.manual_seed(args.seed)
model = ParallelLSTMEncoding(p_in, p_out, args.hidden, 1, args.lr, 'prox', args.lam)

# Run experiment
train_loss, val_loss, best_properties = run_recurrent_experiment(model, X_train, Y_train, X_val, Y_val, 
	args.nepoch, window_size = args.window, stride_size = args.stride, truncation = args.truncation, predictions = True, loss_check = args.loss_check)

# Format results
experiment_params = {
	'nepoch': args.nepoch,
	'lam': args.lam,
	'lr': args.lr,
	'seed': args.seed,
	'hidden': args.hidden
}

data_params = {
	'sparsity': args.sparsity,
	'p': args.p,
	'T': args.T,
	'lag': args.lag,
	'GC_true': GC
}

best_results = {
	'best_nepoch': [props['nepoch'] for props in best_properties],
	'best_val_loss': [props['val_loss'] for props in best_properties],
	'predictions_train': np.concatenate([props['predictions_train'][:, np.newaxis] for props in best_properties], axis = 1),
	'predictions_val': np.concatenate([props['predictions_val'][:, np.newaxis] for props in best_properties], axis = 1),
	'GC_est': [np.linalg.norm(props['weights'], axis = 0) for props in best_properties]
}

results_dict = {
	'experiment_params': experiment_params,
	'data_params': data_params,
	'best_results': best_results
}

# Save results
with open(experiment_name, 'wb') as f:
	pickle.dump(results_dict, f)
