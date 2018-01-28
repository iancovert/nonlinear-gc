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
from Data.data_processing import split_data, normalize

# Model modules
sys.path.append('../Model')
from lstm import ParallelLSTMEncoding
from experiment_line import run_recurrent_experiment

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--lam', type = float, default = 0.1, help = 'lambda for weight decay')
parser.add_argument('--seed', type = int, default = 12345, help = 'seed')
parser.add_argument('--hidden', type = int, default = 10, help = 'hidden units')

parser.add_argument('--nepoch', type = int, default = 1000, help = 'number of training epochs')
parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate')

parser.add_argument('--sparsity', type = float, default = 0.2, help = 'sparsity of time series')
parser.add_argument('--p', type = int, default = 10, help = 'dimensionality of time series')
parser.add_argument('--T', type = int, default = 1000, help = 'length of time series')
parser.add_argument('--lag', type = int, default = 1, help = 'lag in VAR model')

parser.add_argument('--loss_check', type = int, default = 10, help = 'interval for checking loss')

args = parser.parse_args()

# Prepare filename
experiment_base = 'Standardized VAR LSTM Encoding'
results_dir = 'Results/' + experiment_base

experiment_name = results_dir + '/expt'
experiment_name += '_nepoch=%d_lr=%e_cooldown=%s' % (args.nepoch, args.lr, args.cooldown)
experiment_name += '_lam=%e_seed=%d_hidden=%d' % (args.lam, args.seed, args.hidden)
experiment_name += '_spars=%e_p=%d_T=%d_lag=%d.out' % (args.sparsity, args.p, args.T, args.lag)

# Create directory, if necessary
if not os.path.exists(results_dir):
     os.makedirs(results_dir)

# Verify that experiment doesn't exist
if os.path.isfile(experiment_name):
	print('Skipping experiment')
	sys.exit(0)

# Prepare data
X, _, GC = standardized_var_model(args.sparsity, args.p, 5, 1.0, args.T, args.lag)
X = normalize(X)
Y_train = X[1:, :]
X_train = X[:-1, :]

# Get model
if args.seed != 0:
	torch.manual_seed(args.seed)
model = ParallelLSTMEncoding(Y_train.shape[1], Y_train.shape[1], args.hidden, 1, args.lr, 'line', args.lam)

# Run experiment
train_loss, train_objective, weights, pred = run_recurrent_experiment(model, X_train, Y_train, 
	args.nepoch, predictions = True, loss_check = args.loss_check)

# Format results
experiment_params = {
	'nepoch': args.nepoch,
	'lr': args.lr,
	'lam': args.lam,
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
	'predictions_train': pred,
	'GC_est': [np.linalg.norm(w, axis = 0) for w in weights]
}

results_dict = {
	'experiment_params': experiment_params,
	'data_params': data_params,
	'best_results': best_results
}

# Save results
with open(experiment_name, 'wb') as f:
	pickle.dump(results_dict, f)
