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
from Data.generate_synthetic import kuramoto_model
from Data.data_processing import split_data, normalize

# Model modules
sys.path.append('../Model')
from lstm import ParallelLSTMEncoding
from experiment_opt import run_recurrent_experiment

# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--lam', type = float, default = 0.1, help = 'lambda for weight decay')
parser.add_argument('--seed', type = int, default = 12345, help = 'seed')
parser.add_argument('--hidden', type = int, default = 10, help = 'hidden units')

parser.add_argument('--nepoch', type = int, default = 1000, help = 'number of training epochs')
parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
parser.add_argument('--cooldown', type = str, default = 'N', help = 'learning rate cooldown')

parser.add_argument('--sparsity', type = float, default = 0.3, help = 'sparsity of time series')
parser.add_argument('--p', type = int, default = 10, help = 'dimensionality of time series')
parser.add_argument('--T', type = int, default = 250, help = 'length of time series')
parser.add_argument('--trials', type = int, default = 10, help = 'number of replicates')

parser.add_argument('--window', type = int, default = 20, help = 'size of sliding windows for splitting training data')
parser.add_argument('--stride', type = int, default = 10, help = 'size of stride of sliding windows for splitting training data')
parser.add_argument('--truncation', type = int, default = None, help = 'length of gradient truncation')
parser.add_argument('--loss_check', type = int, default = 10, help = 'interval for checking loss')

args = parser.parse_args()

# Prepare filename
experiment_base = 'Oscillator LSTM Encoding'
results_dir = 'Results/' + experiment_base

experiment_name = results_dir + '/expt'
experiment_name += '_nepoch=%d_lr=%e_cooldown=%s' % (args.nepoch, args.lr, args.cooldown)
experiment_name += '_lam=%e_seed=%d_hidden=%d' % (args.lam, args.seed, args.hidden)
experiment_name += '_spars=%e_p=%d_T=%d_trials=%d.out' % (args.sparsity, args.p, args.T, args.trials)

# Create directory, if necessary
if not os.path.exists(results_dir):
     os.makedirs(results_dir)

# Verify that experiment doesn't exist
if os.path.isfile(experiment_name):
	print('Skipping experiment')
	sys.exit(0)

# Prepare data
X, GC = kuramoto_model(args.sparsity, args.p, num_trials = args.trials, N = args.T, standardized = True)
X = normalize(X)
Y_train = X[1:, :]
X_train = X[:-1, :]

# Get model
if args.seed != 0:
	torch.manual_seed(args.seed)
model = ParallelLSTMEncoding(Y_train.shape[2], Y_train.shape[2], args.hidden, 1, args.lr, 'prox', args.lam)

# Run experiment
train_loss, train_objective, best_properties = run_recurrent_experiment(model, X_train, Y_train, 
	args.nepoch, window_size = args.window, stride_size = args.stride, truncation = args.truncation, predictions = True, loss_check = args.loss_check, cooldown = args.cooldown.lower() == 'y')

# Format results
experiment_params = {
	'nepoch': args.nepoch,
	'lr': args.lr,
	'cooldown': args.cooldown,
	'lam': args.lam,
	'seed': args.seed,
	'hidden': args.hidden
}

data_params = {
	'sparsity': args.sparsity,
	'p': args.p,
	'T': args.T,
	'trials': args.trials,
	'GC_true': GC
}

best_results = {
	'best_nepoch': [props['nepoch'] for props in best_properties],
	'best_objective': [props['train_objective'] for props in best_properties],
	'predictions_train': np.concatenate([props['predictions_train'][:, np.newaxis] for props in best_properties], axis = 1),
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
