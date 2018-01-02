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
from Data.generate_synthetic import hmm_model
from Data.data_processing import split_data, normalize

# Model modules
sys.path.append('../Model')
from lstm import ParallelLSTMEncoding
from experiment import run_recurrent_experiment

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type = int, default = 1000, help = 'number of training epochs')
parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
parser.add_argument('--lam', type = float, default = 0.1, help = 'lambda for weight decay')
parser.add_argument('--seed', type = int, default = 12345, help = 'seed')
parser.add_argument('--arch', type = int, default = 1, help = 'architecture')
args = parser.parse_args()

nepoch = args.nepoch
lr = args.lr
opt_type = 'prox'
seed = args.seed
lam = args.lam
arch = args.arch

window_size = 100
stride_size = 10
truncation = None

# Prepare data
X, _, GC = hmm_model(10, 1000, num_states = 3, sd_e = 0.1, sparsity = 0.2, tau = 2)
X = normalize(X)
X_train, X_val = split_data(X, validation = 0.1)
Y_train = X_train[1:, :]
X_train = X_train[:-1, :]
Y_val = X_val[1:, :]
X_val = X_val[:-1, :]

p_in = X_val.shape[1]
p_out = Y_val.shape[1]

# Determine architecture
if arch == 1:
	hidden_size = 5
	hidden_layers = 1
elif arch == 2:
	hidden_size = 10
	hidden_layers = 1
elif arch == 3:
	hidden_size = 20
	hidden_layers = 1
elif arch == 4:
	hidden_size = 40
	hidden_layers = 1
else:
	raise ValueError('arch must be in {1, 2, 3, 4}')

# Begin to create filename, create directory if necessary
experiment_base = 'Medium VAR LSTM Encoding Checkin'
results_dir = 'Results/' + experiment_base
if not os.path.exists(results_dir):
     os.makedirs(results_dir)

# Get model
if seed != 0:
	torch.manual_seed(seed)
model = ParallelLSTMEncoding(p_in, p_out, hidden_size, hidden_layers, lr, opt_type, lam)

# Set up empty variables for previous training, validation loss
old_train_loss = None
old_val_loss = None

# Determine properties of checkin periods
checkin_size = 100
checkin_num = int(nepoch / checkin_size)

for checkin_ind in range(1, checkin_num + 1):

	# Number of epochs
	epochs = checkin_ind * checkin_size

	# Prepare filename
	experiment_name = results_dir + '/expt'
	experiment_name += '_lam=%e_nepoch=%d_lr=%e_seed-%d_arch-%d.out' % (lam, epochs, lr, seed, arch)

	# Verify that experiment doesn't exist
	if os.path.isfile(experiment_name):
		print('Skipping experiment')
		sys.exit(0)

	# Run experiment
	train_loss, val_loss, weights_list, forecasts_train, forecasts_val = run_recurrent_experiment(model, X_train, Y_train, X_val, Y_val, 
		checkin_size, window_size = window_size, stride_size = stride_size, truncation = truncation, predictions = True, loss_check = 10)

	if old_train_loss is not None:
		train_loss = np.concatenate((old_train_loss, train_loss), axis = 0)
		val_loss = np.concatenate((old_val_loss, val_loss), axis = 0)
	old_train_loss = train_loss
	old_val_loss = val_loss

	# Create GC estimate grid
	GC_est = np.zeros((p_out, p_in))
	for target in range(p_out):
		W = weights_list[target]
		GC_est[target, :] = np.linalg.norm(W, axis = 0, ord = 2)

	# Save results
	results_dict = {
		'train_loss': train_loss,
		'val_loss': val_loss,
		'weights_list': weights_list,
		'GC_est': GC_est,
		'forecasts_train': forecasts_train,
		'forecasts_val': forecasts_val,
		'GC_true': GC,
		'lam': lam,
		'nepoch': epochs,
		'lr': lr,
		'opt_type': opt_type,
		'seed': seed,
		'arch': hidden_size
	}

	# Save results
	with open(experiment_name, 'wb') as f:
		pickle.dump(results_dict, f)
