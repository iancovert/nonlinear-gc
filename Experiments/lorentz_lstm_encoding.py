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
from Data.generate_synthetic import lorentz_96_model
from Data.data_processing import split_data

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
args = parser.parse_args()

nepoch = args.nepoch
lr = args.lr
opt_type = 'prox'
seed = args.seed
lam = args.lam

window_size = 100
stride_size = None
truncation = 5

arch = 2

# Prepare filename
experiment_base = 'Lorentz LSTM Encoding'
results_dir = 'Results/' + experiment_base

experiment_name = results_dir + '/expt'
experiment_name += '_lam=%e_nepoch=%d_lr=%e_seed-%d.out' % (lam, nepoch, lr, seed)

# Create directory, if necessary
if not os.path.exists(results_dir):
     os.makedirs(results_dir)

# Verify that experiment doesn't exist
if os.path.isfile(experiment_name):
	print('Skipping experiment')
	sys.exit(0)

# Prepare data
X, GC = lorentz_96_model(8, 10, 1000)
X_train, X_val = split_data(X, validation = 0.1)
Y_train = X_train[1:, :]
X_train = X_train[:-1, :]
Y_val = X_val[1:, :]
X_val = X_val[:-1, :]

p_in = X_val.shape[1]
p_out = Y_val.shape[1]

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

# Get model
torch.manual_seed(seed)
model = ParallelLSTMEncoding(p_in, p_out, hidden_size, hidden_layers, lr, opt_type, lam)

# Run experiment
train_loss, val_loss, weights_list, forecasts_train, forecasts_val = run_recurrent_experiment(model, X_train, Y_train, X_val, Y_val, 
	nepoch, window_size = window_size, stride_size = stride_size, truncation = truncation, predictions = True, loss_check = 10)

# Create GC estimate grid
GC_est = np.zeros((p_out, p_in))
for target in range(p_out):
	W = weights_list[target]
	GC_est[target, :] = np.linalg.norm(W, axis = 0, ord = 2)

# Save results
results_dict = {
	'lam': lam,
	'train_loss': train_loss,
	'val_loss': val_loss,
	'weights_list': weights_list,
	'GC_est': GC_est,
	'forecasts_train': forecasts_train,
	'forecasts_val': forecasts_val,
	'GC_true': GC
}

# Save results
with open(experiment_name, 'wb') as f:
	pickle.dump(results_dict, f)
