import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from itertools import chain
import copy

from regularize import *

class ParallelLSTMEncoding:
	def __init__(self, input_series, output_series, hidden_size, hidden_layers, lr, opt, lam, lr_decay = 0.5, weight_decay = 0.01):
		# Set up networks
		self.lstms = [nn.LSTM(input_series, hidden_size, hidden_layers) for _ in range(output_series)]
		self.out_layers = [nn.Linear(hidden_size, 1) for _ in range(output_series)]

		# Save important arguments
		self.input_series = input_series
		self.output_series = output_series
		self.hidden_size = hidden_size
		self.hidden_layers = hidden_layers

		# Prepare to set up optimizer
		self.task = 'regression'
		self.loss_fn = nn.MSELoss()
		self.lam = lam
		self.lr = [lr] * output_series
		self.lr_decay = lr_decay
		self.weight_decay = weight_decay

		# Set up optimizer
		self.opt = opt

		if opt == 'prox':
			self.optimizers = [optim.SGD(list(lstm.parameters()) + list(out.parameters()), lr = lr, momentum = 0.0) for lstm, out in zip(self.lstms, self.out_layers)]
			self.train = self._train_prox

		elif opt == 'line':
			self.train = self._train_prox_line
			self.lstm_copy = copy.deepcopy(self.lstms[0])
			self.out_copy = copy.deepcopy(self.out_layers[0])

		else:
			if opt == 'adam':
				self.optimizers = [optim.Adam(list(lstm.parameters()) + list(out.parameters()), lr = lr) for lstm, out in zip(self.lstms, self.out_layers)]
			elif opt == 'sgd':
				self.optimizers = [optim.SGD(list(lstm.parameters()) + list(out.parameters()), lr = lr) for lstm, out in zip(self.lstms, self.out_layers)]
			elif opt == 'momentum':
				self.optimizers = [optim.SGD(list(lstm.parameters()) + list(out.parameters()), lr = lr, momentum = 0.9) for lstm, out in zip(self.lstms, self.out_layers)]
			else:
				raise ValueError('opt must be a valid option')

			self.train = self._train_builtin

	def cooldown(self, p):
		self.lr[p] *= self.lr_decay
		if self.opt == 'prox':
			self.optimizers[p] = optim.SGD(list(self.lstms[p].parameters()) + list(self.out_layers[p].parameters()), lr = self.lr[p], momentum = 0.0)
		elif self.opt == 'momentum':
			self.optimizers[p] = optim.SGD(list(self.lstms[p].parameters()) + list(self.out_layers[p].parameters()), lr = self.lr[p], momentum = 0.9)
		elif self.opt == 'adam':
			self.optimizers[p] = optim.Adam(list(self.lstms[p].parameters()) + list(self.out_layers[p].parameters()), lr = self.lr[p])
		else:
			self.optimizers[p] = optim.SGD(list(self.lstms[p].parameters()) + list(self.out_layers[p].parameters()), lr = self.lr[p])

	def _init_hidden(self, replicates = 1):
		return tuple(
			(Variable(torch.zeros(self.hidden_layers, replicates, self.hidden_size)), 
				Variable(torch.zeros(self.hidden_layers, replicates, self.hidden_size)))
			for _ in self.lstms
		)

	def _forward(self, X, hidden = None, return_hidden = False):
		if hidden is None:
			if len(X.shape) == 2:
				hidden = self._init_hidden()
			else:
				hidden = self._init_hidden(X.shape[1])
		
		X_var = Variable(torch.from_numpy(X).float())

		if len(X.shape) == 2:
			n, p = X.shape
			X_var = X_var.view(n, 1, p)

		lstm_return = [self.lstms[target](X_var, hidden[target]) for target in range(self.output_series)]

		lstm_out, lstm_hidden = list(zip(*lstm_return))

		# .view(-1, self.hidden_size) works for 2-, 3-dimensional X
		net_out = [self.out_layers[target](lstm_out[target].view(-1, self.hidden_size)) for target in range(self.output_series)]
		
		if return_hidden:
			return net_out, lstm_hidden
		else:
			return net_out

	def _loss(self, X, Y, hidden = None, return_hidden = False):
		Y_pred, hidden = self._forward(X, hidden = hidden, return_hidden = True)
		
		if len(Y.shape) == 2:
			Y_var = [Variable(torch.from_numpy(Y[:, target][:, np.newaxis]).float()) for target in range(self.output_series)]

		else:
			Y_var = [Variable(torch.from_numpy(Y[:, :, target]).float()).view(-1, 1) for target in range(self.output_series)]

		loss = [self.loss_fn(Y_pred[target], Y_var[target]) for target in range(self.output_series)]
		
		if return_hidden:
			return loss, hidden
		else:
			return loss

	def _ridge(self, p = None):
		if p is None:
			return [self._ridge(p = target) for target in range(self.output_series)]
		else:
			params = list(self.out_layers[p].parameters())
			return self.weight_decay * sum([torch.sum(params[i]**2) for i in range(0, len(params), 2)])

	def _objective(self, X, Y, hidden = None, return_hidden = False):
		loss, hidden = self._loss(X, Y, hidden = hidden, return_hidden = True)
		ridge = self._ridge()
		penalty = [self.lam * apply_penalty(lstm.weight_ih_l0, 'group_lasso', self.input_series) for lstm in self.lstms]

		if return_hidden:
			return [l + r + p for (l, r, p) in zip(loss, ridge, penalty)], hidden
		else:
			return [l + r + p for (l, r, p) in zip(loss, ridge, penalty)]

	def _train_prox(self, X, Y, truncation = None):
		if truncation is None:
			self._train_prox_helper(X, Y)

		else:
			self._truncated_training(X, Y, truncation, self._train_prox_helper)

	def _train_builtin(self, X, Y, truncation = None):
		if truncation is None:
			self._train_builtin_helper(X, Y)

		else:
			self._truncated_training(X, Y, truncation, self._train_builtin_helper)

	def _train_prox_line(self, X, Y):
		# Compute loss and objective
		loss = self._loss(X, Y, hidden = None, return_hidden = False)
		ridge = self._ridge()
		total_loss = sum(loss) + sum(ridge)
		penalty = [self.lam * apply_penalty(lstm.weight_ih_l0, 'group_lasso', self.input_series) for lstm in self.lstms]

		# Compute gradient from loss
		[net.zero_grad() for net in chain(self.lstms, self.out_layers)]

		total_loss.backward()

		# Parameters for line search
		t = 0.9
		s = 0.8
		min_lr = 1e-18

		# Return value, to indicate whether improvements have been made
		return_value = False

		# Torch Variables for line search
		X_var = Variable(torch.from_numpy(X).float())
		if len(X.shape) == 2:
			n, p = X.shape
			X_var = X_var.view(n, 1, p)

		if len(Y.shape) == 2:
			Y_var = [Variable(torch.from_numpy(Y[:, target][:, np.newaxis]).float()) for target in range(self.output_series)]
		else:
			Y_var = [Variable(torch.from_numpy(Y[:, :, target]).float()).view(-1, 1) for target in range(self.output_series)]

		# Copy of single network, used for testing new param values
		new_lstm = self.lstm_copy
		new_out = self.out_copy
		new_lstm_params = list(new_lstm.parameters())
		new_out_params = list(new_out.parameters())

		for target, (lstm, out) in enumerate(list(zip(self.lstms, self.out_layers))):
			# Set up initial parameters values
			lr = self.lr[target]
			original_objective = loss[target] + ridge[target] + penalty[target]
			original_lstm_params = list(lstm.parameters())
			original_out_params = list(out.parameters())

			while lr > min_lr:
				# Take gradient step in new params
				for params, o_params in chain(zip(new_lstm_params, original_lstm_params), zip(new_out_params, original_out_params)):
					params.data = o_params.data - o_params.grad.data * lr

				# Apply proximal operator to new params
				prox_operator(new_lstm.weight_ih_l0, 'group_lasso', self.input_series, lr, self.lam)

				# Compute objective using new params
				if len(X.shape) == 2:
					hidden = (Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)), Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)))
				else:
					hidden = (Variable(torch.zeros(self.hidden_layers, X.shape[1], self.hidden_size)), Variable(torch.zeros(self.hidden_layers, X.shape[1], self.hidden_size)))
				
				Y_pred = new_out(new_lstm(X_var, hidden)[0].view(-1, self.hidden_size))
				new_objective = self.loss_fn(Y_pred, Y_var[target]) 
				new_objective += self.weight_decay * sum([torch.sum(new_out_params[i]**2) for i in range(0, len(new_out_params), 2)])
				new_objective += self.lam * apply_penalty(new_lstm.weight_ih_l0, 'group_lasso', self.input_series)

				diff_squared = sum([torch.sum((o_params.data - params.data)**2) for params, o_params in chain(zip(new_lstm_params, original_lstm_params), zip(new_out_params, original_out_params))])

				if (new_objective < original_objective - t * lr * diff_squared).data[0]:
					# Replace parameter values
					for params, o_params in chain(zip(new_lstm_params, original_lstm_params), zip(new_out_params, original_out_params)):
						o_params.data = params.data
					
					return_value = True
					break

				else:
					# Try a lower learning rate
					lr *= s

			# Update initial learning rate for next iteration
			self.lr[target] = np.sqrt(self.lr[target] * lr)

		return return_value

	def _truncated_training(self, X, Y, truncation, training_func):
		T = X.shape[0]
		num = int(np.ceil(T / truncation))
		hidden = self._init_hidden()
		for i in range(num - 1):
			X_subset = X[range(i * truncation, (i + 1) * truncation), :]
			Y_subset = Y[range(i * truncation, (i + 1) * truncation), :]
			hidden = training_func(X_subset, Y_subset, hidden = hidden, return_hidden = True)

			# Repackage hidden
			hidden = repackage_hidden(hidden)

		X_subset = X[((num - 1) * truncation):, :]
		Y_subset = Y[((num - 1) * truncation):, :]
		training_func(X_subset, Y_subset, hidden = hidden)

	def _train_prox_helper(self, X, Y, hidden = None, return_hidden = False):
		loss, hidden = self._loss(X, Y, hidden = hidden, return_hidden = True)
		ridge = self._ridge()
		total_loss = sum(loss) + sum(ridge)

		# Take gradient step
		[net.zero_grad() for net in chain(self.lstms, self.out_layers)]

		total_loss.backward()
		[optimizer.step() for optimizer in self.optimizers]

		# Apply proximal operator
		[prox_operator(lstm.weight_ih_l0, 'group_lasso', self.input_series, lr, self.lam) for (lstm, lr) in zip(self.lstms, self.lr)]

		if return_hidden:
			return hidden

	def _train_builtin_helper(self, X, Y, hidden = None, return_hidden = False):
		objective, hidden = self._objective(X, Y, hidden = hidden, return_hidden = True)

		# Run optimizer
		[net.zero_grad() for net in chain(self.lstms, self.out_layers)]

		sum(objective).backward()
		[optimizer.step() for optimizer in self.optimizers]

		if return_hidden:
			return hidden

	def calculate_loss(self, X, Y):
		loss = self._loss(X, Y)
		return [num.data[0] for num in loss]

	def calculate_objective(self, X, Y):
		objective = self._objective(X, Y)
		return np.array([num.data[0] for num in objective])

	def predict(self, X):
		Y_pred = self._forward(X)
		return np.array([Y[:, 0].data.numpy() for Y in Y_pred]).T

	def get_weights(self, p = None):
		if p is None:
			return [lstm.weight_ih_l0.data.numpy().copy() for lstm in self.lstms]
		else:
			return self.lstms[p].weight_ih_l0.data.numpy().copy()

class ParallelLSTMDecoding:
	def __init__(self, input_series, output_series, hidden_size, hidden_layers, fc_units, lr, opt, lam, nonlinearity = 'relu'):
		# Set up networks
		self.lstms = [nn.LSTM(1, hidden_size, hidden_layers) for _ in range(input_series)]
		
		self.out_networks = []
		for target in range(output_series):
			net = nn.Sequential()
			for i, (d_in, d_out) in enumerate(list(zip([hidden_size * input_series] + fc_units[:-1], fc_units))):
				net.add_module('fc%d' % i, nn.Linear(d_in, d_out, bias = True))
				if nonlinearity == 'relu':
					net.add_module('relu%d' % i, nn.ReLU())
				elif nonlinearity == 'sigmoid':
					net.add_module('sigmoid%d' % i, nn.Sigmoid())
				else:
					raise ValueError('nonlinearity must be "relu" or "sigmoid"')
			net.add_module('out', nn.Linear(fc_units[-1], 1, bias = True))
			self.out_networks.append(net)

		# Save important arguments
		self.input_series = input_series
		self.output_series = output_series
		self.hidden_size = hidden_size
		self.hidden_layers = hidden_layers

		# Set up optimizer
		self.task = 'regression'
		self.loss_fn = nn.MSELoss()
		self.lam = lam
		self.lr = lr

		param_list = []
		for net in chain(self.lstms, self.out_networks):
			param_list = param_list + list(net.parameters())

		if opt == 'prox':
			self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.0)
			self.train = self._train_prox
		else:
			if opt == 'adam':
				self.optimizer = optim.Adam(param_list, lr = lr)
			elif opt == 'sgd':
				self.optimizer = optim.SGD(param_list, lr = lr)
			elif opt == 'momentum':
				self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)
			else:
				raise ValueError('opt must be a valid option')

			self.train = self._train_builtin

	def _init_hidden(self):
		return tuple(
			(Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)), 
				Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)))
			for _ in self.lstms
		)

	def _forward(self, X, hidden = None, return_hidden = False):
		n, p = X.shape
		X_var = [Variable(torch.from_numpy(X[:, in_series][:, np.newaxis]).float()) for in_series in range(self.input_series)]
		if hidden is None:
			hidden = self._init_hidden()
		lstm_return = [self.lstms[in_series](X_var[in_series].view(n, 1, 1), hidden[in_series]) for in_series in range(self.input_series)]
		lstm_out, lstm_hidden = list(zip(*lstm_return))
		lstm_layer = torch.cat([out.view(n, self.hidden_size) for out in lstm_out], dim = 1)
		net_out = [net(lstm_layer) for net in self.out_networks]

		if return_hidden:
			return net_out, lstm_hidden
		else:
			return net_out

	def _loss(self, X, Y, hidden = None, return_hidden = False):
		Y_pred, hidden = self._forward(X, hidden = hidden, return_hidden = True)
		Y_var = [Variable(torch.from_numpy(Y[:, target]).float()) for target in range(self.output_series)]
		loss = [self.loss_fn(Y_pred[target], Y_var[target]) for target in range(self.output_series)]

		if return_hidden:
			return loss, hidden
		else:
			return loss

	def _objective(self, X, Y, hidden = None, return_hidden = False):
		loss, hidden = self._loss(X, Y, hidden = hidden, return_hidden = True)
		penalty = [self.lam * apply_penalty(list(net.parameters())[0], 'group_lasso', self.input_series, lag = self.hidden_size) for net in self.out_networks]

		if return_hidden:
			return [l + p for (l, p) in zip(loss, penalty)], hidden
		else:
			return [l + p for (l, p) in zip(loss, penalty)]

	def _train_prox(self, X, Y, truncation = None):
		if truncation is None:
			self._train_prox_helper(X, Y)

		else:
			self._truncated_training(X, Y, truncation, self._train_prox_helper)

	def _train_builtin(self, X, Y, truncation = None):
		if truncation is None:
			self._train_builtin_helper(X, Y)

		else:
			self._truncated_training(X, Y, truncation, self._train_builtin_helper)

	def _truncated_training(self, X, Y, truncation, training_func):
		T = X.shape[0]
		num = int(np.ceil(T / truncation))
		hidden = self._init_hidden()
		for i in range(num - 1):
			X_subset = X[range(i * truncation, (i + 1) * truncation), :]
			Y_subset = Y[range(i * truncation, (i + 1) * truncation), :]
			hidden = training_func(X_subset, Y_subset, hidden = hidden, return_hidden = True)

			# Repackage hidden
			hidden = repackage_hidden(hidden)

		X_subset = X[((num - 1) * truncation):, :]
		Y_subset = Y[((num - 1) * truncation):, :]
		training_func(X_subset, Y_subset, hidden = hidden)

	def _train_prox_helper(self, X, Y, hidden = None, return_hidden = False):
		loss, hidden = self._loss(X, Y, hidden = hidden, return_hidden = True)
		total_loss = sum(loss)

		# Take gradient step
		[net.zero_grad() for net in chain(self.lstms, self.out_networks)]

		total_loss.backward()
		self.optimizer.step()

		# Apply proximal operator
		[prox_operator(list(net.parameters())[0], 'group_lasso', self.input_series, self.lr, self.lam, lag = self.hidden_size) for net in self.out_networks]

		if return_hidden:
			return hidden

	def _train_builtin_helper(self, X, Y, hidden = None, return_hidden = False):
		objective, hidden = self._objective(X, Y, hidden = hidden, return_hidden = True)

		# Run optimizer
		[net.zero_grad() for net in chain(self.lstms, self.out_networks)]

		sum(objective).backward()
		self.optimizer.step()

		if return_hidden:
			return hidden

	def predict(self, X):
		Y_pred = self._forward(X)
		return np.array([Y[:, 0].data.numpy() for Y in Y_pred]).T

	def calculate_loss(self, X, Y):
		loss = self._loss(X, Y)
		return [num.data[0] for num in loss]

	def calculate_objective(self, X, Y):
		objective = self._objective(X, Y)
		return np.array([num.data[0] for num in objective])

	def get_weights(self, p = None):
		if p is None:
			return [list(net.parameters())[0].data.numpy().copy() for net in self.out_networks]
		else:
			return list(self.out_networks[p].parameters())[0].data.numpy().copy()


def repackage_hidden(hidden):
	if type(hidden) == Variable:
		return Variable(hidden.data)
	else:
		return tuple(repackage_hidden(h) for h in hidden)
