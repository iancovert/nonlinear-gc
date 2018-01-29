import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import copy
import numpy as np
from itertools import chain

from regularize import *

class ParallelMLPEncoding:
	def __init__(self, input_series, output_series, lag, hidden_units, lr, opt, lam, penalty, nonlinearity = 'relu', lr_decay = 0.5):
		# Set up networks for each output series
		self.sequentials = []
		self.p = output_series
		self.lag = lag
		self.n = input_series
		layers = list(zip([input_series * lag] + hidden_units[:-1], hidden_units))

		for target in range(output_series):
			net = torch.nn.Sequential()
			for i, (d_in, d_out) in enumerate(layers):
				net.add_module('fc%d' % i, nn.Linear(d_in, d_out, bias = True))
				if nonlinearity == 'relu':
					net.add_module('relu%d' % i, nn.ReLU())
				elif nonlinearity == 'sigmoid':
					net.add_module('sigmoid%d' % i, nn.Sigmoid())
				else:
					raise ValueError('nonlinearity must be "relu" or "sigmoid"')
			net.add_module('out', nn.Linear(hidden_units[-1], 1, bias = True))

			self.sequentials.append(net)

		# Prepare to set up optimizer
		self.task = 'regression'
		self.loss_fn = nn.MSELoss()
		self.lr = [lr] * output_series
		self.lr_decay = lr_decay
		self.lam = lam
		self.penalty = penalty

		# Set up optimizer
		self.opt = opt

		if opt == 'prox':
			self.optimizers = [optim.SGD(list(net.parameters()), lr = lr, momentum = 0.0) for net in self.sequentials]
			self.train = self._train_prox

		elif opt == 'line':
			self.train = self._train_prox_line

		else:
			if opt == 'adam':
				self.optimizers = [optim.Adam(list(net.parameters()), lr = lr) for net in self.sequentials]
			elif opt == 'sgd':
				self.optimizers = [optim.SGD(list(net.parameters()), lr = lr) for net in self.sequentials]
			elif opt == 'momentum':
				self.optimizers = [optim.SGD(list(net.parameters()), lr = lr, momentum = 0.9) for net in self.sequentials]
			else:
				raise ValueError('opt must be a valid option')

			self.train = self._train_builtin

	def cooldown(self, p):
		self.lr[p] *= self.lr_decay
		if self.opt == 'prox':
			self.optimizers[p] = optim.SGD(list(self.sequentials[p].parameters()), lr = self.lr[p], momentum = 0.0)
		elif self.opt == 'momentum':
			self.optimizers[p] = optim.SGD(list(self.sequentials[p].parameters()), lr = self.lr[p], momentum = 0.9)
		elif self.opt == 'adam':
			self.optimizers[p] = optim.Adam(list(self.sequentials[p].parameters()), lr = self.lr[p])
		else:
			self.optimizers[p] = optim.SGD(list(self.sequentials[p].parameters()), lr = self.lr[p])

	def _train_prox_line(self, X, Y):
		# Compute loss and objective
		loss = self._loss(X, Y)
		penalty = [self.lam * apply_penalty(list(net.parameters())[0], self.penalty, self.n, lag = self.lag) for net in self.sequentials]
		total_loss = sum(loss)

		# Compute gradient from loss
		[net.zero_grad() for net in self.sequentials]

		total_loss.backward()

		# Parameters for line search
		t = 0.9
		s = 0.8
		min_lr = 1e-7

		# Return value, to indicate whether improvements have been made
		return_value = False

		# Torch Variables for line search
		X_var = Variable(torch.from_numpy(X).float())
		Y_var = [Variable(torch.from_numpy(Y[:, target][:, np.newaxis]).float()) for target in range(self.p)]

		for target, net in enumerate(self.sequentials):
			# Set up initial parameter values
			lr = self.lr[target]
			original_objective = loss[target] + penalty[target]
			new_net = copy.deepcopy(net)

			while lr > min_lr:
				# Take gradient step in new params
				for params, o_params in zip(new_net.parameters(), net.parameters()):
					params.data = o_params.data - o_params.grad.data * lr
				
				# Apply proximal operator to new params
				prox_operator(list(new_net.parameters())[0], self.penalty, self.n, lr, self.lam, lag = self.lag)
				
				# Compute objective using new params
				Y_pred = new_net(X_var)
				new_objective = self.loss_fn(Y_pred, Y_var[target]) + self.lam * apply_penalty(list(new_net.parameters())[0], self.penalty, self.n, lag = self.lag)
				
				diff_squared = sum([torch.sum((o_params.data - params.data)**2) for (params, o_params) in zip(new_net.parameters(), net.parameters())])
				# diff_squared = 0.0
				# for params, o_params in zip(new_net.parameters(), net.parameters()):
				# 	diff_squared += torch.sum((o_params.data - params.data)**2)
				if (new_objective < original_objective - t * lr * diff_squared).data[0]:
					# Replace parameter values
					for params, o_params in zip(new_net.parameters(), net.parameters()):
						o_params.data = params.data
					
					return_value = True
					break

				else:
					# Try a lower learning rate
					lr *= s

			# Update initial learning rate for next training iteration
			self.lr[target] = np.sqrt(self.lr[target] * lr)

		return return_value

	def _train_prox(self, X, Y):
		# Compute total loss
		loss = self._loss(X, Y)
		total_loss = sum(loss)

		# Run optimizer
		[net.zero_grad() for net in self.sequentials]

		total_loss.backward()
		[optimizer.step() for optimizer in self.optimizers]

		# Apply prox operator
		[prox_operator(list(net.parameters())[0], self.penalty, self.n, lr, self.lam, lag = self.lag) for (net, lr) in zip(self.sequentials, self.lr)]

	def _train_builtin(self, X, Y):
		# Compute objective
		objective = self._objective(X, Y)

		# Run optimizer
		[net.zero_grad() for net in self.sequentials]

		sum(objective).backward()
		[optimizer.step() for optimizer in self.optimizers]

	def _forward(self, X):
		X_var = Variable(torch.from_numpy(X).float())
		return [net(X_var) for net in self.sequentials]

	def _loss(self, X, Y):
		Y_pred = self._forward(X)
		Y_var = [Variable(torch.from_numpy(Y[:, target][:, np.newaxis]).float()) for target in range(self.p)]
		return [self.loss_fn(Y_pred[target], Y_var[target]) for target in range(self.p)]

	def _objective(self, X, Y):
		loss = self._loss(X, Y)
		penalty = [self.lam * apply_penalty(list(net.parameters())[0], self.penalty, self.n, lag = self.lag) for net in self.sequentials]
		return [l + p for (l, p) in zip(loss, penalty)]

	def calculate_loss(self, X, Y):
		loss = self._loss(X, Y)
		return np.array([num.data[0] for num in loss])

	def calculate_objective(self, X, Y):
		objective = self._objective(X, Y)
		return np.array([num.data[0] for num in objective])

	def get_weights(self, p = None):
		if p is None:
			return [list(net.parameters())[0].data.numpy().copy() for net in self.sequentials]
		else:
			return list(self.sequentials[p].parameters())[0].data.numpy().copy()

	def predict(self, X):
		Y_pred = self._forward(X)
		return np.array([Y[:, 0].data.numpy() for Y in Y_pred]).T

class ParallelMLPDecoding:
	def __init__(self, input_series, output_series, lag, series_units, fc_units, lr, opt, lam, penalty, nonlinearity = 'relu'):
		# Save important arguments
		self.p = output_series
		self.n = input_series
		self.lag = lag
		self.series_out_size = series_units[-1]

		# Set up series networks
		self.series_nets = []
		series_layers = list(zip([lag] + series_units[:-2], series_units[2:]))
		for series in range(input_series):
			net = nn.Sequential()
			for i, (d_in, d_out) in enumerate(series_layers):
				net.add_module('series fc%d' % i, nn.Linear(d_in, d_out, bias = True))
				if nonlinearity == 'series relu':
					net.add_module('relu%d' % i, nn.ReLU())
				elif nonlinearity == 'sigmoid':
					net.add_module('series sigmoid%d' % i, nn.Sigmoid())
				else:
					raise ValueError('nonlinearity must be "relu" or "sigmoid"')
			if len(series_units) == 1:
				d_prev = lag
			else:
				d_prev = series_units[-2]
			net.add_module('series out', nn.Linear(d_prev, series_units[-1], bias = True))
			self.series_nets.append(net)

		# Set up fully connected output networks
		self.out_nets = []
		out_layers = list(zip([series_units[-1] * input_series] + fc_units[:-1], fc_units))
		for target in range(output_series):
			net = nn.Sequential()
			for i, (d_in, d_out) in enumerate(out_layers):
				net.add_module('fc%d' % i, nn.Linear(d_in, d_out, bias = True))
				if nonlinearity == 'relu':
					net.add_module('relu%d' % i, nn.ReLU())
				elif nonlinearity == 'sigmoid':
					net.add_module('sigmoid%d' % i, nn.Sigmoid())
				else:
					raise ValueError('nonlinearity must be "relu" or "sigmoid"')
			net.add_module('out', nn.Linear(fc_units[-1], 1, bias = True))
			self.out_nets.append(net)

		# Set up optimizer
		self.task = 'regression'
		self.loss_fn = nn.MSELoss()
		self.lr = lr
		self.lam = lam
		self.penalty = penalty

		param_list = []
		for net in chain(self.series_nets, self.out_nets):
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








	def _train_prox(self, X, Y):
		loss = self._loss(X, Y)
		total_loss = sum(loss)

		# Take gradient step
		[net.zero_grad() for net in chain(self.series_nets, self.out_nets)]

		total_loss.backward()



		self.optimizer.step()

		# Apply prox operator
		[prox_operator(list(net.parameters())[0], self.penalty, self.n, self.lr, self.lam, lag = self.series_out_size) for net in self.out_nets]

	def _train_builtin(self, X, Y):
		objective = self._objective(X, Y)

		# Run optimizer
		[net.zero_grad() for net in chain(self.series_nets, self.out_nets)]

		# total_loss.backward()
		sum(objective).backward()
		self.optimizer.step()

	def _forward(self, X):
		X_var = Variable(torch.from_numpy(X).float())
		series_out = [self.series_nets[i](X_var[:, (i * self.lag):((i + 1) * self.lag)]) for i in range(self.n)]
		series_layer = torch.cat(series_out, dim = 1)
		return [net(series_layer) for net in self.out_nets]

	def _loss(self, X, Y):
		Y_pred = self._forward(X)
		Y_var = [Variable(torch.from_numpy(Y[:, target][:, np.newaxis]).float()) for target in range(self.p)]
		return [self.loss_fn(Y_pred[target], Y_var[target]) for target in range(self.p)]

	def _objective(self, X, Y):
		loss = self._loss(X, Y)
		penalty = [self.lam * apply_penalty(list(net.parameters())[0], self.penalty, self.n, lag = self.series_out_size) for net in self.out_nets]
		return [l + p for (l, p) in zip(loss, penalty)]

	def calculate_loss(self, X, Y):
		loss = self._loss(X, Y)
		return [num.data[0] for num in loss]

	def calculate_objective(self, X, Y):
		objective = self._objective(X, Y)
		return np.array([num.data[0] for num in objective])

	def get_weights(self, p = None):
		if p is None:
			return [list(net.parameters())[0].data.numpy().copy() for net in self.out_nets]
		else:
			return list(self.out_nets[p].parameters())[0].data.numpy().copy()

	def predict(self, X):
		Y_pred = self._forward(X)
		return np.array([Y[:, 0].data.numpy() for Y in Y_pred]).T
