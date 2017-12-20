import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from itertools import chain

from regularize import *

class ParallelMLPEncoding:
	def __init__(self, input_series, output_series, lag, hidden_units, lr, opt, lam, penalty, nonlinearity = 'relu'):
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

		# Set up optimizer
		self.task = 'regression'
		self.loss_fn = nn.MSELoss()
		self.lr = lr
		self.lam = lam
		self.penalty = penalty

		param_list = []
		for net in self.sequentials:
			param_list = param_list + list(net.parameters())
		
		if opt == 'prox':
			self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)
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
		# Compute total loss
		loss = self._loss(X, Y)
		total_loss = sum(loss)

		# Run optimizer
		[net.zero_grad() for net in self.sequentials]

		total_loss.backward()
		self.optimizer.step()

		# Apply prox operator
		[prox_operator(list(net.parameters())[0], self.penalty, self.n, self.lr, self.lam, lag = self.lag) for net in self.sequentials]

	def _train_builtin(self, X, Y):
		# Compute total loss
		loss = self._loss(X, Y)
		penalty = [apply_penalty(list(net.parameters())[0], self.penalty, self.n, lag = self.lag) for net in self.sequentials]
		total_loss = sum(loss) + self.lam * sum(penalty)

		# Run optimizer
		[net.zero_grad() for net in self.sequentials]

		total_loss.backward()
		self.optimizer.step()

	def _forward(self, X):
		X_var = Variable(torch.from_numpy(X).float())
		return [net(X_var) for net in self.sequentials]

	def _loss(self, X, Y):
		Y_pred = self._forward(X)
		Y_var = [Variable(torch.from_numpy(Y[:, target][:, np.newaxis]).float()) for target in range(self.p)]
		return [self.loss_fn(Y_pred[target], Y_var[target]) for target in range(self.p)]

	def calculate_loss(self, X, Y):
		loss = self._loss(X, Y)
		return np.array([num.data[0] for num in loss])

	def get_weights(self):
		return [list(net.parameters())[0].data.numpy() for net in self.sequentials]

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
			self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)
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
		loss = self._loss(X, Y)
		penalty = [apply_penalty(list(net.parameters())[0], self.penalty, self.n, lag = self.series_out_size) for net in self.out_nets]
		total_loss = sum(loss) + self.lam * sum(penalty)

		# Run optimizer
		[net.zero_grad() for net in chain(self.series_nets, self.out_nets)]

		total_loss.backward()
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

	def calculate_loss(self, X, Y):
		loss = self._loss(X, Y)
		return [num.data[0] for num in loss]

	def get_weights(self):
		return [list(net.parameters())[0].data.numpy() for net in self.out_nets]

	def predict(self, X):
		Y_pred = self._forward(X)
		return np.array([Y[:, 0].data.numpy() for Y in Y_pred]).T
