import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from regularize import *

import numpy as np

class IIDEncoding:
	def __init__(self, input_size, output_size, hidden_units, lr, opt, lam, penalty_groups, nonlinearity = 'relu', task = 'regression'):
		# Set up network
		net = torch.nn.Sequential()
		layers = list(zip([input_size] + hidden_units[1:], hidden_units))

		for i, (d_in, d_out) in enumerate(layers):
			net.add_module('fc%d' % i, nn.Linear(d_in, d_out, bias = True))
			if nonlinearity == 'relu':
				net.add_module('relu%d' % i, nn.ReLU())
			elif nonlinearity == 'sigmoid':
				net.add_module('sigmoid%d' % i, nn.Sigmoid())
			else:
				raise ValueError('nonlinearity must be "relu" or "sigmoid"')
		net.add_module('out', nn.Linear(hidden_units[-1], output_size, bias = True))
		
		self.net = net

		# Set up optimizer
		self.task = task
		if task == 'regression':
			self.loss_fn = nn.MSELoss()
		elif task == 'classification':
			self.loss_fn = nn.CrossEntropyLoss()
		else:
			raise ValueError('task must be regression or classification')
		self.lr = lr
		self.lam = lam
		self.penalty_groups = penalty_groups
		self.input_size = input_size

		if opt == 'prox':
			self.optimizer = optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
			self.train = self._train_prox

		else:
			if opt == 'adam':
				self.optimizer = optim.Adam(net.parameters(), lr = lr)
			elif opt == 'sgd':
				self.optimizer = optim.SGD(net.parameters(), lr = lr)
			elif opt == 'momentum':
				self.optimizer = optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
			else:
				raise ValueError('opt must be a valid option')

			self.train = self._train_builtin

	def _train_builtin(self, X, Y):
		loss = self._loss(X, Y)

		# Add regularization penalties
		W = list(self.net.parameters())[0]
		if self.penalty_groups == -1:
			penalty = apply_penalty(W, 'group_lasso', self.input_size)
		else:
			penalty_list = []
			for start, end in zip(self.penalty_groups[:-1], self.penalty_groups[1:]):
				penalty_list.append(torch.norm(W[:, start:end], p = 2))
			penalty = sum(penalty_list)

		# Compute total loss
		total_loss = loss + self.lam * penalty

		# Run optimizer
		self.net.zero_grad()

		total_loss.backward()
		self.optimizer.step()

	def _train_prox(self, X, Y):
		loss = self._loss(X, Y)

		# Run optimizer
		self.net.zero_grad()
		loss.backward()
		self.optimizer.step()

		# Apply prox operator
		W = list(self.net.parameters())[0]
		if self.penalty_groups == -1:
			prox_operator(W, 'group_lasso', self.input_size, self.lr, self.lam)
		else:
			C = W.data.clone().numpy()
			for start, end in zip(self.penalty_groups[:-1], self.penalty_groups[1:]):
				norm = np.linalg.norm(C[:, start:end], ord = 2)
				if norm >= self.lr * self.lam:
					C[:, start:end] = C[:, start:end] * (1 - self.lr * self.lam / norm)
				else: 
					C[:, start:end] = 0.0
			W.data = torch.from_numpy(C)

	def calculate_loss(self, X, Y):
		return self._loss(X, Y).data.numpy()[0]

	def calculate_accuracy(self, X, Y):
		if self.task != 'classification':
			raise Exception('cannot calculate accuracy for regression task')
		Y_var = Variable(torch.from_numpy(Y).long())
		Y_pred = self._forward(X)
		maxes, inds = torch.max(Y_pred, dim = 1)
		return (Y_var == inds).float().sum().data[0] / float(len(inds))

	def _forward(self, X):
		X_var = Variable(torch.from_numpy(X).float())
		return self.net(X_var)

	def _loss(self, X, Y):
		if self.task == 'regression':
			Y_var = Variable(torch.from_numpy(Y).float())
		else:
			Y_var = Variable(torch.from_numpy(Y).long())
		Y_pred = self._forward(X)
		return self.loss_fn(Y_pred, Y_var)

	def get_weights(self):
		return list(self.net.parameters())[0].data.numpy()

	def predict(self, X):
		Y_pred = self._forward(X)
		return Y_pred.data.numpy()
