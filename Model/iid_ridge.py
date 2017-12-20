import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

class RidgeNetwork:
	def __init__(self, input_size, output_size, hidden_units, lr, opt, lam, nonlinearity = 'relu', task = 'regression'):
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
		self.lam = lam

		if opt == 'adam':
			self.optimizer = optim.Adam(net.parameters(), lr = lr)
		elif opt == 'sgd':
			self.optimizer = optim.SGD(net.parameters(), lr = lr)
		elif opt == 'momentum':
			self.optimizer = optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
		else:
			raise ValueError('opt must be a valid option')

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

	def calculate_loss(self, X, Y):
		return self._loss(X, Y).data.numpy()[0]

	def calculate_accuracy(self, X, Y):
		if self.task != 'classification':
			raise Exception('cannot calculate accuracy for regression task')
		Y_var = Variable(torch.from_numpy(Y).long())
		Y_pred = self._forward(X)
		maxes, inds = torch.max(Y_pred, dim = 1)
		return (Y_var == inds).float().sum().data[0] / float(len(inds))

	def train(self, X, Y):
		loss = self._loss(X, Y)

		# Add regularization penalties
		for param in self.net.parameters():
			# Check if parameter is a weight matrix
			if len(param.size()) == 2:
				loss = loss + torch.norm(param, p = 2) ** 2

		# Run optimizer
		self.net.zero_grad()

		loss.backward()
		self.optimizer.step()

	def predict(self, X):
		Y_pred = self._forward(X)
		return Y_pred.data.numpy()

	def get_weights(self):
		return None
