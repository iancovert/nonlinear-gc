import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from itertools import chain

class ParallelLSTMVanilla:
	def __init__(self, input_series, output_series, hidden_size, hidden_layers, lr, opt):
		# Set up networks
		self.lstms = [nn.LSTM(input_series, hidden_size, hidden_layers) for _ in range(output_series)]
		self.out_layers = [nn.Linear(hidden_size, 1) for _ in range(output_series)]

		# Save important arguments
		self.input_series = input_series
		self.output_series = output_series
		self.hidden_size = hidden_size
		self.hidden_layers = hidden_layers

		# Set up optimizer
		self.task = 'regression'
		self.loss_fn = nn.MSELoss()
		self.lr = lr

		param_list = []
		for net in chain(self.lstms, self.out_layers):
			param_list = param_list + list(net.parameters())

		if opt == 'adam':
			self.optimizer = optim.Adam(param_list, lr = lr)
		elif opt == 'sgd':
			self.optimizer = optim.SGD(param_list, lr = lr)
		elif opt == 'momentum':
			self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)
		else:
			raise ValueError('opt must be a valid option')

	def _init_hidden(self):
		return tuple(
			(Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)), 
				Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)))
			for _ in self.lstms
		)

	def _forward(self, X, hidden = None, return_hidden = False):
		n, p = X.shape
		X_var = Variable(torch.from_numpy(X).float())
		if hidden is None:
			hidden = self._init_hidden()
		
		lstm_return = [self.lstms[target](X_var.view(n, 1, p), hidden[target]) for target in range(self.output_series)]
		lstm_return = list(zip(*lstm_return))
		lstm_out, lstm_hidden = lstm_return
		net_out = [self.out_layers[target](lstm_out[target].view(n, self.hidden_size)) for target in range(self.output_series)]
		
		if return_hidden:
			return net_out, lstm_hidden
		else:
			return net_out
	
	def _loss(self, X, Y, hidden = None, return_hidden = False):
		Y_pred, hidden = self._forward(X, hidden = hidden, return_hidden = True)
		Y_var = [Variable(torch.from_numpy(Y[:, target][:, np.newaxis]).float()) for target in range(self.output_series)]
		loss = [self.loss_fn(Y_pred[target], Y_var[target]) for target in range(self.output_series)]
		
		if return_hidden:
			return loss, hidden
		else:
			return loss

	def train(self, X, Y, truncation = None):
		if truncation is None:
			self._train_helper(X, Y)

		else:
			self._truncated_training(X, Y, truncation)

	def _train_helper(self, X, Y, hidden = None, return_hidden = False):
		loss, hidden = self._loss(X, Y, hidden = hidden, return_hidden = True)
		total_loss = sum(loss)

		# Run optimizer
		[net.zero_grad() for net in chain(self.lstms, self.out_layers)]

		total_loss.backward()
		self.optimizer.step()

		if return_hidden:
			return hidden

	def _truncated_training(self, X, Y, truncation):
		T = X.shape[0]
		num = int(np.ceil(T / truncation))
		hidden = self._init_hidden()
		for i in range(num - 1):
			X_subset = X[range(i * truncation, (i + 1) * truncation), :]
			Y_subset = Y[range(i * truncation, (i + 1) * truncation), :]
			hidden = self._train_helper(X_subset, Y_subset, hidden = hidden, return_hidden = True)

			# Repackage hidden
			hidden = repackage_hidden(hidden)

		X_subset = X[((num - 1) * truncation):, :]
		Y_subset = Y[((num - 1) * truncation):, :]
		self._train_helper(X_subset, Y_subset, hidden = hidden)

	def calculate_loss(self, X, Y):
		loss = self._loss(X, Y)
		return [num.data[0] for num in loss]

	def predict(self, X):
		Y_pred = self._forward(X)
		return np.array([Y[:, 0].data.numpy() for Y in Y_pred]).T

	def get_weights(self):
		return None	


def repackage_hidden(hidden):
	if type(hidden) == Variable:
		return Variable(hidden.data)
	else:
		return tuple(repackage_hidden(h) for h in hidden)