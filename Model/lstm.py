import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from itertools import chain

from regularize import *

class ParallelLSTMEncoding:
	def __init__(self, input_series, output_series, hidden_size, hidden_layers, lr, opt, lam):
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
		self.lam = lam
		self.lr = lr

		param_list = []
		for net in chain(self.lstms, self.out_layers):
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
		[net.zero_grad() for net in chain(self.lstms, self.out_layers)]

		total_loss.backward()
		self.optimizer.step()

		# Apply proximal operator
		[prox_operator(lstm.weight_ih_l0, 'group_lasso', self.input_series, self.lr, self.lam) for lstm in self.lstms]

		if return_hidden:
			return hidden

	def _train_builtin_helper(self, X, Y, hidden = None, return_hidden = False):
		loss, hidden = self._loss(X, Y, hidden = hidden, return_hidden = True)
		penalty = [apply_penalty(lstm.weight_ih_l0, 'group_lasso', self.input_series) for lstm in self.lstms]
		total_loss = sum(loss) + self.lam * sum(penalty)

		# Run optimizer
		[net.zero_grad() for net in chain(self.lstms, self.out_layers)]

		total_loss.backward()
		self.optimizer.step()

		if return_hidden:
			return hidden

	def calculate_loss(self, X, Y):
		loss = self._loss(X, Y)
		return [num.data[0] for num in loss]

	def predict(self, X):
		Y_pred = self._forward(X)
		return np.array([Y[:, 0].data.numpy() for Y in Y_pred]).T

	def get_weights(self, p = None):
		if p is None:
			return [lstm.weight_ih_l0.data.numpy() for lstm in self.lstms]
		else:
			return self.lstms[p].weight_ih_l0.data.numpy()

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
		lstm_return = list(zip(*lstm_return))
		lstm_out, lstm_hidden = lstm_return
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
		loss, hidden = self._loss(X, Y, hidden = hidden, return_hidden = True)
		penalty = [apply_penalty(list(net.parameters())[0], 'group_lasso', self.input_series, lag = self.hidden_size) for net in self.out_networks]
		total_loss = sum(loss) + self.lam * sum(penalty)

		# Run optimizer
		[net.zero_grad() for net in chain(self.lstms, self.out_networks)]

		total_loss.backward()
		self.optimizer.step()

		if return_hidden:
			return hidden

	def predict(self, X):
		Y_pred = self._forward(X)
		return np.array([Y[:, 0].data.numpy() for Y in Y_pred]).T

	def calculate_loss(self, X, Y):
		loss = self._loss(X, Y)
		return [num.data[0] for num in loss]

	def get_weights(self):
		return [list(net.parameters())[0].data.numpy() for net in self.out_networks]


def repackage_hidden(hidden):
	if type(hidden) == Variable:
		return Variable(hidden.data)
	else:
		return tuple(repackage_hidden(h) for h in hidden)

# class SingleLSTM:
# 	def __init__(self, input_series, output_series, hidden_size, hidden_layers, lr, opt, lam, penalty):
# 		# Set up networks
# 		self.lstm = nn.LSTM(input_series, hidden_size, hidden_layers)
# 		self.out = nn.Linear(hidden_size, output_series, bias = True)

# 		# Save important arguments
# 		self.input_series = input_series
# 		self.output_series = output_series
# 		self.hidden_size = hidden_size
# 		self.hidden_layers = hidden_layers

# 		# Set up optimizer
# 		self.loss_fn = nn.MSELoss()
# 		self.penalty = penalty
# 		self.lam = lam
# 		self.lr = lr

# 		param_list = []
# 		param_list = param_list + list(self.lstm.parameters())
# 		param_list = param_list + list(self.out.parameters())

# 		if opt == 'prox':
# 			self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)
# 			self.train = self._train_prox
# 		else:
# 			if opt == 'adam':
# 				self.optimizer = optim.Adam(param_list, lr = lr)
# 			elif opt == 'sgd':
# 				self.optimizer = optim.SGD(param_list, lr = lr)
# 			elif opt == 'momentum':
# 				self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)
# 			else:
# 				raise ValueError('opt must be a valid option')

# 			self.train = self._train_builtin

# 	def predict(self, X, hidden = None):
# 		out = self._forward(X, hidden = hidden)
# 		return out.data.numpy()

# 	def _forward(self, X, hidden = None):
# 		if hidden is None:
# 			hidden = self.init_hidden()

# 		n, p = X.shape
# 		X_var = Variable(torch.from_numpy(X).float())

# 		lstm_out, hidden = self.lstm(X_var.view(n, 1, p), hidden)
# 		out = self.out(lstm_out.view(n, self.hidden_size))

# 		return out

# 	def init_hidden(self):
# 		return (Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)), 
# 			Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)))

# 	def _train_prox(self, X, Y, hidden = None):
# 		# Compute mse
# 		Y_pred = self._forward(X, hidden = hidden)
# 		Y_var = Variable(torch.from_numpy(Y).float())
# 		mse = self.loss_fn(Y_pred, Y_var)

# 		# Take gradient step
# 		self.lstm.zero_grad()
# 		self.out.zero_grad()

# 		mse.backward()
# 		self.optimizer.step()

# 		# Apply proximal operator to first weight matrix

# 	def _train_builtin(self, X, Y, hidden = None):
# 		# Compute mse
# 		Y_pred = self._forward(X, hidden = hidden)
# 		Y_var = Variable(torch.from_numpy(Y).float())
# 		mse = self.loss_fn(Y_pred, Y_var)

# 		# Compute regularization penalties on first weight matrix
# 		loss = mse

# 		# Take gradient step
# 		self.lstm.zero_grad()
# 		self.out.zero_grad()

# 		loss.backward()
# 		self.optimizer.step()

# 	def calculate_mse(self, X, Y, hidden = None):
# 		Y_pred = self._forward(X, hidden = hidden)

# 		Y_var = Variable(torch.from_numpy(Y).float())
# 		mse = self.loss_fn(Y_pred, Y_var)
# 		return mse.data[0]

# 	def get_weights(self):
# 		return self.lstm.weight_ih_l0.data.numpy()