import torch

import numpy as np

def apply_penalty(W, penalty, num_inputs, lag = 1):
	if penalty == 'group_lasso':
		group_loss = [torch.norm(W[:, (i * lag):((i + 1) * lag)], p = 2) for i in range(num_inputs)]
		total = sum(group_loss)
	elif penalty == 'hierarchical':
		hierarchical_loss = [torch.norm(W[:, (i * lag):((i + 1) * lag - j)], p = 2) for i in range(num_inputs) for j in range(lag)]
		total = sum(hierarchical_loss)
	elif penalty == 'stacked':
		column_loss = [torch.norm(W[:, i], p = 2) for i in range(lag * num_inputs)]
		group_loss = [torch.norm(W[:, (i * lag):((i + 1) * lag)], p = 2) for i in range(num_inputs)]
		total = sum(column_loss) + sum(group_loss)
	else:
		raise ValueError('unsupported penalty')

	return total

def prox_operator(W, penalty, num_inputs, lr, lam, lag = 1):
	if penalty == 'group_lasso':
		_prox_group_lasso(W, num_inputs, lag, lr, lam)
	elif penalty == 'hierarchical':
		_prox_hierarchical(W, num_inputs, lag, lr, lam)
	elif penalty == 'stacked':
		_prox_stacked(W, num_inputs, lag, lr, lam)
	else:
		raise ValueError('unsupported penalty')

def _prox_group_lasso(W, num_inputs, lag, lr, lam):
	'''
		Apply prox operator
	'''
	C = W.data.clone().numpy()

	h, l = C.shape
	C = np.reshape(C, newshape = (lag * h, num_inputs), order = 'F')
	C = _prox_update(C, lam, lr)
	C = np.reshape(C, newshape = (h, l), order = 'F')

	W.data = torch.from_numpy(C)

def _prox_hierarchical(W, num_inputs, lag, lr, lam):
	''' 
		Apply prox operator
	'''
	C = W.data.clone().numpy()

	h, l = C.shape
	C = np.reshape(C, newshape = (lag * h, num_inputs), order = 'F')
	for i in range(1, lag + 1):
		end = i * h
		temp = C[range(end), :]
		C[range(end), :] = _prox_update(temp, lam, lr)

	C = np.reshape(C, newshape = (h, l), order = 'F')

	W.data = torch.from_numpy(C)

def _prox_stacked(W, num_inputs, lag, lr, lam):
	'''
		Apply prox operator
	'''
	C = W.data.clone().numpy()

	h, l = C.shape
	C = np.reshape(C, newshape = (lag * h, num_inputs), order = 'F')
	for i in range(lag):
		start = i * h
		end = (i + 1) * h
		temp = C[range(start, end), :]
		C[range(start, end), :] = _prox_update(temp, lam, lr)
	C = _prox_update(C, lam, lr)
	C = np.reshape(C, newshape = (h, l), order = 'F')

	W.data = torch.from_numpy(C)

def _prox_update(W, lam, lr):
	'''
		Apply prox operator to a matrix, where columns each have group lasso penalty
	'''
	W = W.copy()

	norm_value = np.linalg.norm(W, axis = 0, ord = 2)
	norm_value_gt = norm_value >= (lam * lr)
	
	W[:, np.logical_not(norm_value_gt)] = 0.0
	W[:, norm_value_gt] = W[:, norm_value_gt] * (1 - np.divide(lam * lr, norm_value[norm_value_gt][np.newaxis, :]))

	return W
