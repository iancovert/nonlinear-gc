import torch

import numpy as np

def apply_penalty(W, penalty, num_inputs, lag = 1):
	if penalty == 'group_lasso':
		loss_list = [torch.norm(W[:, (i * lag):((i + 1) * lag)], p = 2) for i in range(num_inputs)]
	elif penalty == 'hierarchical':
		loss_list = [torch.norm(W[:, (i * lag):((i + 1) * lag - j)], p = 2) for i in range(num_inputs) for j in range(lag)]
	else:
		raise ValueError('penalty must be group_lasso or hierarchical')

	return sum(loss_list)

def prox_operator(W, penalty, num_inputs, lr, lam, lag = 1):
	if penalty == 'group_lasso':
		_prox_group_lasso(W, penalty, num_inputs, lag, lr, lam)
	elif penalty == 'hierarchical':
		_prox_hierarchical(W, penalty, num_inputs, lag, lr, lam)
	else:
		raise ValueError('penalty must be group_lasso or hierarchical')

def _prox_group_lasso(W, penalty, num_inputs, lag, lr, lam):
	'''
		Apply prox operator directly (not through prox of conjugate)
	'''
	C = W.data.clone().numpy()

	# if retain is not None:
	# 	col = C[(retain + 1) * lag - 1, :].copy()
	# 	C[(retain + 1) * lag - 1, :] = 0.0

	h, l = C.shape
	C = np.reshape(C, newshape = (lag * h, num_inputs), order = 'F')
	C = _prox_update(C, lam, lr)
	C = np.reshape(C, newshape = (h, l), order = 'F')

	# if retain is not None:
	# 	C[(retain + 1) * lag - 1, :] = col

	W.data = torch.from_numpy(C)

def _prox_hierarchical(W, penalty, num_inputs, lag, lr, lam):
	''' 
		Apply prox operator for each penalty
	'''
	C = W.data.clone().numpy()

	h, l = C.shape
	C = np.reshape(C, newshape = (lag * h, num_inputs), order = 'F')
	for i in range(1, lag + 1):
		end = i * h
		temp = C[range(end), :]
		C[range(end), :] = _prox_update(temp, lam, lr)

		# if i <= lag or retain is None:
		# 	temp = C[range(end), :]
		# 	C[range(end), :] = _prox_update(temp, lam, lr)
		# else:
		# 	inds = np.concatenate((np.arange(retain - 1), np.arange(retain + 1, p)))
		# 	temp = C[range(end), inds]
		# 	C[range(end), inds] = _prox_update(temp, lam, lr[inds])

	C = np.reshape(C, newshape = (h, l), order = 'F')

	W.data = torch.from_numpy(C)

def _prox_update(W, lam, lr):
	'''
		Apply prox operator to a matrix, where columns each have group lasso penalty
	'''
	if type(lr) is not float:
		lr = np.array(lr)

	W = W.copy()

	norm_value = np.linalg.norm(W, axis = 0, ord = 2)
	norm_value_gt = norm_value >= (lam * lr)
	
	W[:, np.logical_not(norm_value_gt)] = 0.0
	W[:, norm_value_gt] = W[:, norm_value_gt] * (1 - lam * np.divide(lr[norm_value_gt], norm_value[norm_value_gt][np.newaxis, :]))

	return W
