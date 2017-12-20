import numpy as np

def format_ts_data(X, lag = 1, validation = 0.1):
	T, p = X.shape
	T_new = T - lag
	X_out = np.zeros((T_new, p * lag))
	Y_out = np.zeros((T_new, p))

	for t in range(lag, T):
		X_out[t - lag, :] = X[range(t - lag, t), :].flatten(order = 'F')
		Y_out[t - lag, :] = X[t, :]

	T_val = int(T_new * validation)
	T_train = T_new - T_val

	X_train = X_out[range(T_train), :]
	X_val = X_out[range(T_train, T_new), :]
	Y_train = Y_out[range(T_train), :]
	Y_val = Y_out[range(T_train, T_new), :]

	return X_train, Y_train, X_val, Y_val

def format_ts_data_subset(X, on_vector, lag = 1, validation = 0.1):
	if len(on_vector) != X.shape[1]:
		raise ValueError('on_vector must have length p')

	X_train, Y_train, X_val, Y_val = format_ts_data(X, lag, validation)

	return X_train, Y_train[:, on_vector], X_val, Y_val[:, on_vector]

def split_data(X, validation = 0.1, test = None, shuffle = False):
	if shuffle:
		np.random.shuffle(X)
	N = X.shape[0]
	N_val = int(N * validation)
	if test is None:
		N_train = N - N_val
		X_train = X[range(N_train), :]
		X_val = X[range(N_train, N), :]
		return X_train, X_val
	else:
		N_test = int(N * test)
		N_train = N - N_test - N_val
		X_train = X[range(N_train), :]
		X_val = X[range(N_train, N_train + N_val), :]
		X_test = X[range(N_train + N_val, N), :]
		return X_train, X_val, X_test

def whiten_data_cholesky(X):
	X_centered = X - np.mean(X, axis = 0)

	Sigma = np.dot(X_centered.T, X_centered) / X.shape[0]

	if not np.all(np.linalg.eigvals(Sigma) > 0):
		raise ValueError('data matrix is not positive definite')

	L = np.linalg.cholesky(Sigma)
	L_inv = np.linalg.inv(L)

	return np.dot(X_centered, L_inv.T)

def normalize(X):
	X_centered = X - np.mean(X, axis = 0)
	sigma = np.sqrt(np.var(X, axis = 0))
	return np.divide(X_centered, sigma)