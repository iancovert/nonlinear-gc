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

def format_replicate_ts_data(X, lag = 1, validation = 0.1):
	T, d, p = X.shape
	T_new = T - lag
	X_out = np.zeros((T_new, d, p * lag))
	Y_out = np.zeros((T_new, d, p))

	for r in range(d):
		for t in range(lag, T):
			X_out[t - lag, r, :] = X[range(t - lag, t), r, :].flatten(order = 'F')
			Y_out[t - lag, r, :] = X[t, r, :]

	d_val = int(d * validation)
	d_train = d - d_val

	X_train = X_out[:, range(d_train), :]
	X_val = X_out[:, range(d_train, d), :]
	Y_train = Y_out[:, range(d_train), :]
	Y_val = Y_out[:, range(d_train, d), :]

	# Reshape
	X_train = np.transpose(X_train, axes = (1, 0, 2))
	X_train = np.reshape(X_train, newshape = ((T - lag) * d_train, p * lag), order = 'C')
	X_val = np.transpose(X_val, axes = (1, 0, 2))
	X_val = np.reshape(X_val, newshape = ((T - lag) * d_val, p * lag), order = 'C')
	Y_train = np.transpose(Y_train, axes = (1, 0, 2))
	Y_train = np.reshape(Y_train, newshape = ((T - lag) * d_train, p), order = 'C')
	Y_val = np.transpose(Y_val, axes = (1, 0, 2))
	Y_val = np.reshape(Y_val, newshape = ((T - lag) * d_val, p), order = 'C')

	return X_train, Y_train, X_val, Y_val

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

def YX_list(X):
	n = len(X)
	Y_train = list()
	X_train = list()
	for i in range(n):
		Y_train.append(X[i][1:,:])
		X_train.append(X[i][:-1,:])

	return(Y_train, X_train)

def list_to_matrix(X):
    return(np.concatenate(X))

def reshapape_list(X,d=20):
	nl = len(X)
	temp_tense = list()
	for i in range(nl):
		t = X[i].shape[0]
		nb = int(np.floor(t/d))
		temp_tense.append(np.zeros((d,nb,X[i].shape[1])))
		for j in range(nb):
			start = j*d
			end = (j + 1)*d
			print(range(start,end))
			temp_tense[i][:,j,:] = X[i][range(start,end),:]


	final_tense = np.concatenate(temp_tense,axis=1)
	return(final_tense)

def tensorize_sequence(X, window = 20, stride = None):
	if stride is None:
		stride = window

	sequence_list = []
	T, p = X.shape
	start = 0
	end = window
	while end < T:
		tensor = np.zeros((window, 1, p))
		tensor[:, 0, :] = X[range(start, end), :]
		sequence_list.append(tensor)

		start += stride
		end += stride

	return np.concatenate(sequence_list, axis = 1)

def normalize_list(X,type="cat"):
    if (type == "cat"):
        Xd = np.concatenate(X)
        stds = np.std(Xd,axis = 0)
        means = Xd.mean(axis=0)
    
        for i in range(len(X)):
            X[i] = (X[i] - means)/stds

    return(X)

def normalize(X, scale_global = False):
	if len(X.shape) == 2:
		X_centered = X - np.mean(X, axis = 0)
		
		if scale_global:
			sigma = np.sqrt(np.var(X))
		else:
			sigma = np.sqrt(np.var(X, axis = 0))
		
		return np.divide(X_centered, sigma)
		
	else:
		X_normalized = X.copy()
		for i in range(X.shape[2]):
			X_normalized[:, :, i] = X_normalized[:, :, i] - np.mean(X_normalized[:, :, i])
			
		if scale_global:
			X_normalized = np.divide(X_normalized, np.sqrt(np.var(X_normalized)))
		else:
			for i in range(X.shape[2]):
				X_normalized[:, :, i] = np.divide(X_normalized[:, :, i], np.sqrt(np.var(X_normalized[:, :, i])))
		
		return X_normalized
