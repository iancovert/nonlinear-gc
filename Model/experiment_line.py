import numpy as np

def run_experiment(model, X_train, Y_train, nepoch, mbsize = None, verbose = True, loss_check = 100, predictions = False):
	if not model.opt == 'line':
		raise ValueError('optimization method must be line search')

	# Batch parameters
	minibatches = not mbsize is None
	if minibatches:
		T_train = X_train.shape[0]
		n_batches = int(np.ceil(T_train / mbsize))

		def minibatch(X, Y, start, end):
			x_batch = X[range(start, end), :]
			if len(Y.shape) == 1:
				y_batch = Y[range(start, end)]
			else:
				y_batch = Y[range(start, end), :]
			return x_batch, y_batch

	# Determine output size
	nchecks = max(int(nepoch / loss_check), 1)
	if len(Y_train.shape) == 1:
		d_out = 1
	else:
		d_out = Y_train.shape[1]

	# Prepare for training
	train_loss = np.zeros((nchecks, d_out))
	train_objective = np.zeros((nchecks, d_out))
	counter = 0
	improvement = True
	epoch = 0

	# Begin training
	while epoch < nepoch and improvement:

		# Run training step
		if minibatches:
			improvement = False

			for i in range(n_batches - 1):
				x_batch, y_batch = minibatch(X_train, Y_train, i * mbsize, (i + 1) * mbsize)
				improvement = improvement or model.train(x_batch, y_batch)

			x_batch, y_batch = minibatch(X_train, Y_train, i * mbsize, T_train)
			improvement = improvement or model.train(x_batch, y_batch)

		else:
			improvement = model.train(X_train, Y_train)

		# Check progress
		if (epoch + 1) % loss_check == 0:
			# Save results
			train_loss[counter, :] = model.calculate_loss(X_train, Y_train)
			train_objective[counter, :] = model.calculate_objective(X_train, Y_train)

			# Print results
			if verbose:
				print('----------')
				print('epoch %d' % epoch)
				print('train loss = %e' % train_loss[counter, 0])
				print('train objective = %e' % train_objective[counter, 0])
				print('----------')

			counter += 1
		
		epoch += 1

	if verbose:
		print('Done training')

	if predictions:
		return train_loss[:counter, :], train_objective[:counter, :], model.get_weights(), model.predict(X_train)
	else:
		return train_loss[:counter, :], train_objective[:counter, :], model.get_weights()

def run_recurrent_experiment(model, X_train, Y_train, nepoch, window_size = None, stride_size = None, verbose = True, loss_check = 100, predictions = False):
	if not model.opt == 'line':
		raise ValueError('optimization method must be line search')

	if type(X_train) is np.ndarray:	
		if len(X_train.shape) == 3:
			format = 'replicates'
		else:
			format = None
	else:
		format = 'list'

	# Window parameters
	if window_size is not None:
		windowed = True
		if stride_size is None:
			stride_size = window_size

	def train_on_series(X, Y):
		T = X.shape[0]
		if windowed and T > window_size:
			improvement = False
			start = 0
			end = window_size

			while end < T + 1:
				x_batch = X[range(start, end), :]
				y_batch = Y[range(start, end), :]
				improvement = improvement or model.train(x_batch, y_batch)

				start = start + stride_size
				end = start + window_size

			return improvement

		else:
			return model.train(X, Y)

	# Determine output size
	nchecks = max(int(nepoch / loss_check), 1)
	if not format == 'list':
		if len(Y_train.shape) == 1:
			d_out = 1
		else:
			d_out = Y_train.shape[-1]
	else:
		d_out = Y_train[0].shape[0]

	# Prepare for training
	train_loss = np.zeros((nchecks, d_out))
	train_objective = np.zeros((nchecks, d_out))
	counter = 0
	improvement = True
	epoch = 0

	# Begin training
	while epoch < nepoch and improvement:

		if format == 'replicates':
			improvement = False
			for i in range(X_train.shape[1]):
				x_batch = X_train[:, i, :]
				y_batch = Y_train[:, i, :]
				improvement = improvement or train_on_series(x_batch, y_batch)

		elif format == 'list':
			improvement = False
			for x_batch, y_batch in zip(X_train, Y_train):
				improvement = improvement or train_on_series(x_batch, y_batch)

		else:
			improvement = train_on_series(X_train, Y_train)

		# Check progress
		if (epoch + 1) % loss_check == 0:
			# Save results
			if not format == 'list':
				train_loss[counter, :] = model.calculate_loss(X_train, Y_train)
				train_objective[counter, :] = model.calculate_objective(X_train, Y_train)

			else:
				t_loss = [model.calculate_loss(x, y) for (x, y) in zip(X_train, Y_train)]
				train_loss[counter, :] = np.average(np.array(t_loss), weights = [x.shape[0] for x in X_train])

				t_obj = [model.calculate_objective(x, y) for (x, y) in zip(X_train, Y_train)]
				train_objective[counter, :] = np.average(np.array(t_obj), weights = [x.shape[0] for x in X_train])

			# Print results
			if verbose:
				print('----------')
				print('epoch %d' % epoch)
				print('train loss = %e' % train_loss[counter, 0])
				print('train objective = %e' % train_objective[counter, 0])
				print('----------')

			counter += 1

		epoch += 1

	if verbose:
		print('Done training')

	if predictions:
		return train_loss[:counter, :], train_objective[:counter, :], model.get_weights(), model.predict(X_train)
	else:
		return train_loss[:counter, :], train_objective[:counter, :], model.get_weights()
