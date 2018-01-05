import numpy as np

def run_experiment(model, X_train, Y_train, X_val, Y_val, nepoch, mbsize = None, verbose = True, loss_check = 100, predictions = False):
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
	val_loss = np.zeros((nchecks, d_out))
	if model.task == 'classification':
		train_accuracy = np.zeros((nchecks, d_out))
		val_accuracy = np.zeros((nchecks, d_out))
	counter = 0

	# Begin training
	for epoch in range(nepoch):

		# Run training step
		if minibatches:
			for i in range(n_batches - 1):
				x_batch, y_batch = minibatch(X_train, Y_train, i * mbsize, (i + 1) * mbsize)
				model.train(x_batch, y_batch)

			x_batch, y_batch = minibatch(X_train, Y_train, i * mbsize, T_train)
			model.train(x_batch, y_batch)

		else:
			model.train(X_train, Y_train)

		# Check progress
		if epoch % loss_check == 0:
			# Save results
			train_loss[counter, :] = model.calculate_loss(X_train, Y_train)
			val_loss[counter, :] = model.calculate_loss(X_val, Y_val)
			if model.task == 'classification':
				train_accuracy[counter, :] = model.calculate_accuracy(X_train, Y_train)
				val_accuracy[counter, :] = model.calculate_accuracy(X_val, Y_val)

			# Print results
			if verbose:
				print('----------')
				print('epoch %d' % epoch)
				print('train loss = %e' % train_loss[counter, 0])
				print('val loss = %e' % val_loss[counter, 0])
				print('----------')

			counter += 1

	weights = model.get_weights()

	if verbose:
		print('Done training')

	if predictions:
		if model.task == 'classification':
			return train_loss, val_loss, train_accuracy, val_accuracy, weights, model.predict(X_train), model.predict(X_val)
		else:
			return train_loss, val_loss, weights, model.predict(X_train), model.predict(X_val)
	else:
		if model.task == 'classification':
			return train_loss, val_loss, train_accuracy, val_accuracy, weights
		else:
			return train_loss, val_loss, weights

def run_experiment_cv(model, X_train, Y_train, X_val, Y_val, nepoch, mbsize = None, verbose = True, loss_check = 100, predictions = False):
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
	val_loss = np.zeros((nchecks, d_out))
	if model.task == 'classification':
		train_accuracy = np.zeros((nchecks, d_out))
		val_accuracy = np.zeros((nchecks, d_out))
	counter = 0

	# Store best results
	best_properties = [None] * d_out
	best_val_loss = [np.inf] * d_out

	# Begin training
	for epoch in range(nepoch):

		# Run training step
		if minibatches:
			for i in range(n_batches - 1):
				x_batch, y_batch = minibatch(X_train, Y_train, i * mbsize, (i + 1) * mbsize)
				model.train(x_batch, y_batch)

			x_batch, y_batch = minibatch(X_train, Y_train, i * mbsize, T_train)
			model.train(x_batch, y_batch)

		else:
			model.train(X_train, Y_train)

		# Check progress
		if epoch % loss_check == 0:
			# Save results
			train_loss[counter, :] = model.calculate_loss(X_train, Y_train)
			val_loss[counter, :] = model.calculate_loss(X_val, Y_val)
			if model.task == 'classification':
				train_accuracy[counter, :] = model.calculate_accuracy(X_train, Y_train)
				val_accuracy[counter, :] = model.calculate_accuracy(X_val, Y_val)

			# Print results
			if verbose:
				print('----------')
				print('epoch %d' % epoch)
				print('train loss = %e' % train_loss[counter, 0])
				print('val loss = %e' % val_loss[counter, 0])
				print('----------')

			# Check if this is best result so far
			modified = [False] * d_out
			for p in range(d_out):
				if val_loss[counter, p] < best_val_loss[p]:
					modified[p] = True

					best_val_loss[p] = val_loss[counter, p]
					best_properties[p] = {
						'val_loss': val_loss[counter, p],
						'nepoch': epoch,
						'weights': model.get_weights(p = p)
					}

			# Add accuracies, if necessary
			if model.task == 'classification':
				for p in range(d_out):
					if modified[p]:
						best_properties[p]['val_accuracy'] = val_accuracy[counter, p]

			# Add predictions, if necessary
			if predictions and sum(modified) > 0:
				predictions_train = model.predict(X_train)
				predictions_val = model.predict(X_val)

				for p in range(d_out):
					if modified[p]:
						best_properties[p]['predictions_train'] = predictions_train[:, p]
						best_properties[p]['predictions_val'] = predictions_val[:, p]

			counter += 1

	if verbose:
		print('Done training')

	# Assemble results

	return train_loss, val_loss, best_properties

def run_recurrent_experiment(model, X_train, Y_train, X_val, Y_val, nepoch, window_size = None, stride_size = None, truncation = None, verbose = True, loss_check = 100, predictions = False):
	# Window parameters
	T = X_train.shape[0]
	if window_size is not None:
		windowed = True
		if stride_size is None:
			stride_size = window_size

	# Determine output size
	nchecks = max(int(nepoch / loss_check), 1)
	if len(Y_train.shape) == 1:
		d_out = 1
	else:
		d_out = Y_train.shape[1]

	# Prepare for training
	train_loss = np.zeros((nchecks, d_out))
	val_loss = np.zeros((nchecks, d_out))
	counter = 0

	# Begin training
	for epoch in range(nepoch):

		if windowed:
			start = 0
			end = window_size

			while end < T + 1:
				x_batch = X_train[range(start, end), :]
				y_batch = Y_train[range(start, end), :]
				model.train(x_batch, y_batch, truncation = truncation)

				start = start + stride_size
				end = start + window_size


			if start < end:
				x_batch = X_train[range(start, T), :]
				y_batch = Y_train[range(start, T), :]
				model.train(x_batch, y_batch, truncation = truncation)

		else:
			model.train(X_train, Y_train, truncation = truncation)

		# Check progress
		if epoch % loss_check == 0:
			# Save results
			train_loss[counter, :] = model.calculate_loss(X_train, Y_train)
			val_loss[counter, :] = model.calculate_loss(X_val, Y_val)

			# Print results
			if verbose:
				print('----------')
				print('epoch %d' % epoch)
				print('train loss = %e' % train_loss[counter, 0])
				print('val loss = %e' % val_loss[counter, 0])
				print('----------')

			counter += 1

	weights = model.get_weights()

	if verbose:
		print('Done training')

	if predictions:
		return train_loss, val_loss, weights, model.predict(X_train), model.predict(X_val)
	else:
		return train_loss, val_loss, weights

def run_recurrent_experiment_cv(model, X_train, Y_train, X_val, Y_val, nepoch, window_size = None, stride_size = None, truncation = None, verbose = True, loss_check = 100, predictions = False):
	# Window parameters
	T = X_train.shape[0]
	if window_size is not None:
		windowed = True
		if stride_size is None:
			stride_size = window_size

	# Determine output size
	nchecks = max(int(nepoch / loss_check), 1)
	if len(Y_train.shape) == 1:
		d_out = 1
	else:
		d_out = Y_train.shape[1]

	# Prepare for training
	train_loss = np.zeros((nchecks, d_out))
	val_loss = np.zeros((nchecks, d_out))
	counter = 0

	# Store best results
	best_properties = [None] * d_out
	best_val_loss = [np.inf] * d_out

	# Begin training
	for epoch in range(nepoch):

		if windowed:
			start = 0
			end = window_size

			while end < T + 1:
				x_batch = X_train[range(start, end), :]
				y_batch = Y_train[range(start, end), :]
				model.train(x_batch, y_batch, truncation = truncation)

				start = start + stride_size
				end = start + window_size


			if start < end:
				x_batch = X_train[range(start, T), :]
				y_batch = Y_train[range(start, T), :]
				model.train(x_batch, y_batch, truncation = truncation)

		else:
			model.train(X_train, Y_train, truncation = truncation)

		# Check progress
		if epoch % loss_check == 0:
			# Save results
			train_loss[counter, :] = model.calculate_loss(X_train, Y_train)
			val_loss[counter, :] = model.calculate_loss(X_val, Y_val)

			# Print results
			if verbose:
				print('----------')
				print('epoch %d' % epoch)
				print('train loss = %e' % train_loss[counter, 0])
				print('val loss = %e' % val_loss[counter, 0])
				print('----------')

			# Check if this is best result so far
			modified = [False] * d_out	
			for p in range(d_out):
				if val_loss[counter, p] < best_val_loss[p]:
					modified[p] = True

					best_val_loss[p] = val_loss[counter, p]
					best_properties[p] = {
						'val_loss': val_loss[counter, p],
						'nepoch': epoch,
						'weights': model.get_weights(p = p)
					}

			# Add predictions, if necessary
			if predictions and sum(modified) > 0:
				train_forecasts = model.predict(X_train)
				val_forecasts = model.predict(X_val)
					
				for p in range(d_out):
					if modified[p]:
						best_properties[p]['predictions_train'] = train_forecasts[:, p]
						best_properties[p]['predictions_val'] = val_forecasts[:, p]

			counter += 1

	if verbose:
		print('Done training')

	# Assemble results

	return train_loss, val_loss, best_properties