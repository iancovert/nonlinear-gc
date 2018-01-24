import numpy as np

def run_experiment(model, X_train, Y_train, nepoch, mbsize = None, verbose = True, loss_check = 100, predictions = False, min_lr = 1e-8, cooldown = False):
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
	if model.task == 'classification':
		train_accuracy = np.zeros((nchecks, d_out))
	counter = 0

	# Store best results
	best_properties = [None] * d_out
	best_obj = [np.inf] * d_out

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
			train_objective[counter, :] = model.calculate_objective(X_train, Y_train)
			
			if model.task == 'classification':
				train_accuracy[counter, :] = model.calculate_accuracy(X_train, Y_train)

			# Print results
			if verbose:
				print('----------')
				print('epoch %d' % epoch)
				print('train loss = %e' % train_loss[counter, 0])
				print('train objective = %e' % train_objective[counter, 0])
				print('----------')

			# Check if this is best result so far
			modified = [False] * d_out
			for p in range(d_out):
				if train_objective[counter, p] < best_obj[p]:
					modified[p] = True

					best_obj[p] = train_objective[counter, p]
					best_properties[p] = {
						'train_objective': train_objective[counter, p],
						'nepoch': epoch,
						'weights': model.get_weights(p = p)
					}
				elif cooldown and train_objective[counter, p] > train_objective[counter - 1, p]:
					model.cooldown(p)

			# Add accuracies, if necessary
			if model.task == 'classification':
				for p in range(d_out):
					if modified[p]:
						best_properties[p]['train_accuracy'] = train_accuracy[counter, p]

			# Add predictions, if necessary
			if predictions and sum(modified) > 0:
				predictions_train = model.predict(X_train)

				for p in range(d_out):
					if modified[p]:
						best_properties[p]['predictions_train'] = predictions_train[:, p]

			counter += 1

			# Check if all learning rates are low enough to stop
			if cooldown and p - sum(modified) > 0 and sum([lr > min_lr for lr in model.lr]) == 0:
				break

	if verbose:
		print('Done training')

	return train_loss[:counter, :], train_objective[:counter, :], best_properties

# def run_recurrent_experiment(model, X_train, Y_train, nepoch, window_size = None, stride_size = None, truncation = None, verbose = True, loss_check = 100, predictions = False, min_lr = 1e-8, cooldown = False):
# 	if len(X_train.shape) == 2:
# 		replicates = False
	
# 		# Window parameters
# 		T = X_train.shape[0]
# 		if window_size is not None:
# 			windowed = True
# 			if stride_size is None:
# 				stride_size = window_size

# 	else:
# 		replicates = True

# 	# Determine output size
# 	nchecks = max(int(nepoch / loss_check), 1)
# 	if len(Y_train.shape) == 1:
# 		d_out = 1
# 	else:
# 		d_out = Y_train.shape[-1]

# 	# Prepare for training
# 	train_loss = np.zeros((nchecks, d_out))
# 	train_objective = np.zeros((nchecks, d_out))
# 	counter = 0

# 	# Store best results
# 	best_properties = [None] * d_out
# 	best_obj = [np.inf] * d_out

# 	# Begin training
# 	for epoch in range(nepoch):

# 		if replicates:
# 			for i in range(X_train.shape[1]):
# 				x_batch = X_train[:, i, :]
# 				y_batch = Y_train[:, i, :]
# 				model.train(x_batch, y_batch, truncation = truncation)

# 		else:
# 			if windowed:
# 				start = 0
# 				end = window_size

# 				while end < T + 1:
# 					x_batch = X_train[range(start, end), :]
# 					y_batch = Y_train[range(start, end), :]
# 					model.train(x_batch, y_batch, truncation = truncation)

# 					start = start + stride_size
# 					end = start + window_size

# 			else:
# 				model.train(X_train, Y_train, truncation = truncation)

# 		# Check progress
# 		if epoch % loss_check == 0:
# 			# Save results
# 			train_loss[counter, :] = model.calculate_loss(X_train, Y_train)
# 			train_objective[counter, :] = model.calculate_objective(X_train, Y_train)

# 			# Print results
# 			if verbose:
# 				print('----------')
# 				print('epoch %d' % epoch)
# 				print('train loss = %e' % train_loss[counter, 0])
# 				print('train objective = %e' % train_objective[counter, 0])
# 				print('----------')

# 			# Check if this is best result so far
# 			modified = [False] * d_out
# 			for p in range(d_out):
# 				if train_objective[counter, p] < best_obj[p]:
# 					modified[p] = True

# 					best_obj[p] = train_objective[counter, p]
# 					best_properties[p] = {
# 						'train_objective': train_objective[counter, p],
# 						'nepoch': epoch,
# 						'weights': model.get_weights(p = p)
# 					}
# 				elif cooldown and train_objective[counter, p] > train_objective[counter - 1, p]:
# 					model.cooldown(p)

# 			# Add predictions, if necessary
# 			if predictions and sum(modified) > 0:
# 				predictions_train = model.predict(X_train)

# 				for p in range(d_out):
# 					if modified[p]:
# 						best_properties[p]['predictions_train'] = predictions_train[:, p]

# 			counter += 1

# 			# Check if all learning rates are low enough to stop
# 			if cooldown and p - sum(modified) > 0 and sum([lr > min_lr for lr in model.lr]) == 0:
# 				break

# 	if verbose:
# 		print('Done training')

# 	return train_loss[:counter, :], train_objective[:counter, :], best_properties

def run_recurrent_experiment(model, X_train, Y_train, nepoch, window_size = None, stride_size = None, truncation = None, verbose = True, loss_check = 100, predictions = False, min_lr = 1e-8, cooldown = False):
	if type(X_train) is np.ndarray:	
		if len(X_train.shape) == 3:
			format = 'replicates'
		else:
			format = None
	else:
		format = 'list'

	# Window parameters
	T = X_train.shape[0]
	if window_size is not None:
		windowed = True
		if stride_size is None:
			stride_size = window_size

	def train_on_series(X, Y):
		if windowed and X.shape[0] > window_size:
			start = 0
			end = window_size

			while end < T + 1:
				x_batch = X[range(start, end), :]
				y_batch = Y[range(start, end), :]
				model.train(x_batch, y_batch, truncation = truncation)

				start = start + stride_size
				end = start + window_size

		else:
			model.train(X, Y, truncation = truncation)

	# Determine output size
	nchecks = max(int(nepoch / loss_check), 1)
	if len(Y_train.shape) == 1:
		d_out = 1
	else:
		d_out = Y_train.shape[-1]

	# Prepare for training
	train_loss = np.zeros((nchecks, d_out))
	train_objective = np.zeros((nchecks, d_out))
	counter = 0

	# Store best results
	best_properties = [None] * d_out
	best_obj = [np.inf] * d_out

	# Begin training
	for epoch in range(nepoch):

		if format == 'replicates':
			for i in range(X_train.shape[1]):
				x_batch = X_train[:, i, :]
				y_batch = Y_train[:, i, :]
				train_on_series(x_batch, y_batch)

		elif format == 'list':
			for x_batch, y_batch in zip(X_train, Y_train):
				train_on_series(x_batch, y_batch)

		else:
			train_on_series(X_train, Y_train)

		# Check progress
		if epoch % loss_check == 0:
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

			# Check if this is best result so far
			modified = [False] * d_out
			for p in range(d_out):
				if train_objective[counter, p] < best_obj[p]:
					modified[p] = True

					best_obj[p] = train_objective[counter, p]
					best_properties[p] = {
						'train_objective': train_objective[counter, p],
						'nepoch': epoch,
						'weights': model.get_weights(p = p)
					}
				elif cooldown and train_objective[counter, p] > train_objective[counter - 1, p]:
					model.cooldown(p)

			# Add predictions, if necessary
			if predictions and sum(modified) > 0:
				predictions_train = model.predict(X_train)

				for p in range(d_out):
					if modified[p]:
						best_properties[p]['predictions_train'] = predictions_train[:, p]

			counter += 1

			# Check if all learning rates are low enough to stop
			if cooldown and p - sum(modified) > 0 and sum([lr > min_lr for lr in model.lr]) == 0:
				break

	if verbose:
		print('Done training')

	return train_loss[:counter, :], train_objective[:counter, :], best_properties