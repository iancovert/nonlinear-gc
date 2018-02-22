import numpy as np

def run_experiment(model, X_train, Y_train, nepoch, verbose = True, loss_check = 100, predictions = False):
	if not model.opt == 'line':
		raise ValueError('optimization method must be line search')

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

def run_recurrent_experiment(model, X_train, Y_train, nepoch, verbose = True, loss_check = 100, predictions = False):
	if not model.opt == 'line':
		raise ValueError('optimization method must be line search')

	if type(X_train) is list:	
		raise ValueError('data must be formatted as 2D or 3D tensor (np.ndarray)')

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
