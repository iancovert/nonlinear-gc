from __future__ import division
import numpy as np
from scipy.special import logsumexp
from scipy.integrate import odeint

def lorentz_96(y, t, force, b):
	p = y.shape[0]
	dydt = np.zeros(y.shape[0])
	for i in range(p):
		dydt[i] = (y[(i + 1) % p] - y[i - 2]) * y[i - 1] - y[i] + force
	return dydt

"""
K - connection strength
omega - natural frequency for each osccilator
A - the connectivity matrix
"""
def Kuramoto(y, t, omega, A, K):
	p = y.shape[0]
	dydt = np.zeros(p)
	for i in range(p):
		dydt[i] = omega[i]
		base = 0
		for j in range(p):
			base += A[i,j] * np.sin(y[j] - y[i])
		dydt[i] += (K/p) * base
	return dydt

"""
Code to generate coupled kuramoto oscilators 
sparsity - the fraction of edges in graph
p - dimension of system
K - strength of interactions
N - number of output time points
delta_t - sampling between time points
sd - the noise added ontop of the series
num_trials - the number of trials/runs from this grid

outputs: 
Z - a #time points x # replicates x size (p) of series tensor
GC - size (p) x size (p) graph of directed interactions
"""
def kuramoto_model(sparsity, p, K = 2, N = 250, delta_t = 0.1, sd = 0.1, seed = 345, num_trials = 100, standardized = True, cos_transform = True):
	np.random.seed(seed)

	# Determine dependencies
	if standardized:
		GC_on = np.zeros((p,p))
		con_per_comp = int(p * sparsity) - 1
		for i in range(p):
			possible_choices = np.setdiff1d(np.arange(p),i)
			selected = np.random.choice(possible_choices, con_per_comp, replace = False)
			GC_on[i, selected] = 1
			GC_on[i,i] = 1
	else:
		GC_on = np.maximum(np.random.binomial(n = 1, p = sparsity, size = (p, p)), np.eye(p))

	# Generate data
	t = np.linspace(0,N*delta_t,N)

	if num_trials is None:
		omega = np.random.uniform(0.0,2.0,size=p)
		y0 = np.random.uniform(0.0, 2.0 * np.pi, size=p)
		Z = odeint(Kuramoto, y0, t, args = (omega,GC_on,K))
		Z += np.random.normal(loc = 0, scale = sd, size = (N, p))
		if cos_transform:
			Z = np.cos(Z)
	else:
		Z = np.zeros((N,num_trials,p))
		for k in range(num_trials):
			omega = np.random.uniform(0.0,2.0,size=p)
			y0 = np.random.uniform(0.0, 2.0 * np.pi, size=p)
			z = odeint(Kuramoto, y0, t, args = (omega,GC_on,K))
			z += np.random.normal(loc = 0, scale = sd, size = (N, p))
			if cos_transform:
				z = np.cos(z)
			Z[:,k,:] = z

	return Z, GC_on

def lorentz_96_model_2(F, p, N, delta_t = 0.1, sd = 0.1, seed = 543):
	np.random.seed(seed)

	burnin = 100
	N += burnin
	b = 10
	y0 = np.random.normal(loc = 0, scale = 0.01, size = p)
	t = np.linspace(0, N * delta_t, N)

	z = odeint(lorentz_96, y0, t, args = (F,b))

	z += np.random.normal(loc = 0, scale = sd, size = (N, p))

	GC_on = np.zeros((p, p))
	for i in range(p):
		GC_on[i, i] = 1
		GC_on[i, (i - 1) % p] = 1
		GC_on[i, (i - 2) % p] = 1
		GC_on[i, (i + 1) % p] = 1

	return z[range(burnin, N), :], GC_on

def lorentz_96_model(forcing_constant, p, N, delta_t = 0.01, sd = 0.1, noise_add = 'global', seed = 543):
	np.random.seed(seed)

	burnin = 4000
	N += burnin
	z = np.zeros((N, p))
	z[0, :] = np.random.normal(loc = 0, scale = 0.01, size = p)
	for t in range(1, N):
		for i in range(p):
			grad = (z[t - 1, (i + 1) % p] - z[t - 1, i - 2]) * z[t - 1, i - 1] - z[t - 1, i] + forcing_constant
			z[t, i] = delta_t * grad + z[t - 1, i]
			if noise_add == 'step':
				z[t, i] += np.random.normal(loc = 0, scale = sd, size = 1)

	if noise_add == 'global':
		z += np.random.normal(loc = 0, scale = sd, size = (N, p))

	GC_on = np.zeros((p, p))
	for i in range(p):
		GC_on[i, i] = 1
		GC_on[i, (i - 1) % p] = 1
		GC_on[i, (i - 2) % p] = 1
		GC_on[i, (i + 1) % p] = 1

	return z[range(burnin, N), :], GC_on

def stationary_var(beta, p, lag, radius):
	bottom = np.hstack((np.eye(p * (lag-1)), np.zeros((p * (lag - 1), p))))  
	beta_tilde = np.vstack((beta,bottom))
	eig = np.linalg.eigvals(beta_tilde)
	maxeig = max(np.absolute(eig))
	not_stationary = maxeig >= radius
	return beta * 0.95, not_stationary

def var_model(sparsity, p, sd_beta, sd_e, N, lag, seed = 543):
	np.random.seed(seed)

	radius = 0.97
	min_effect = 1
	beta = np.random.normal(loc = 0, scale = sd_beta, size = (p, p * lag))
	beta[(beta < min_effect) & (beta > 0)] = min_effect
	beta[(beta > - min_effect) & (beta < 0)] = - min_effect

	GC_on = np.random.binomial(n = 1, p = sparsity, size = (p, p))
	for i in range(p):
		beta[i, i] = min_effect
		GC_on[i, i] = 1

	GC_lag = GC_on
	for i in range(lag - 1):
		GC_lag = np.hstack((GC_lag, GC_on))

	beta = np.multiply(GC_lag, beta)
	errors = np.random.normal(loc = 0, scale = sd_e, size = (p, N))

	not_stationary = True
	while not_stationary:
		beta, not_stationary = stationary_var(beta, p, lag, radius)

	X = np.zeros((p, N))
	X[:, range(lag)] = errors[:, range(lag)]
	for i in range(lag, N):
		X[:, i] = np.dot(beta, X[:, range(i - lag, i)].flatten(order = 'F')) + errors[:, i]

	return X.T, beta, GC_on

def standardized_var_model(sparsity, p, beta_value, sd_e, N, lag, seed = 654):
	np.random.seed(seed)

	radius = 0.97
	beta = np.eye(p) * beta_value
	GC_on = np.eye(p)

	# Set dependencies for each component
	num_nonzero = int(p * sparsity) - 1
	for i in range(p):
		choice = np.random.choice(p - 1, size = num_nonzero, replace = False)
		choice[choice >= i] += 1
		beta[i, choice] = beta_value
		GC_on[i, choice] = 1

	# Create full beta matrix
	beta_full = beta
	for i in range(1, lag):
		beta_full = np.hstack((beta_full, beta))

	not_stationary = True
	while not_stationary:
		beta_full, not_stationary = stationary_var(beta_full, p, lag, radius)

	errors = np.random.normal(loc = 0, scale = sd_e, size = (p, N))
	X = np.zeros((p, N))
	X[:, range(lag)] = errors[:, range(lag)]
	for i in range(lag, N):
		X[:, i] = np.dot(beta_full, X[:, range(i - lag, i)].flatten(order = 'F')) + errors[:, i]

	return X.T, beta_full, GC_on

def long_lag_var_model(sparsity, p, sd_beta, sd_e, N, lag = 20, seed = 765, mixed = True):
	np.random.seed(seed)
	
	radius = 0.97
	min_effect = 1
	GC_on = np.random.binomial(n = 1, p = sparsity, size = (p, p))
	GC_lag = np.zeros((p, p * lag))
	if mixed:
		GC_lag[:, range(p * (lag - 1), p * lag)] = np.eye(p)
		GC_lag[:, range(p)] = GC_on
	else:
		GC_lag[:, range(p)] = np.maximum(GC_on, np.eye(p))

	# if mixed:
	# 	GC_lag[:, range(0, p)] = np.eye(p)
	# 	GC_lag[:, range(p * (lag - 1), p * lag)] = GC_on
	# else:
	# 	GC_lag[:, range(p * (lag - 1), p * lag)] = np.maximum(GC_on, np.eye(p))

	beta = np.random.normal(loc = 0, scale = sd_beta, size = (p, p * lag))
	beta[(beta < min_effect) & (beta > 0)] = min_effect
	beta[(beta > min_effect) & (beta < 0)] = - min_effect
	beta = np.multiply(beta, GC_lag)

	not_stationary = True
	while not_stationary:
		beta, not_stationary = stationary_var(beta, p, lag, radius)

	errors = np.random.normal(loc = 0, scale = sd_e, size = (p, N))

	X = np.zeros((p, N))
	X[:, range(lag)] = errors[:, range(lag)]
	for i in range(lag, N):
		X[:, i] = np.dot(beta, X[:, range(i - lag, i)].flatten(order = 'F')) + errors[:, i]

	return X.T, beta, np.maximum(GC_on, np.eye(p))

def standardized_long_lag_var_model(sparsity, p, beta_value, sd_e, N, lag = 20, seed = 765, mixed = True):
	np.random.seed(seed)
	
	radius = 0.97
	min_effect = 1

	# Determine dependencies
	GC_on = np.zeros((p, p))
	num_nonzero = int(p * sparsity) - 1
	for i in range(p):
		choice = np.random.choice(p - 1, size = num_nonzero, replace = False)
		choice[choice >= i] += 1
		GC_on[i, choice] = 1

	# Determine full beta
	GC_lag = np.zeros((p, p * lag))
	if mixed:
		GC_lag[:, range(p * (lag - 1), p * lag)] = np.eye(p)
		GC_lag[:, range(p)] = GC_on
	else:
		GC_lag[:, range(p)] = np.maximum(GC_on, np.eye(p))

	beta = beta_value * GC_lag

	not_stationary = True
	while not_stationary:
		beta, not_stationary = stationary_var(beta, p, lag, radius)

	errors = np.random.normal(loc = 0, scale = sd_e, size = (p, N))

	X = np.zeros((p, N))
	X[:, range(lag)] = errors[:, range(lag)]
	for i in range(lag, N):
		X[:, i] = np.dot(beta, X[:, range(i - lag, i)].flatten(order = 'F')) + errors[:, i]

	return X.T, beta, np.maximum(GC_on, np.eye(p))

def hmm_model(p, N, num_states = 3, sd_e = 0.1, sparsity = 0.2, tau = 2, seed = 321, standardized = True):
	np.random.seed(seed)

	# Determine dependencies
	if standardized:
		GC_on = np.zeros((p,p))
		con_per_comp = int(p * sparsity) - 1
		for i in range(p):
			possible_choices = np.setdiff1d(np.arange(p),i)
			selected = np.random.choice(possible_choices, con_per_comp, replace = False)
			GC_on[i, selected] = 1
			GC_on[i,i] = 1
	else:
		GC_on = np.maximum(np.random.binomial(n = 1, p = sparsity, size = (p, p)), np.eye(p))

	# Generate mean emission for each state
	Z_sig = 0.3
	Z = np.zeros((p, p, num_states, num_states))
	
	mu = np.random.uniform(low = -5.0, high = 5.0, size = (p, num_states))
	for i in range(p):
		for j in range(p):
			if GC_on[i,j]:
				Z[i,j,:,:] = Z_sig * np.random.randn(num_states,num_states)

    # Generate state sequence
	L = np.zeros((N,p)).astype(int)
	for t in range(1,N):
		for i in range(p):
			switch_prob = np.zeros(num_states)
			for j in range(p):
				switch_prob += Z[i,j,L[t-1,j],:]
			switch_prob = switch_prob * tau
			switch_prob = np.exp(switch_prob - logsumexp(switch_prob))
			L[t,i] = np.nonzero(np.random.multinomial(1, switch_prob))[0][0]

    # Generate emissions from state sequence
	X = np.zeros((N,p))
	for i in range(N):
		for j in range(p):
			X[i,j] = sd_e * np.random.randn(1) + mu[j,L[i,j]]

	return X, L, GC_on

if __name__ == "__main__": 
	import matplotlib.pyplot as plt
	#z,GC = lorentz_96_model_2(5,10,100,.1,.1)
	#plt.plot(z[:, 0], 'b', label='theta(t)')
	#plt.plot(z[:, 1], 'g', label='omega(t)')
	#plt.show()
	sparsity = 1
	p = 2

	z, GC = kuramoto_model(sparsity, p,N=25000,delta_t = .001, sd=.1, seed=23)
	plt.plot(z[:, 0,0], 'b', label='theta(t)')
	plt.plot(z[:, 0,1], 'g', label='omega(t)')
	plt.show()

	#X,L,GC_on = hmm_model(10, 200)


	GC_true = np.eye(10)
	GC_est = np.zeros((3,10,10))
	GC_est[0,:,:] = np.eye(10) + .05
	GC_est[1,:,:] = np.eye(10) + .05
	GC_est[2,:,:] = np.eye(10) + .05
	thresh = .1
	tp,fp,auc = compute_AUC(GC_true,GC_est,thresh,self_con=True)



