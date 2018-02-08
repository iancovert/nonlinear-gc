import numpy as np

def rectify_roc(fpr_vals, tpr_vals):
    fpr_inds = np.argsort(fpr_vals)
    fpr_sort = fpr_vals[fpr_inds]
    tpr_sort = tpr_vals[fpr_inds]
    
    fpr_cvx = [fpr_sort[0]]
    tpr_cvx = [tpr_sort[0]]
    i = 0
    while i < len(fpr_sort):
        dy = tpr_sort[i+1:] - tpr_cvx[-1]
        next_idx = np.where(dy > 0)[0]
        if len(next_idx) > 0:
            next_idx = next_idx[0] + 1
            i += next_idx
            fpr_cvx.append(fpr_sort[i])
            tpr_cvx.append(tpr_sort[i])
        else:
            break

    # if the final point isn't (1,1), then append it (for plotting)
    if not (np.allclose(fpr_cvx[-1], 1.) and np.allclose(tpr_cvx[-1], 1.)):
        fpr_cvx.append(1.)
        tpr_cvx.append(1.)
    # same for left point being (0,0)
    if not (np.allclose(fpr_cvx[0], 0.) and np.allclose(tpr_cvx[0], 0.)):
        fpr_cvx.insert(0, 0.)
        tpr_cvx.insert(0, 0.)

    return np.array(fpr_cvx), np.array(tpr_cvx)

"""
input:
GC_true - pxp binary matrix 
GC_est - list of pxp floating point matrices
thresh - double, value to threshold the GC_est to determine connections
self_con - boolean, to count self connections or not

output:
FP - length m False positive rate vector 
TP - length m True positive rate vector
auc - area under the ROC curve
"""

def compute_AUC(GC_true, GC_list, thresh, self_con = True, rectify = False):
	m = len(GC_list)
	p = GC_list[0].shape[0]
	thresh_list = []

	for i in range(m):
		thresh_grid = np.zeros((p, p)).astype(int)
		thresh_grid[GC_list[i] > thresh] = 1
		thresh_list.append(thresh_grid)
	
	if not self_con:
		GC_true = np.maximum(GC_true, 2 * np.eye(p))
		for i in range(m):
			thresh_list[i] = np.maximum(thresh_list[i], 2 * np.eye(p))

	FP_rate = np.zeros(m)
	TP_rate = np.zeros(m)

	N = np.sum(GC_true == 0)
	P = np.sum(GC_true == 1)

	for i in range(m):
		FP = np.sum((thresh_list[i] == 1) * (GC_true == 0))
		FP_rate[i] = FP/N

		TP = np.sum((thresh_list[i] == 1) * (GC_true == 1))
		TP_rate[i] = TP/P
	
	FP_rate = np.array([1] + list(FP_rate) + [0])
	TP_rate = np.array([1] + list(TP_rate) + [0])

	if rectify:
		FP_rate, TP_rate = rectify_roc(FP_rate, TP_rate)

	# else:
	# 	inds = np.argsort(FP_rate)
	# 	FP_rate = FP_rate[inds]
	# 	TP_rate = TP_rate[inds]

	return TP_rate, FP_rate, np.trapz(TP_rate[::-1],FP_rate[::-1])

def compute_AUPR(GC_true, GC_list, thresh, self_con = True, rectify = False):
	m = len(GC_list)
	p = GC_list[0].shape[0]
	thresh_list = []

	for i in range(m):
		thresh_grid = np.zeros((p, p)).astype(int)
		thresh_grid[GC_list[i] > thresh] = 1
		thresh_list.append(thresh_grid)
	
	if not self_con:
		GC_true = np.maximum(GC_true, 2 * np.eye(p))
		for i in range(m):
			thresh_list[i] = np.maximum(thresh_list[i], 2 * np.eye(p))

	PR_rate = np.zeros(m)
	RE_rate = np.zeros(m)

	P = np.sum(GC_true == 1)

	for i in range(m):
		TP = np.sum((thresh_list[i] == 1) * (GC_true == 1))
		FP = np.sum((thresh_list[i] == 1) * (GC_true == 0))

		if TP + FP == 0:
			PR_rate[i] = 1
		else:
			PR_rate[i] = TP / (TP + FP)
		RE_rate[i] = TP / P

	PR_rate = np.array([0] + list(PR_rate) + [1])
	RE_rate = np.array([1] + list(RE_rate) + [0])

	if rectify:
		RE_rate, PR_rate = rectify_roc(RE_rate, PR_rate)
		RE_rate = RE_rate
		PR_rate = PR_rate

	# else:
	# 	inds = np.argsort(RE_rate)
	# 	RE_rate = RE_rate[inds]
	# 	PR_rate = PR_rate[inds]

	return PR_rate, RE_rate, np.trapz(PR_rate[::-1],RE_rate[::-1])
