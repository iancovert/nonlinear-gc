from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import rpy2


def run_bigvar(Y, p, struct, nlambdas=10, lamratio=10., T1=0, T2=None,
               use_intercept=True):
    """ Run BigVAR model on grid of lambda values.

        Y : T x d ndarray
            Observed time series

        p : int
            maximum lag

        struct : string
            structured penalty to use (see R docs for complete list):
                 - "HVARC"    (componentwise HVAR)
                 - "HVAROO"   (own/other HVAR)
                 - "HVARELEM" (elementwise HVAR)

        nlambdas : int
            number of lambda values to run

        lamratio : float
            fraction of the largest lambda that the smallest should be

        T1 : int
            Y[T1:T2] is the training set

        T2 : int
            Y[T2:] is the validation set. There must be at least one time step
            in the validation set

        use_intercept : bool
            whether to use an intercept tern

        Returns:

        coefs : d x d*p x nlambdas, ndarray
            coefficient array
        lambdas : nlambda ndarray
            lambda values
        intercept : d ndarray
            if use_intercept is True the intercept term, else None


        Notes:
        
        - The maximum lambda value in the grid is the smallest value
          that produces all zero coefficient estimates. The smallest values is
          then given as lambda_max / lamratio

    """

    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()

    if not rpackages.isinstalled('BigVAR'):
        raise RuntimeError("Must install BigVAR package in R",
                           "See: https://github.com/wbnicholson/BigVAR")

    # recast T1 as R-index
    T1 = T1+1
    if T2 is None:
        T2 = Y.shape[0] - 2

    if T2 > Y.shape[0] - 2:
        raise ValueError("T2 must leave at least one entry at the end of the series")

    rbigvar = rpackages.importr('BigVAR')

    gran = robjects.FloatVector([nlambdas, lamratio])
    model = rbigvar.constructModel(Y, p, struct, gran,
                                   T1=T1, T2=T2, intercept=use_intercept)

    model_res = rbigvar.BigVAR_est(model)

    intercept = None
    coefs = np.array(model_res[0])
    if use_intercept:
        intercept = np.squeeze(coefs[:, 0, :])
        coefs = coefs[:, 1:, :]
    lambdas = np.array(model_res[1])

    return coefs, lambdas, intercept
