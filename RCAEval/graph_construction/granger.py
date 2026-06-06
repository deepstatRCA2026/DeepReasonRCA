import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import InfeasibleTestError
def granger(
    data,
    maxlag=3,
    p_val_threshold=0.05,
    test="ssr_ftest",
    min_obs_factor=5,   # NEW: safety factor
):
    assert test in ['ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']

    node_names = data.columns.to_list()
    n = len(node_names)
    adj = np.zeros((n, n))

    # safety: ensure enough observations
    if len(data) < min_obs_factor * maxlag:
        return adj  # empty graph is correct behavior

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            try:
                # test j -> i
                results = grangercausalitytests(
                    data[[node_names[i], node_names[j]]],
                    maxlag=maxlag,
                    verbose=False,
                )

                for lag, res in results.items():
                    p_val = res[0][test][1]
                    if p_val < p_val_threshold:
                        adj[i, j] = 1
                        break

            except InfeasibleTestError:
                # perfect fit → no evidence of causality
                continue

            except (np.linalg.LinAlgError, ValueError):
                # singular matrix / constant series
                continue

    return adj