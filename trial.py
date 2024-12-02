param_grid = {'C': [1, 10, 100, 1000], 'kernel': ['linear']}
param_comb = [(c, k) for c in param_grid['C'] for k in param_grid['kernel']]

print(param_comb)
