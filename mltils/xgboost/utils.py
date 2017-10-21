
xgb_to_sklearn = {
    'eta': 'learning_rate',
    'num_boost_round': 'n_estimators',
    'alpha': 'reg_alpha',
    'lambda': 'reg_lambda',
    'seed': 'random_state',
}

def to_sklearn_api(params):
    return {
        xgb_to_sklearn.get(key, key): value
        for key, value in params.items()
    }
