from ml import MLModule

config_1050ti = {
    'feature_engineering': {
        'n_features': 30,
        'feature_selection': True
    },
    'model_training': {
        'model_types': ['xgb', 'rf'],  # Skip LSTM dulu
        'test_size': 0.2
    },
    'prediction': {
        'model_types': ['xgb', 'rf'],
        'use_ensemble': True
    }
}

ml_module = MLModule(config_1050ti)