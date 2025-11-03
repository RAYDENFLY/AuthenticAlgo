OPTIMIZED_FEATURES = {
    'lookback_periods': [10, 20],    # Kurangi dari [5, 10, 20, 50]
    'rolling_windows': [10, 20],     # Kurangi window sizes
    'max_lag_features': 3,           # Kurangi lag features
    'feature_selection': True,       # Penting! Pilih feature terbaik
    'n_features': 30,                # Batasi total features
}