class OptimizedModelTrainer(ModelTrainer):
    """Model trainer yang dioptimalkan untuk GTX 1050 Ti"""
    
    def _train_lstm_optimized(self, X_train, X_test, y_train, y_test):
        """LSTM training yang dioptimalkan untuk GPU 4GB"""
        sequence_length = 20  # Reduced from 50
        batch_size = 16       # Reduced from 32
        
        X_train_seq, y_train_seq = self._create_sequences(X_train.values, y_train.values, sequence_length)
        X_test_seq, y_test_seq = self._create_sequences(X_test.values, y_test.values, sequence_length)
        
        # Build lightweight model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True, 
                               input_shape=(sequence_length, X_train.shape[1])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(16, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Use smaller batch size and fewer epochs
        history = model.fit(
            X_train_seq, y_train_seq,
            batch_size=batch_size,
            epochs=50,  # Reduced from 100
            validation_data=(X_test_seq, y_test_seq),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
            ],
            verbose=1
        )
        
        return model, self._evaluate_lstm_model(model, X_train_seq, X_test_seq, y_train_seq, y_test_seq, history)