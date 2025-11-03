import tensorflow as tf
from tensorflow.keras import mixed_precision

class GPUOptimizedLSTM:
    def __init__(self, config):
        self.config = config
        self.setup_gpu_memory()
    
    def setup_gpu_memory(self):
        """Setup GPU memory untuk 1050 Ti 4GB"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Batasi memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Limit GPU memory usage (2GB untuk 1050 Ti)
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
                )
                print("GPU memory optimized for GTX 1050 Ti")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
    
    def create_lightweight_lstm(self, input_shape):
        """Buat LSTM model yang ringan"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True, input_shape=input_shape),
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
        
        return model