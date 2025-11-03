"""
GPU Memory Manager for GTX 1050 Ti 4GB
Optimized memory management for limited GPU
"""

import tensorflow as tf
import logging
import psutil
import os

class GPUManager1050Ti:
    """GPU Memory Manager khusus untuk GTX 1050 Ti 4GB"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("gpu.manager")
        self.setup_gpu_memory()
    
    def setup_gpu_memory(self):
        """Setup GPU memory dengan optimisasi untuk 1050 Ti"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if not gpus:
            self.logger.warning("No GPU found, using CPU only")
            return False
        
        try:
            # Enable memory growth - PENTING untuk 1050 Ti
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit (3GB dari 4GB)
            memory_limit = self.config.get('gpu_settings', {}).get('memory_limit_mb', 3072)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
            )
            
            self.logger.info(f"GPU memory optimized: {memory_limit}MB limit, memory growth enabled")
            return True
            
        except RuntimeError as e:
            self.logger.error(f"GPU memory setup failed: {e}")
            return False
    
    def check_memory_safe(self) -> tuple:
        """Check jika memory dalam kondisi aman untuk training"""
        # Check system RAM
        ram = psutil.virtual_memory()
        ram_safe = ram.percent < 85
        
        # Check GPU memory jika available
        gpu_safe = True
        try:
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            gpu_usage = gpu_memory['current'] / gpu_memory['limit']
            gpu_safe = gpu_usage < 0.8  # 80% usage
        except:
            pass
        
        is_safe = ram_safe and gpu_safe
        status = {
            'ram_used_percent': ram.percent,
            'ram_safe': ram_safe,
            'gpu_safe': gpu_safe,
            'overall_safe': is_safe
        }
        
        return is_safe, status
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        try:
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            self.logger.debug("GPU memory cleared")
        except Exception as e:
            self.logger.warning(f"Error clearing GPU memory: {e}")