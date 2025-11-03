"""
Real-time memory monitor untuk 1050 Ti
"""

import psutil
import time
import logging
from datetime import datetime

class MemoryMonitor1050Ti:
    """Monitor memory usage selama trading"""
    
    def __init__(self, alert_threshold=80):
        self.alert_threshold = alert_threshold
        self.logger = logging.getLogger("memory.monitor")
    
    def start_monitoring(self, interval=60):
        """Start background memory monitoring"""
        try:
            while True:
                self.check_memory()
                time.sleep(interval)
        except KeyboardInterrupt:
            self.logger.info("Memory monitoring stopped")
    
    def check_memory(self):
        """Check current memory usage"""
        # System RAM
        ram = psutil.virtual_memory()
        ram_usage = ram.percent
        
        # GPU memory (jika available)
        gpu_usage = "N/A"
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_usage = f"{gpu.memoryUtil * 100:.1f}%"
        except ImportError:
            pass
        
        status = f"Memory Usage - RAM: {ram_usage:.1f}%, GPU: {gpu_usage}"
        
        if ram_usage > self.alert_threshold:
            self.logger.warning(f"HIGH MEMORY USAGE: {status}")
        else:
            self.logger.debug(status)
        
        return {
            'timestamp': datetime.now(),
            'ram_usage': ram_usage,
            'gpu_usage': gpu_usage,
            'status': 'warning' if ram_usage > self.alert_threshold else 'normal'
        }