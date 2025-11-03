import psutil
import GPUtil

class MemoryMonitor:
    """Monitor memory usage selama training"""
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage"""
        gpus = GPUtil.getGPUs()
        memory_info = {
            'ram_used': psutil.virtual_memory().percent,
            'ram_available': psutil.virtual_memory().available / (1024**3),  # GB
        }
        
        if gpus:
            gpu = gpus[0]
            memory_info.update({
                'gpu_used': gpu.memoryUsed,
                'gpu_total': gpu.memoryTotal,
                'gpu_utilization': gpu.load * 100
            })
        
        return memory_info
    
    @staticmethod
    def check_memory_safe():
        """Check jika memory aman untuk training"""
        memory = MemoryMonitor.get_memory_usage()
        
        # Safety thresholds untuk 1050 Ti
        safe_conditions = [
            memory.get('ram_used', 0) < 85,
            memory.get('gpu_used', 0) < 3500,  # 3.5GB dari 4GB
            memory.get('gpu_utilization', 0) < 90
        ]
        
        return all(safe_conditions), memory