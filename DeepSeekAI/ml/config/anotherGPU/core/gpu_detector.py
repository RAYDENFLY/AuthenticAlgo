"""
Auto GPU Detection and Configuration Loader
Detects GPU and loads appropriate configuration
"""

import subprocess
import platform
import logging
from pathlib import Path

class GPUDetector:
    """Detect GPU and load optimal configuration"""
    
    def __init__(self):
        self.logger = logging.getLogger("gpu.detector")
        self.gpu_info = self._detect_gpu()
    
    def _detect_gpu(self) -> dict:
        """Detect GPU hardware"""
        system = platform.system()
        gpu_info = {}
        
        try:
            if system == "Windows":
                gpu_info = self._detect_windows_gpu()
            elif system == "Linux":
                gpu_info = self._detect_linux_gpu()
            elif system == "Darwin":  # macOS
                gpu_info = self._detect_macos_gpu()
            else:
                self.logger.warning(f"Unsupported system: {system}")
                
        except Exception as e:
            self.logger.error(f"GPU detection failed: {e}")
            
        return gpu_info
    
    def _detect_windows_gpu(self) -> dict:
        """Detect GPU on Windows using WMIC"""
        try:
            result = subprocess.run([
                'wmic', 'path', 'win32_VideoController', 'get', 'name'
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            gpu_names = [line.strip() for line in lines if line.strip()]
            
            return self._parse_gpu_names(gpu_names)
            
        except Exception as e:
            self.logger.warning(f"WMIC detection failed: {e}")
            return self._fallback_detection()
    
    def _detect_linux_gpu(self) -> dict:
        """Detect GPU on Linux using lspci"""
        try:
            result = subprocess.run([
                'lspci', '-v'
            ], capture_output=True, text=True, check=True)
            
            # Parse lspci output for GPU information
            gpu_info = self._parse_lspci_output(result.stdout)
            return gpu_info
            
        except Exception as e:
            self.logger.warning(f"lspci detection failed: {e}")
            return self._fallback_detection()
    
    def _detect_macos_gpu(self) -> dict:
        """Detect GPU on macOS"""
        try:
            result = subprocess.run([
                'system_profiler', 'SPDisplaysDataType'
            ], capture_output=True, text=True, check=True)
            
            return self._parse_macos_gpu_output(result.stdout)
            
        except Exception as e:
            self.logger.warning(f"macOS GPU detection failed: {e}")
            return self._fallback_detection()
    
    def _parse_gpu_names(self, gpu_names: list) -> dict:
        """Parse GPU names to determine model and VRAM"""
        gpu_info = {}
        
        for name in gpu_names:
            name_lower = name.lower()
            
            # NVIDIA GPUs
            if 'nvidia' in name_lower or 'geforce' in name_lower:
                gpu_info['vendor'] = 'nvidia'
                
                if '1660' in name_lower:
                    gpu_info['model'] = 'gtx_1660'
                    gpu_info['vram_mb'] = 6144  # 6GB
                elif '2060' in name_lower and 'super' in name_lower:
                    gpu_info['model'] = 'rtx_2060_super'
                    gpu_info['vram_mb'] = 8192  # 8GB
                elif '3050' in name_lower:
                    gpu_info['model'] = 'rtx_3050'
                    gpu_info['vram_mb'] = 8192 if '8gb' in name_lower else 4096
                elif '3060' in name_lower:
                    gpu_info['model'] = 'rtx_3060'
                    gpu_info['vram_mb'] = 12288  # 12GB
                elif '3070' in name_lower:
                    gpu_info['model'] = 'rtx_3070'
                    gpu_info['vram_mb'] = 8192
                elif '3080' in name_lower:
                    gpu_info['model'] = 'rtx_3080'
                    gpu_info['vram_mb'] = 10240  # 10GB
                elif '3090' in name_lower:
                    gpu_info['model'] = 'rtx_3090'
                    gpu_info['vram_mb'] = 24576  # 24GB
                elif '4050' in name_lower:
                    gpu_info['model'] = 'rtx_4050'
                    gpu_info['vram_mb'] = 6144
                elif '4060' in name_lower:
                    gpu_info['model'] = 'rtx_4060'
                    gpu_info['vram_mb'] = 8192
                elif '4070' in name_lower:
                    gpu_info['model'] = 'rtx_4070'
                    gpu_info['vram_mb'] = 12288
                elif '4080' in name_lower:
                    gpu_info['model'] = 'rtx_4080'
                    gpu_info['vram_mb'] = 16384
                elif '4090' in name_lower:
                    gpu_info['model'] = 'rtx_4090'
                    gpu_info['vram_mb'] = 24576
            
            # AMD GPUs
            elif 'amd' in name_lower or 'radeon' in name_lower:
                gpu_info['vendor'] = 'amd'
                
                if '6600' in name_lower:
                    gpu_info['model'] = 'rx_6600'
                    gpu_info['vram_mb'] = 8192  # 8GB
                elif '6700' in name_lower:
                    gpu_info['model'] = 'rx_6700'
                    gpu_info['vram_mb'] = 12288  # 12GB
                elif '6800' in name_lower:
                    gpu_info['model'] = 'rx_6800'
                    gpu_info['vram_mb'] = 16384  # 16GB
                elif '6900' in name_lower:
                    gpu_info['model'] = 'rx_6900'
                    gpu_info['vram_mb'] = 16384
                elif '7700' in name_lower:
                    gpu_info['model'] = 'rx_7700'
                    gpu_info['vram_mb'] = 12288
                elif '7800' in name_lower:
                    gpu_info['model'] = 'rx_7800'
                    gpu_info['vram_mb'] = 16384
                elif '7900' in name_lower:
                    gpu_info['model'] = 'rx_7900'
                    gpu_info['vram_mb'] = 24576  # 24GB
            
            # Intel GPUs
            elif 'intel' in name_lower:
                gpu_info['vendor'] = 'intel'
                gpu_info['vram_mb'] = 1024  # Conservative estimate
        
        # If no specific model detected, use fallback
        if 'model' not in gpu_info:
            gpu_info.update(self._fallback_detection())
            
        return gpu_info
    
    def _parse_lspci_output(self, output: str) -> dict:
        """Parse lspci output for GPU info"""
        # Simplified parsing - in practice you'd want more robust parsing
        lines = output.split('\n')
        gpu_info = {}
        
        for line in lines:
            if 'VGA compatible controller' in line or '3D controller' in line:
                if 'NVIDIA' in line:
                    gpu_info['vendor'] = 'nvidia'
                elif 'AMD' in line or 'ATI' in line:
                    gpu_info['vendor'] = 'amd'
                elif 'Intel' in line:
                    gpu_info['vendor'] = 'intel'
        
        return gpu_info if gpu_info else self._fallback_detection()
    
    def _parse_macos_gpu_output(self, output: str) -> dict:
        """Parse macOS system_profiler output"""
        gpu_info = {}
        lines = output.split('\n')
        
        for line in lines:
            if 'Chipset Model' in line:
                if 'AMD' in line:
                    gpu_info['vendor'] = 'amd'
                elif 'Intel' in line:
                    gpu_info['vendor'] = 'intel'
                elif 'Apple' in line:
                    gpu_info['vendor'] = 'apple'
        
        return gpu_info if gpu_info else self._fallback_detection()
    
    def _fallback_detection(self) -> dict:
        """Fallback GPU detection using TensorFlow"""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            
            if gpus:
                # Try to get GPU info
                gpu_info = {
                    'vendor': 'nvidia',  # Assume NVIDIA if TF detects GPU
                    'model': 'unknown',
                    'vram_mb': 4096,  # Conservative default
                    'detection_method': 'tensorflow_fallback'
                }
                return gpu_info
                
        except ImportError:
            pass
        
        # Ultimate fallback
        return {
            'vendor': 'unknown',
            'model': 'unknown',
            'vram_mb': 2048,  # Very conservative
            'detection_method': 'ultimate_fallback'
        }
    
    def get_recommended_config(self) -> str:
        """Get recommended configuration file based on GPU detection"""
        if not self.gpu_info:
            return "config/gtx_1050ti_config.yaml"  # Most conservative
        
        model = self.gpu_info.get('model', 'unknown')
        vram_mb = self.gpu_info.get('vram_mb', 0)
        
        # Map GPU models to config files
        config_mapping = {
            'gtx_1660': 'config/gtx_1660_6gb.yaml',
            'rtx_2060_super': 'config/rtx_2060s_8gb.yaml',
            'rx_6600': 'config/rx_6600_8gb.yaml',
            'rtx_3050': 'config/rtx_2060s_8gb.yaml',  # Similar to 2060S
            'rtx_3060': 'config/rtx_2060s_8gb.yaml',  # Conservative for 12GB
            'rtx_3070': 'config/rtx_2060s_8gb.yaml',
            'rtx_4060': 'config/rtx_2060s_8gb.yaml',
        }
        
        # VRAM-based fallback
        if model in config_mapping:
            config_file = config_mapping[model]
        elif vram_mb >= 8192:  # 8GB+
            config_file = 'config/rtx_2060s_8gb.yaml'
        elif vram_mb >= 6144:  # 6GB+
            config_file = 'config/gtx_1660_6gb.yaml'
        else:  # Less than 6GB
            config_file = 'config/gtx_1050ti_config.yaml'
        
        self.logger.info(f"Detected GPU: {model}, VRAM: {vram_mb}MB, Recommended config: {config_file}")
        return config_file
    
    def print_gpu_info(self):
        """Print detected GPU information"""
        if self.gpu_info:
            print("=== GPU DETECTION RESULTS ===")
            for key, value in self.gpu_info.items():
                print(f"{key}: {value}")
            print(f"Recommended config: {self.get_recommended_config()}")
        else:
            print("No GPU information detected")