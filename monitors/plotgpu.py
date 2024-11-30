import sys
import time
import psutil
import pynvml
from collections import deque

class TransparentGraphs:
    def __init__(self, width=50, height=7):
        self.width = width
        self.height = height
        self.gpu_util_history = deque([0] * width, maxlen=width)
        self.gpu_mem_history = deque([0] * width, maxlen=width)
        self.gpu_temp_history = deque([0] * width, maxlen=width)
        
        # Initialize NVIDIA Management Library
        pynvml.nvmlInit()
        try:
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            print("No NVIDIA GPU found!")
            sys.exit(1)
        
        # ANSI escape codes
        self.BLUE = '\033[34m'
        self.YELLOW = '\033[33m'
        self.RED = '\033[31m'
        self.RESET = '\033[0m'
        self.CLEAR_LINE = '\033[K'
        self.MOVE_UP = '\033[A'
        self.CLEAR_SCREEN = '\033[2J'
        self.CURSOR_HOME = '\033[H'
        
    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass
    
    def get_gpu_stats(self):
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            temperature = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
            clock = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(self.gpu_handle, pynvml.NVML_CLOCK_MEM)
            
            return {
                'utilization': utilization.gpu,
                'memory_used': (memory.used / memory.total) * 100,
                'temperature': temperature,
                'power': power,
                'clock': clock,
                'mem_clock': mem_clock
            }
        except:
            return {
                'utilization': 0,
                'memory_used': 0,
                'temperature': 0,
                'power': 0,
                'clock': 0,
                'mem_clock': 0
            }
    
    def update_data(self):
        gpu_stats = self.get_gpu_stats()
        self.gpu_util_history.append(gpu_stats['utilization'])
        self.gpu_mem_history.append(gpu_stats['memory_used'])
        self.gpu_temp_history.append(gpu_stats['temperature'])
        self.current_stats = gpu_stats
    
    def draw_graph(self, values, color, label, max_val=100):
        # Draw label and current value
        current = values[-1]
        sys.stdout.write(f'{color}{label}: {current:>5.1f}%{self.RESET}{self.CLEAR_LINE}\n')
        
        # Draw y-axis and graph
        for y in range(self.height, -1, -1):
            line = f'{color}{y*max_val//self.height:3d}|'
            threshold = y * max_val / self.height
            
            for value in values:
                if value >= threshold:
                    line += '█'
                else:
                    line += ' '
            
            line += f'{self.RESET}{self.CLEAR_LINE}\n'
            sys.stdout.write(line)
        
        # Draw x-axis
        sys.stdout.write(f'{color}   +{"-" * self.width}{self.RESET}{self.CLEAR_LINE}\n')
    
    def draw_nvidia_stats(self):
        stats = self.current_stats
        sys.stdout.write(f'{self.YELLOW}NVIDIA GPU Statistics:{self.RESET}{self.CLEAR_LINE}\n')
        sys.stdout.write(f'Power: {stats["power"]:>6.2f}W | Core: {stats["clock"]:>4d}MHz | Memory: {stats["mem_clock"]:>4d}MHz{self.CLEAR_LINE}\n')
    
    def draw(self):
        # Calculate total lines for cursor movement
        total_lines = (self.height + 3) * 3 + 3  # Three graphs + stats lines
        
        if hasattr(self, '_drawn'):
            # Clear the screen and move cursor to home position
            sys.stdout.write(self.CLEAR_SCREEN + self.CURSOR_HOME)
        self._drawn = True
        
        # Draw all components
        self.draw_graph(self.gpu_util_history, self.BLUE, 'GPU Usage')
        self.draw_graph(self.gpu_mem_history, self.YELLOW, 'GPU Memory')
        self.draw_graph(self.gpu_temp_history, self.RED, 'GPU Temp', max_val=100)
        self.draw_nvidia_stats()
        
        sys.stdout.flush()

def main():
    try:
        graphs = TransparentGraphs()
        while True:
            graphs.update_data()
            graphs.draw()
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n' * (graphs.height * 3 + 8))  # Clear space after exit
        sys.stdout.write('\033[?25h')  # Show cursor
        sys.stdout.flush()

if __name__ == '__main__':
    sys.stdout.write('\033[?25l')  # Hide cursor
    main()
