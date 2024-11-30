import sys
import time
import psutil
import os
from collections import deque
import subprocess

class CPUMonitor:
    def __init__(self, width=50, height=5):
        self.width = width
        self.height = height
        self.cpu_util_history = deque([0] * width, maxlen=width)
        self.cpu_freq_history = deque([0] * width, maxlen=width)
        
        # ANSI escape codes
        self.CYAN = '\033[36m'
        self.BLUE = '\033[34m'
        self.RESET = '\033[0m'
        self.CLEAR_LINE = '\033[K'
        self.MOVE_UP = '\033[A'

    def get_cpu_stats(self):
        try:
            cpu_freq = psutil.cpu_freq()
            return {
                'utilization': psutil.cpu_percent(),
                'frequency': cpu_freq.current if cpu_freq else 0,
                'core_count': psutil.cpu_count(logical=False),
                'thread_count': psutil.cpu_count(logical=True)
            }
        except:
            return {
                'utilization': 0,
                'frequency': 0,
                'core_count': 0,
                'thread_count': 0
            }
    
    def update_data(self):
        stats = self.get_cpu_stats()
        self.cpu_util_history.append(stats['utilization'])
        freq_percent = (stats['frequency'] / 4000) * 100
        self.cpu_freq_history.append(freq_percent)
        self.current_stats = stats
    
    def draw_graph(self, values, color, label, max_val=100):
        current = values[-1]
        sys.stdout.write(f'{color}{label}: {current:>5.1f}%{self.RESET}{self.CLEAR_LINE}\n')
        
        # Characters for different intensities
        chars = ' ▁█'  # Space for 0, thin line for very low values, full block for normal
        
        for y in range(self.height, -1, -1):
            line = f'{color}{y*max_val//self.height:3d}|'
            threshold = y * max_val / self.height
            
            for value in values:
                if value >= threshold:
                    if value < 1.0:  # For very low values
                        line += '▁'
                    else:
                        line += '█'
                else:
                    line += ' '
            
            line += f'{self.RESET}{self.CLEAR_LINE}\n'
            sys.stdout.write(line)
        
        sys.stdout.write(f'{color}   +{"-" * self.width}{self.RESET}{self.CLEAR_LINE}\n')
    
    def draw_cpu_stats(self):
        stats = self.current_stats
        sys.stdout.write(f'{self.CYAN}CPU Statistics:{self.RESET}{self.CLEAR_LINE}\n')
        sys.stdout.write(f'Frequency: {stats["frequency"]:>4.0f}MHz | Cores: {stats["core_count"]}/{stats["thread_count"]}{self.CLEAR_LINE}\n')
    
    def clear_previous_output(self, total_lines):
        for _ in range(total_lines):
            sys.stdout.write(self.MOVE_UP)
    
    def draw(self):
        total_lines = (self.height + 3) * 2 + 2  # Two graphs + stats (2 lines)
        
        if hasattr(self, '_drawn'):
            self.clear_previous_output(total_lines)
        self._drawn = True
        
        self.draw_graph(self.cpu_util_history, self.CYAN, 'CPU Usage')
        self.draw_graph(self.cpu_freq_history, self.BLUE, 'CPU Freq')
        self.draw_cpu_stats()
        
        sys.stdout.flush()

def main():
    try:
        monitor = CPUMonitor()
        while True:
            monitor.update_data()
            monitor.draw()
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n' * (monitor.height * 2 + 4))
        sys.stdout.write('\033[?25h')
        sys.stdout.flush()

if __name__ == '__main__':
    sys.stdout.write('\033[?25l')
    main()
