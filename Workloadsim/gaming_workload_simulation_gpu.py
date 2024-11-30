import numpy as np
import cupy as cp
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import psutil
import logging
from functools import partial

class GamingWorkloadSimulatorGPU:
    def __init__(self, resolution=(1920, 1080), num_threads=None, workload_intensity='medium'):
        self.resolution = resolution
        self.num_threads = num_threads or psutil.cpu_count(logical=True)
        self.workload_intensity = workload_intensity
        self.running = False
        self.frame_queue = queue.Queue()
        
        # Configure workload intensity
        self.intensities = {
            'light': {'matrix_size': 2048, 'particle_count': 10000, 'physics_iterations': 200},  # Increased for GPU
            'medium': {'matrix_size': 4096, 'particle_count': 50000, 'physics_iterations': 500},
            'heavy': {'matrix_size': 8192, 'particle_count': 100000, 'physics_iterations': 1000}
        }
        
        logging.basicConfig(
            filename='workload_simulation_gpu.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def gpu_render_frame(self):
        """Simulate GPU rendering workload"""
        size = self.intensities[self.workload_intensity]['matrix_size']
        
        try:
            # Create random matrices on GPU
            matrix1 = cp.random.random((size, size), dtype=cp.float32)
            matrix2 = cp.random.random((size, size), dtype=cp.float32)
            
            start_time = time.time()
            
            # Matrix multiplication on GPU
            result = cp.matmul(matrix1, matrix2)
            
            # Post-processing effects
            result = cp.power(result, 0.5)  # Gamma correction
            result = cp.clip(result, 0, 1)  # Color clamping
            
            # Ensure GPU operations are complete
            cp.cuda.Stream.null.synchronize()
            
            duration = time.time() - start_time
            logging.info(f"GPU Frame rendered in {duration:.4f} seconds")
            
            return cp.asnumpy(result)  # Convert back to CPU for compatibility
            
        except Exception as e:
            logging.error(f"GPU rendering error: {str(e)}")
            raise

    def gpu_physics_simulation(self):
        """GPU-accelerated physics simulation"""
        particle_count = self.intensities[self.workload_intensity]['particle_count']
        iterations = self.intensities[self.workload_intensity]['physics_iterations']
        
        # Initialize on GPU
        positions = cp.random.random((particle_count, 3), dtype=cp.float32)
        velocities = cp.random.random((particle_count, 3), dtype=cp.float32) * 0.1
        
        start_time = time.time()
        
        # Update physics on GPU
        for _ in range(iterations):
            positions += velocities
            
            # Boundary checking
            mask = positions > 1.0
            velocities *= cp.where(mask, -1, 1)
            positions = cp.clip(positions, 0, 1)
        
        cp.cuda.Stream.null.synchronize()
        duration = time.time() - start_time
        logging.info(f"GPU Physics simulation completed in {duration:.4f} seconds")
        
        return cp.asnumpy(positions), cp.asnumpy(velocities)

    def process_ai_region(self, region_data):
        """Keep AI on CPU as it's less parallelizable"""
        start_pos, grid = region_data
        visited = set()
        start_pos_tuple = tuple(map(int, start_pos))
        frontier = [(start_pos_tuple, 0)]
        
        while frontier and len(visited) < 2000:
            pos, cost = frontier.pop(0)
            if pos not in visited:
                visited.add(pos)
                x, y = pos
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_x < grid.shape[0] and 0 <= new_y < grid.shape[1] and 
                        grid[new_x, new_y] == 0):
                        frontier.append(((new_x, new_y), cost + 1))
        
        return len(visited)

    def cpu_ai_simulation(self):
        """Keep AI pathfinding on CPU"""
        start_time = time.time()
        grid_size = self.intensities[self.workload_intensity]['matrix_size']
        grid = np.random.choice([0, 1], size=(grid_size, grid_size), p=[0.7, 0.3])
        
        regions = [(np.array([np.random.randint(0, grid_size), 
                            np.random.randint(0, grid_size)]), grid) 
                  for _ in range(self.num_threads * 2)]
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.process_ai_region, region) 
                      for region in regions]
            results = [future.result() for future in as_completed(futures)]
        
        duration = time.time() - start_time
        logging.info(f"AI simulation completed in {duration:.4f} seconds")
        return results

    def run_frame(self):
        """Process one complete frame using GPU where beneficial"""
        frame_start_time = time.time()
        
        # Run GPU operations first
        frame = self.gpu_render_frame()
        physics_results = self.gpu_physics_simulation()
        
        # Run AI on CPU
        ai_results = self.cpu_ai_simulation()
        
        frame_time = time.time() - frame_start_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        
        logging.info(f"Frame completed - Time: {frame_time:.4f}s, FPS: {fps:.2f}")
        
        return {
            'frame_time': frame_time,
            'fps': fps,
            'frame_size': frame.shape,
            'physics_particles': len(physics_results[0]),
            'ai_paths': len(ai_results),
            'cpu_usage': psutil.cpu_percent(percpu=True),
            'gpu_memory': cp.get_default_memory_pool().used_bytes() / 1024**2  # MB
        }

    def start_simulation(self, duration_seconds=10):
        """Run the simulation for a specified duration"""
        self.running = True
        start_time = time.time()
        frame_count = 0
        
        logging.info(f"Starting GPU simulation - Resolution: {self.resolution}, "
                    f"Threads: {self.num_threads}, "
                    f"Intensity: {self.workload_intensity}")
        
        try:
            while self.running and (time.time() - start_time) < duration_seconds:
                frame_stats = self.run_frame()
                frame_count += 1
                
                cpu_usage = np.mean(frame_stats['cpu_usage'])
                print(f"\rFrame {frame_count}: "
                      f"FPS: {frame_stats['fps']:.2f} | "
                      f"Frame Time: {frame_stats['frame_time']*1000:.1f}ms | "
                      f"CPU Usage: {cpu_usage:.1f}% | "
                      f"GPU Memory: {frame_stats['gpu_memory']:.1f}MB | "
                      f"System Memory: {psutil.virtual_memory().percent}%", end="")
                
        except KeyboardInterrupt:
            logging.info("Simulation interrupted by user")
        finally:
            self.running = False
            print("\nSimulation completed")
            
        return frame_count

def main():
    simulator = GamingWorkloadSimulatorGPU(
        resolution=(1920, 1080),
        num_threads=None,
        workload_intensity='medium'
    )
    
    total_frames = simulator.start_simulation(duration_seconds=30)
    print(f"\nProcessed {total_frames} frames")

if __name__ == "__main__":
    main()
