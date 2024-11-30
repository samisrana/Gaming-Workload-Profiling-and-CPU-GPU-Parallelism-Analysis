import numpy as np
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import psutil
import logging
from functools import partial

class GamingWorkloadSimulator:
    def __init__(self, resolution=(1920, 1080), num_threads=None, workload_intensity='medium'):
        self.resolution = resolution
        self.num_threads = num_threads or psutil.cpu_count(logical=True)
        self.workload_intensity = workload_intensity
        self.running = False
        self.frame_queue = queue.Queue()
        
        self.intensities = {
            'light': {'matrix_size': 512, 'particle_count': 5000, 'physics_iterations': 200},
            'medium': {'matrix_size': 1024, 'particle_count': 10000, 'physics_iterations': 500},
            'heavy': {'matrix_size': 2048, 'particle_count': 20000, 'physics_iterations': 1000}
        }
        
        logging.basicConfig(
            filename='workload_simulation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def simulate_render_frame_chunk(self, chunk_data):
        start_idx, end_idx, size = chunk_data
        chunk_result = np.random.random((end_idx - start_idx, size)).astype(np.float32)
        return chunk_result

    def simulate_render_frame(self):
        size = self.intensities[self.workload_intensity]['matrix_size']
        chunk_size = max(1, size // self.num_threads)
        
        try:
            start_time = time.time()
            chunks = [(i, min(i + chunk_size, size), size) 
                     for i in range(0, size, chunk_size)]
            
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                chunk_futures = [executor.submit(self.simulate_render_frame_chunk, chunk) 
                               for chunk in chunks]
                results = [future.result() for future in as_completed(chunk_futures)]
            
            result = np.vstack(results)
            result = np.matmul(result, result.T)
            result = np.power(result, 0.5)
            result = np.clip(result, 0, 1)
            
            duration = time.time() - start_time
            logging.info(f"Frame rendered in {duration:.4f} seconds")
            return result
            
        except Exception as e:
            logging.error(f"Rendering error: {str(e)}")
            raise

    def process_particle_chunk(self, particle_chunk, iterations):
        positions, velocities = particle_chunk
        
        for _ in range(iterations):
            positions += velocities
            mask = positions > 1.0
            velocities[mask] *= -1
            positions[mask] = 1.0
            mask = positions < 0.0
            velocities[mask] *= -1
            positions[mask] = 0.0
            
        return positions, velocities

    def cpu_physics_simulation(self):
        particle_count = self.intensities[self.workload_intensity]['particle_count']
        iterations = self.intensities[self.workload_intensity]['physics_iterations']
        
        positions = np.random.random((particle_count, 3))
        velocities = np.random.random((particle_count, 3)) * 0.1
        
        start_time = time.time()
        chunk_size = max(1, particle_count // self.num_threads)
        chunks = []
        
        for i in range(0, particle_count, chunk_size):
            end_idx = min(i + chunk_size, particle_count)
            chunks.append((positions[i:end_idx], velocities[i:end_idx]))
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.process_particle_chunk, chunk, iterations) 
                      for chunk in chunks]
            results = [future.result() for future in as_completed(futures)]
        
        final_positions = np.vstack([r[0] for r in results])
        final_velocities = np.vstack([r[1] for r in results])
        
        duration = time.time() - start_time
        logging.info(f"Physics simulation completed in {duration:.4f} seconds")
        return final_positions, final_velocities

    def process_ai_region(self, region_data):
        start_pos, grid = region_data
        visited = set()
        # Convert numpy array to tuple for hashing
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
        frame_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            render_future = executor.submit(self.simulate_render_frame)
            physics_future = executor.submit(self.cpu_physics_simulation)
            ai_future = executor.submit(self.cpu_ai_simulation)
            
            frame = render_future.result()
            physics_results = physics_future.result()
            ai_results = ai_future.result()
        
        frame_time = time.time() - frame_start_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        
        logging.info(f"Frame completed - Time: {frame_time:.4f}s, FPS: {fps:.2f}")
        
        return {
            'frame_time': frame_time,
            'fps': fps,
            'frame_size': frame.shape,
            'physics_particles': len(physics_results[0]),
            'ai_paths': len(ai_results),
            'cpu_usage': psutil.cpu_percent(percpu=True)
        }

    def start_simulation(self, duration_seconds=10):
        self.running = True
        start_time = time.time()
        frame_count = 0
        
        logging.info(f"Starting simulation - Resolution: {self.resolution}, "
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
                      f"Memory: {psutil.virtual_memory().percent}%", end="")
                
        except KeyboardInterrupt:
            logging.info("Simulation interrupted by user")
        finally:
            self.running = False
            print("\nSimulation completed")
            
        return frame_count

def main():
    simulator = GamingWorkloadSimulator(
        resolution=(1920, 1080),
        num_threads=None,
        workload_intensity='medium'
    )
    
    total_frames = simulator.start_simulation(duration_seconds=30)
    print(f"\nProcessed {total_frames} frames")

if __name__ == "__main__":
    main()
