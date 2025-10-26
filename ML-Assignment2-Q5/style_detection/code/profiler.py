"""Simple profiler for timing code execution"""
import time

class Profiler:
    """Context manager for profiling code blocks"""
    
    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if self.name:
            print(f"[Profiler] {self.name}: {elapsed:.4f} seconds")
        return False
    
    def elapsed(self):
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None
