import time

class Profiler:
    def __init__(self):
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        
    def stop(self, task_name="Task"):
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"[{task_name}] took {duration:.4f} seconds")
        else:
            print("Profiler not started.")
