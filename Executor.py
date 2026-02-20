# executor.py

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Leave 1 CPU core for FastAPI / system processes
CPU_COUNT = multiprocessing.cpu_count()

MAX_WORKERS = max(1, CPU_COUNT - 1)

# Global ProcessPoolExecutor
# This will be shared by the FastAPI app
executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
