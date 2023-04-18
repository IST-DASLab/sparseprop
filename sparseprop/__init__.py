import backend
import os

def set_num_threads(num_threads):
    os.environ['OMP_NUM_THREADS'] = str(num_threads)