"""
Simple profiling module that emulates Matlab stopwatch timer

>>> import common.profiler as prof
>>> import time
>>> prof.tic()
>>> time.sleep(1)
>>> assert(prof.toc() > 0)
"""
from time import time


_last_time = time()


def tic():
    global _last_time
    _last_time = time()


def toc():
    global _last_time
    curr_time = time()
    elapsed_time = curr_time - _last_time
    _last_time = curr_time

    return elapsed_time
