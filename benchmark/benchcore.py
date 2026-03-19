import timeit
import gc

TIME_FUNCTION_RUNS = 30

def study_fn(fn):
    time = timeit.timeit(fn, number=TIME_FUNCTION_RUNS) / TIME_FUNCTION_RUNS
    gc.collect()

    value = fn()
    gc.collect()

    return value, time