import pytest
import numpy as np

def test_dummy(benchmark):
    def work():
        return sum(range(100))
    benchmark(work)
    
    stats_obj = getattr(benchmark.stats, "stats", benchmark.stats)
    print("\ntype(stats_obj):", type(stats_obj))
    print("stats_obj.mean:", stats_obj.mean)
    print("stats_obj.data length:", len(stats_obj.data))
    p95 = np.percentile(stats_obj.data, 95)
    print("p95 calculated:", p95)
