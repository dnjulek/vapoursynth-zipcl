# GaussBlur sigma=1 (small fused smem path) — vszipcl. No reference. num_streams from BENCH_NS.
import sys, os
sys.path.insert(0, os.getcwd())  # vspipe runs from the script dir (has _bench.py)
import _bench
_bench.load_vszipcl()
_bench.source().vszipcl.GaussBlur(sigma=1, num_streams=_bench.ns()).set_output(0)
