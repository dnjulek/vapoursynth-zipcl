# Deband — vszipcl (OpenCL). num_streams from BENCH_NS.
import sys, os
sys.path.insert(0, os.getcwd())  # vspipe runs from the script dir (has _bench.py)
import _bench
_bench.load_vszipcl()
_bench.source().vszipcl.Deband(iterations=1, threshold=4, radius=16, grain=6, planes=[0], num_streams=_bench.ns()).set_output(0)
