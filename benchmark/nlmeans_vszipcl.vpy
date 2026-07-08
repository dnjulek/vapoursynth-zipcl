# NLMeans — vszipcl (OpenCL). num_streams from BENCH_NS.
import sys, os
sys.path.insert(0, os.getcwd())  # vspipe runs from the script dir (has _bench.py)
import _bench
_bench.load_vszipcl()
_bench.source().vszipcl.NLMeans(d=1, a=2, s=4, h=1.2, num_streams=_bench.ns()).set_output(0)
