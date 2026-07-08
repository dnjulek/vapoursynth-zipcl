# NLMeans — reference nlm_cuda (CUDA). num_streams from BENCH_NS. (rejects f16 -> N/A there)
import sys, os
sys.path.insert(0, os.getcwd())  # vspipe runs from the script dir (has _bench.py)
import _bench
_bench.source().nlm_cuda.NLMeans(d=1, a=2, s=4, h=1.2, num_streams=_bench.ns()).set_output(0)
