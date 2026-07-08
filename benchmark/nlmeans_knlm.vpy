# NLMeans — reference knlm.KNLMeansCL (OpenCL). No num_streams.
import sys, os
sys.path.insert(0, os.getcwd())  # vspipe runs from the script dir (has _bench.py)
import _bench
_bench.source().knlm.KNLMeansCL(d=1, a=2, s=4, h=1.2).set_output(0)
