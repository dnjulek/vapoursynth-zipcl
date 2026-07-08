# Deband — reference placebo (libplacebo / Vulkan). No num_streams.
import sys, os
sys.path.insert(0, os.getcwd())  # vspipe runs from the script dir (has _bench.py)
import _bench
_bench.source().placebo.Deband(iterations=1, threshold=4, radius=16, grain=6, planes=1).set_output(0)
