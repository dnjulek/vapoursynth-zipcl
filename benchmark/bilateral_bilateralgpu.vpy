# Bilateral — reference bilateralgpu (CUDA). num_streams from BENCH_NS. (rejects f16 -> N/A there)
import sys, os
sys.path.insert(0, os.getcwd())  # vspipe runs from the script dir (has _bench.py)
import _bench
_bench.source().bilateralgpu.Bilateral(sigma_spatial=3, sigma_color=0.02, num_streams=_bench.ns()).set_output(0)
