# EEDI3H (horizontal) — reference vszip (CPU). No num_streams. f32-only -> u16/f16 are N/A.
import sys, os
sys.path.insert(0, os.getcwd())  # vspipe runs from the script dir (has _bench.py)
import _bench
_bench.source().vszip.EEDI3H(field=1, mdis=40, nrad=3).set_output(0)
