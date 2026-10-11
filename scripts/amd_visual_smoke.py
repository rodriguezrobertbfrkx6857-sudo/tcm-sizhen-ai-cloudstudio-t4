"""Stable CLI for the reviewed ROCm implementation."""
try:from .amd_visual_impl import *
except ImportError:from amd_visual_impl import *
if __name__=="__main__":raise SystemExit(main())
