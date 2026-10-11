"""Stable CLI name for the reviewed strict verifier."""
try:from .asset_verify_v2 import verify,main
except ImportError:from asset_verify_v2 import verify,main
if __name__=='__main__':raise SystemExit(main())
