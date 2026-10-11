"""Reviewed publication guard; does not publish or print secret values."""
try:from .prepublish_impl import scan,main
except ImportError:from prepublish_impl import scan,main
if __name__=="__main__":raise SystemExit(main())
