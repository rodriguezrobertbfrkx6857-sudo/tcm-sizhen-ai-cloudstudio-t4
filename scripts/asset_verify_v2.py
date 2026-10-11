"""Strict local, streaming asset verifier. No model deserialization or download."""
import argparse, json, re, sys
from pathlib import Path, PureWindowsPath
try:from .amd_common import sha256,finish_report
except ImportError:from amd_common import sha256,finish_report

def verify(root, manifest):
    root=Path(root).resolve();details=[];seen=set()
    try:
        obj=json.loads(Path(manifest).read_text('utf8'))
        entries=obj['assets'] if isinstance(obj,dict) else obj
        if not isinstance(entries,list) or not entries:raise ValueError('EMPTY_OR_INVALID_MANIFEST')
    except (OSError,ValueError,KeyError,TypeError):return dict(status='FAIL',count=0,items=[],reason='INVALID_MANIFEST')
    for item in entries:
        rel=None
        try:
            if not isinstance(item,dict):raise ValueError('BAD_MANIFEST_ENTRY')
            rel=item.get('destination') or item.get('relative_path')
            pin=item.get('sha256') or item.get('SHA256');size=item.get('bytes')
            if not isinstance(rel,str) or not rel or '\\' in rel or Path(rel).is_absolute() or PureWindowsPath(rel).drive or '..' in Path(rel).parts:raise ValueError('PATH_ESCAPE')
            if not isinstance(pin,str) or not re.fullmatch('[0-9a-fA-F]{64}',pin) or type(size) is not int or size<0:raise ValueError('BAD_MANIFEST_ENTRY')
            if rel.casefold() in seen:raise ValueError('DUPLICATE_DESTINATION')
            seen.add(rel.casefold());candidate=root/rel
            if any(p.is_symlink() for p in [candidate,*candidate.parents] if p!=root):raise ValueError('SYMLINK_FORBIDDEN')
            resolved=candidate.resolve()
            if not resolved.is_relative_to(root) or resolved==root:raise ValueError('PATH_ESCAPE')
            if not resolved.is_file():raise ValueError('MISSING')
            if resolved.stat().st_size!=size:raise ValueError('SIZE_MISMATCH')
            digest=sha256(resolved)
            if digest.lower()!=pin.lower():raise ValueError('SHA_MISMATCH')
            details.append(dict(asset=rel,status='PASS',bytes=size,sha256=digest))
        except (ValueError,OSError) as exc:details.append(dict(asset=rel,status=str(exc) if isinstance(exc,ValueError) else 'READ_FAILURE'))
    return dict(status='PASS' if all(d['status']=='PASS' for d in details) else 'FAIL',count=len(details),items=details)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out');a=p.parse_args()
    r=verify(a.root,a.manifest);finish_report(dict(r,manifest_SHA256=sha256(a.manifest) if Path(a.manifest).is_file() else None,measurement='LOCAL_ASSET_READBACK_ONLY_NOT_MODEL_LOAD'),a.out)
    return 0 if r['status']=='PASS' else 2
if __name__=='__main__':sys.exit(main())
