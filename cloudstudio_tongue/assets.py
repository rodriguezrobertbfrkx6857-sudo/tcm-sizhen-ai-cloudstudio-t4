import hashlib,json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def verify(root,manifest):
    root=Path(root).resolve();spec=json.loads(Path(manifest).read_text('utf8'));paths={}
    for entry in spec['assets']:
        relative=Path(entry['destination'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('ASSET_PATH_ESCAPE')
        p=(root/relative).resolve()
        if not p.is_relative_to(root):raise ValueError('ASSET_PATH_ESCAPE')
        if not p.is_file():raise FileNotFoundError('MISSING_ASSET:'+entry['id'])
        if p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise ValueError('ASSET_SHA_MISMATCH:'+entry['id'])
        paths[entry['id']]=p
    return spec,paths
