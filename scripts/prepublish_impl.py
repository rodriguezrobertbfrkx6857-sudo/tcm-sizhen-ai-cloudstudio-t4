"""Scan HEAD, staged blobs and untracked publish candidates; redact matches."""
import argparse,json,re,subprocess
from pathlib import Path
BAD_SUFFIX={'.pt','.pth','.ckpt','.safetensors','.bin','.gguf','.zip','.jpg','.jpeg','.png','.webp','.csv','.jsonl','.parquet','.sqlite','.db','.key','.pem','.p12','.pfx'}
BAD_BASE={'.env','id_rsa','id_ed25519','credentials.json','secrets.json'}
BAD_DIR={'artifacts','weights','patients','datasets','images','raw','checkpoints','outputs','.venv','.venv-rocm','data','assets','logs','reports_private'}
PATTERNS=[re.compile(x,re.I) for x in [r'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----',r'gh[pousr]_[A-Za-z0-9_\-]{20,}',r'github_pat_[A-Za-z0-9_\-]{20,}',r'hf_[A-Za-z0-9]{20,}',r'sk-proj-[A-Za-z0-9_\-]{20,}',r'AKIA[A-Z0-9]{16}',r'(?:api_key|access_token|password)\s*[:=]\s*[\x22\x27][A-Za-z0-9_\-]{20,}']]

def problem(name,raw):
    p=Path(name)
    if p.name.lower() in BAD_BASE or p.name.lower().startswith('.env.') or p.suffix.lower() in BAD_SUFFIX or any(x.lower() in BAD_DIR for x in p.parts):return 'BLOCKED_PATH'
    if len(raw)>512*1024:return 'OVER_TEXT_SIZE_LIMIT'
    try:value=raw.decode('utf8')
    except UnicodeError:return 'NON_TEXT'
    if '\x00' in value:return 'NON_TEXT'
    if any(q.search(value) for q in PATTERNS):return 'POTENTIAL_SECRET_REDACTED'
    return None

def scan(root):
    root=Path(root).resolve()
    if not (root/'.git').exists():return dict(status='NO_GIT_REPOSITORY',hits=[])
    def git(*a):return subprocess.check_output(['git','-C',str(root),*a])
    hits=[];count=0
    try:
        names=set(filter(None,git('ls-files','-z','--cached','--others','--exclude-standard').decode().split('\0')))
        for name in sorted(names):
            p=root/name
            if p.is_symlink():hits.append(dict(path=name,reason='SYMLINK'));continue
            if not p.exists():continue # staged deletion is not a publish payload
            if not p.is_file():hits.append(dict(path=name,reason='NON_FILE'));continue
            reason=problem(name,p.read_bytes());count+=1
            if reason:hits.append(dict(path=name,reason=reason,scope='WORKTREE_OR_UNTRACKED'))
        # Use index bytes as well: the staged content may differ from worktree.
        for line in git('ls-files','--stage').decode().splitlines():
            fields,name=line.split('\t',1);mode,oid,stage=fields.split()
            reason='SYMLINK' if mode=='120000' else problem(name,git('cat-file','blob',oid))
            if reason:hits.append(dict(path=name,reason=reason,scope='INDEX'))
    except (OSError,subprocess.CalledProcessError):return dict(status='GIT_OR_READ_ERROR',hits=hits)
    return dict(status='PASS' if not hits else 'BLOCK',hits=hits,candidate_count=count,scan_limitations='Selected text code and heuristic scan, not universal privacy detection; history separately reviewed.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=scan(a.root);print(json.dumps(r,ensure_ascii=False,indent=2));return 0 if r['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
