"""Audit explicit code-only publish candidates, never print matched secret values."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ALLOW={'cloudstudio_tongue','qiora_rocm','configs','docs','reference','scripts','tests'}
TOP={'.gitignore','README.md','DISCLAIMER.md','LICENSE_NOTICE.md','requirements.txt','requirements-cu126.txt','requirements-yolo.txt','CODE_MANIFEST.json','SHA256SUMS'}
PATTERNS=[re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----'),re.compile(r'(?:ghp_|gho_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{20,}'),re.compile(r'AKIA[A-Z0-9]{16}'),re.compile(r'(?i)(?:api_key|secret_key|access_token|password)\s*[:=]\s*[\x22\x27][A-Za-z0-9_\-]{20,}')]

def candidates():
    return [p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts and p.suffix!='.pyc']

def audit(paths):
    findings=[];files=[]
    for p in paths:
        rel=p.relative_to(ROOT);name=rel.as_posix()
        if not ((len(rel.parts)>1 and rel.parts[0] in ALLOW) or name in TOP):findings.append(dict(file=name,reason='NOT_IN_PUBLISH_ALLOWLIST'));continue
        if p.suffix.lower() not in {'.py','.json','.md','.sh','.txt'} and name not in {'.gitignore','SHA256SUMS'}:findings.append(dict(file=name,reason='NON_CODE_EXTENSION'));continue
        if p.stat().st_size>512*1024:findings.append(dict(file=name,reason='OVER_CODE_SIZE_LIMIT'));continue
        raw=p.read_bytes()
        try:text=raw.decode('utf8')
        except UnicodeError:findings.append(dict(file=name,reason='NON_TEXT'));continue
        for line_no,line in enumerate(text.splitlines(),1):
            if any(pattern.search(line) for pattern in PATTERNS):findings.append(dict(file=name,line=line_no,reason='POTENTIAL_CREDENTIAL_REDACTED'))
        files.append(dict(path=name,bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest()))
    if findings:raise ValueError(json.dumps(findings))
    return dict(status='CODE_ALLOWLIST_AND_SECRET_PATTERN_SCAN_PASS',files=files,known_patient_data_included=False,assets_or_cache_included=False,limitations='Heuristic scan plus manually selected code-only inputs; not a universal PHI detector.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--staged',action='store_true');a=p.parse_args()
    if a.staged:
        names=subprocess.run(['git','diff','--cached','--name-only','--diff-filter=ACMR'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.splitlines();paths=[ROOT/x for x in names]
        for path in paths:
            blob=subprocess.run(['git','show',':'+path.relative_to(ROOT).as_posix()],cwd=ROOT,capture_output=True,check=True).stdout
            if blob!=path.read_bytes():raise ValueError('INDEX_WORKTREE_MISMATCH')
    else:paths=candidates()
    z=audit(paths);print(json.dumps(dict(status=z['status'],files=len(z['files']),bytes=sum(x['bytes'] for x in z['files']))))
if __name__=='__main__':main()
