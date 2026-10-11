import contextlib,hashlib,io,json,os,subprocess,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from scripts.asset_verify import verify
from scripts.amd_common import finish_report,require_hip
from scripts.prepublish_guard import scan
from qiora_rocm.local_model import load_verified_parent

class Safety(unittest.TestCase):
    def manifest(self,root,entries):
        m=root/'m.json';m.write_text(json.dumps({'assets':entries}),encoding='utf8');return m
    def test_missing_size_invalid_hash_duplicate_and_absolute(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'a').write_bytes(b'x');entry=dict(destination='a',bytes=1,sha256=hashlib.sha256(b'x').hexdigest())
            for bad in [dict(entry,bytes=None),dict(entry,bytes=True),dict(entry,sha256='z'*64),dict(entry,destination='C:/private'),dict(entry,destination='/private')]:
                self.assertEqual('FAIL',verify(p,self.manifest(p,[bad]))['status'])
            self.assertEqual('DUPLICATE_DESTINATION',verify(p,self.manifest(p,[entry,entry]))['items'][1]['status'])
            self.assertEqual('FAIL',verify(p,self.manifest(p,[]))['status'])
    def test_symlink_parent_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'real').mkdir();(p/'real/a').write_bytes(b'x')
            try:(p/'link').symlink_to(p/'real',target_is_directory=True)
            except OSError:
                # Windows without symlink privileges: exercise the same guard
                # using the OS predicate, with Linux real-link validation pending.
                m=self.manifest(p,[dict(destination='link/a',bytes=1,sha256=hashlib.sha256(b'x').hexdigest())])
                with patch.object(Path,'is_symlink',lambda x:x==p/'link'):
                    self.assertEqual('SYMLINK_FORBIDDEN',verify(p,m)['items'][0]['status'])
                return
            m=self.manifest(p,[dict(destination='link/a',bytes=1,sha256=hashlib.sha256(b'x').hexdigest())])
            self.assertEqual('SYMLINK_FORBIDDEN',verify(p,m)['items'][0]['status'])
    def test_hip_guard_rejects_nvidia_and_absent_confirmation(self):
        fake=SimpleNamespace(version=SimpleNamespace(hip=None),cuda=SimpleNamespace(is_available=lambda:True))
        with self.assertRaisesRegex(ValueError,'CONFIRMATION'):require_hip(fake,False)
        with self.assertRaisesRegex(ValueError,'NON_ROCM'):require_hip(fake,True)
    def test_full_model_loader_rejects_before_touching_assets(self):
        with self.assertRaisesRegex(ValueError,'APPROVAL'):load_verified_parent('NOT_REAL','NOT_REAL')
    def test_report_roundtrip_no_overwrite_and_failure_saved(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'nested/report.json'
            with contextlib.redirect_stdout(io.StringIO()):r=finish_report(dict(status='FAIL'),p)
            self.assertEqual('FAIL',json.loads(p.read_text('utf8'))['status'])
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),Path(str(p)+'.sha256').read_text().strip())
            with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(FileExistsError):finish_report(dict(status='PASS'),p)
            self.assertEqual('FAIL',json.loads(p.read_text('utf8'))['status'])
    def test_publish_guard_catches_untracked_and_index_secret(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);subprocess.run(['git','init',str(p)],capture_output=True,check=True)
            bad=p/'secret.txt';bad.write_text('hf_'+'X'*40,encoding='utf8')
            self.assertEqual('BLOCK',scan(p)['status'])
            subprocess.run(['git','-C',str(p),'add','secret.txt'],check=True,capture_output=True);bad.write_text('clean',encoding='utf8')
            report=scan(p);self.assertEqual('BLOCK',report['status']);self.assertNotIn('X'*40,json.dumps(report))
    def test_publish_guard_blocks_large_and_weights(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);subprocess.run(['git','init',str(p)],capture_output=True,check=True)
            (p/'model.pt').write_bytes(b'fixture');(p/'huge.txt').write_text('A'*(512*1024+1))
            r=scan(p);self.assertEqual('BLOCK',r['status']);self.assertEqual(2,len(r['hits']))
    def test_numeric_failure_not_reported_as_success(self):
        from scripts.amd_probe_impl import inventory
        with patch('scripts.amd_probe_impl.require_hip',side_effect=ValueError('BLOCKED')):
            r=inventory('.',gpu_smoke=True,confirmed=True)
        self.assertEqual('BLOCKED',r['status']);self.assertEqual('FAIL',r['gpu_smoke']['status'])
    def test_yolo_unknown_hash_rejected_before_import(self):
        from scripts.amd_visual_impl import run
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fake.pt';p.write_bytes(b'not a model')
            with self.assertRaisesRegex(ValueError,'UNTRUSTED'):run(model_path=p,pin=hashlib.sha256(p.read_bytes()).hexdigest())
    def test_local_visual_is_repeatable(self):
        from scripts.amd_visual_impl import run
        self.assertEqual(run()['synthetic_output'],run()['synthetic_output'])
    def test_visual_nms_backend_is_exercised(self):
        from scripts.amd_visual_impl import run
        self.assertEqual([0,2],run()['synthetic_nms_selected_indices'])
    def test_missing_manifest_cli_saves_structured_failure(self):
        from scripts.asset_verify_v2 import main
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'report.json'
            with patch('sys.argv',['asset_verify.py','--root',td,'--manifest',td+'/missing.json','--out',str(p)]),contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(2,main())
            self.assertEqual('FAIL',json.loads(p.read_text('utf8'))['status'])

if __name__=='__main__':unittest.main()
