import json,tempfile,unittest
from pathlib import Path
from cloudstudio_tongue.assets import verify,sha
from cloudstudio_tongue.evaluate import safe

class Contracts(unittest.TestCase):
    def test_SHA_and_missing_assets(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'file').write_bytes(b'fixture');m=root/'manifest.json';m.write_text(json.dumps({'assets':[dict(id='fixture',destination='file',bytes=7,sha256=sha(root/'file'))]}))
            self.assertIn('fixture',verify(root,m)[1]);(root/'file').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify(root,m)
            (root/'file').unlink()
            with self.assertRaises(FileNotFoundError):verify(root,m)
    def test_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();self.assertEqual(safe(root,'relative.png'),root/'relative.png')
            for bad in ['../secret','/etc/passwd']:
                with self.assertRaises(ValueError):safe(root,bad)
    def test_asset_path_escape(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'manifest.json';p.write_text(json.dumps({'assets':[dict(id='bad',destination='../secret',bytes=0,sha256='0'*64)]}))
            with self.assertRaises(ValueError):verify(td,p)
    def test_yolo_dataset_no_download_or_escape(self):
        import yaml
        from cloudstudio_tongue.yolo_evaluate import validate_yaml
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'train').mkdir();(root/'val').mkdir();p=root/'data.yaml';cfg=dict(path='.',train='train',val='val',names=['a'])
            p.write_text(yaml.safe_dump(cfg));self.assertEqual(validate_yaml(p,root)['path'],str(root.resolve()))
            for bad in [dict(cfg,download='forbidden'),dict(cfg,path='..'),dict(cfg,val='https://invalid.invalid/data'),dict(cfg,test='locked_test')]:
                p.write_text(yaml.safe_dump(bad))
                with self.assertRaises(ValueError):validate_yaml(p,root)
    def test_yolo_untrusted_SHA_rejected_before_deserialization(self):
        from cloudstudio_tongue.yolo_infer import load_yolo
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'fake.pt').write_bytes(b'not a checkpoint');p=root/'manifest.json'
            p.write_text(json.dumps({'assets':[dict(id='yolo_six_class',destination='fake.pt',bytes=16,sha256=sha(root/'fake.pt'))]}))
            with self.assertRaises(ValueError):load_yolo(root,p)
    def test_publish_scan_detects_credentials_without_outputting_values(self):
        from scripts import audit_package as audit
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'docs').mkdir();p=root/'docs/fake.txt';p.write_text('ghp_'+'X'*40)
            prior=audit.ROOT;audit.ROOT=root
            try:
                with self.assertRaises(ValueError) as error:audit.audit([p])
                self.assertNotIn('X'*40,str(error.exception));self.assertIn('REDACTED',str(error.exception))
            finally:audit.ROOT=prior
if __name__=='__main__':unittest.main()
