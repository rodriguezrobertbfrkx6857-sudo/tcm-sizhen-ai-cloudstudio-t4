import importlib.util, json, tempfile, unittest, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'scripts'))
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
class Tests(unittest.TestCase):
    def test_rocm_probe_cpu_safe(self):
        m=load('amd_rocm_probe');x=m.inventory(workspace=str(ROOT),gpu_smoke=False)
        self.assertTrue(x['no_training_performed']);self.assertFalse(x['gpu_smoke_requested'])
    def test_asset_hash(self):
        m=load('asset_verify')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'weights').mkdir();(root/'weights'/'a.txt').write_bytes(b'ABC')
            d={'assets':[{'destination':'weights/a.txt','bytes':3,'sha256':hashlib.sha256(b'ABC').hexdigest()}]}
            (root/'m.json').write_text(json.dumps(d))
            self.assertEqual('PASS',m.verify(root,root/'m.json')['status'])
            (root/'weights'/'a.txt').write_bytes(b'BAD')
            self.assertEqual('FAIL',m.verify(root,root/'m.json')['status'])
    def test_path_escape(self):
        m=load('asset_verify')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'m.json').write_text(json.dumps({'assets':[{'destination':'../escape','sha256':'a'*64}]}))
            self.assertEqual('PATH_ESCAPE',m.verify(root,root/'m.json')['items'][0]['status'])
    def test_profile_no_training(self):
        d=json.loads((ROOT/'configs'/'amd_rocm_profile.json').read_text())
        self.assertFalse(d['approved_training']);self.assertFalse(d['allow_model_download'])
    def test_prepublication_guard_no_repo(self):
        m=load('prepublish_guard')
        with tempfile.TemporaryDirectory() as t:self.assertEqual('NO_GIT_REPOSITORY',m.scan(t)['status'])
if __name__=='__main__':unittest.main()
