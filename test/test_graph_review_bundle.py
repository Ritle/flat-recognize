import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from graph_review_bundle import create_bundle, read_bundle, replay_bundle_files
from test.test_review_graph_doors import fixture


class GraphReviewBundleTest(unittest.TestCase):
    def test_bundle_replays_and_detects_corruption_without_input_mutation(self):
        graph,source,rgb,decision=fixture();before=copy.deepcopy((graph,source))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=root/'image.png';Image.fromarray(rgb).save(image)
            decision['image_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
            paths=[root/'graph.json',root/'source.json',root/'decision.json',root/'manifest.json']
            for p,obj in zip(paths,(graph,source,decision,dict(version=1,steps=[dict(kind='gap_door',decision='decision.json')]))):
                p.write_text(json.dumps(obj),encoding='utf-8')
            archive=root/'bundle.zip';templates=[ROOT/'test/template.json',ROOT/'test/template_window.json',ROOT/'test/template_passage.json']
            created=create_bundle(paths[0],paths[1],image,paths[3],templates,archive)
            files=read_bundle(archive);restored=replay_bundle_files(files)
            self.assertEqual(restored['graph'],created['graph'])
            self.assertEqual(restored['audit']['status'],'pass')
            self.assertEqual(len(restored['report']['bindings']['doors']),1)
            self.assertEqual((graph,source),before)
            self.assertIn('image.bin',files)
            with self.assertRaises(FileExistsError):create_bundle(paths[0],paths[1],image,paths[3],templates,archive)
            corrupt=root/'corrupt.zip'
            with ZipFile(corrupt,'w') as z:
                for name,data in files.items():z.writestr(name,data+b' ' if name=='source.json' else data)
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):read_bundle(corrupt)

    def test_unsafe_paths_are_rejected_without_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.zip'
            with ZipFile(path,'w') as z:z.writestr('../escape.json','{}')
            with self.assertRaisesRegex(ValueError,'Unsafe bundle path'):read_bundle(path)


if __name__=='__main__':unittest.main()
