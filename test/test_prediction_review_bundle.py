import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from graph_review_bundle import read_bundle
from prediction_review_bundle import build_files, replay_prediction_files
from test.test_review_graph_doors import fixture


class PredictionBundleTest(unittest.TestCase):
    def inputs(self):
        graph,source,rgb,review=fixture()
        out=io.BytesIO();Image.fromarray(rgb).save(out,format='PNG');image=out.getvalue()
        review['image_sha256']=hashlib.sha256(image).hexdigest()
        manifest=dict(version=1,steps=[dict(kind='gap_door',decision_bytes=json.dumps(review).encode())])
        templates=[(ROOT/'test'/name).read_bytes() for name in ('template.json','template_window.json','template_passage.json')]
        return graph,source,image,manifest,templates

    def test_bundle_replays_saved_prediction_and_reviews(self):
        graph,source,image,manifest,templates=self.inputs();before=copy.deepcopy((graph,source))
        files=build_files(json.dumps(source).encode(),image,b'{}',manifest,templates,scale=5)
        with patch('prediction_review_bundle.prepare_review_graph',return_value=(graph,[dict(stage='graph')])) as prepare:
            result=replay_prediction_files(files)
        self.assertEqual(len(result['report']['bindings']['doors']),1)
        self.assertEqual(result['audit']['status'],'pass')
        self.assertEqual(len(result['report']['preparation_stages']),1)
        self.assertEqual((graph,source),before)
        self.assertEqual(prepare.call_args.kwargs['prediction'],{})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bundle.zip'
            with ZipFile(path,'w') as archive:
                for name,data in files.items():archive.writestr(name,data)
            self.assertEqual(read_bundle(path),files)
            bad=Path(directory)/'bad.zip'
            with ZipFile(bad,'w') as archive:
                for name,data in files.items():archive.writestr(name,data+b' ' if name=='prediction.json' else data)
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):read_bundle(bad)

    def test_rejects_bad_configuration_paths_and_scale(self):
        graph,source,image,manifest,templates=self.inputs()
        for scale in (0,-1,float('nan'),True):
            with self.subTest(scale=scale),self.assertRaises(ValueError):
                build_files(json.dumps(source).encode(),image,b'{}',manifest,templates,scale=scale)
        files=build_files(json.dumps(source).encode(),image,b'{}',manifest,templates,scale=5)
        bad=dict(files);config=json.loads(bad['config.json']);config['scope']='false';bad['config.json']=json.dumps(config).encode()
        with self.assertRaises(ValueError):replay_prediction_files(bad)
        bad=dict(files);bad['unexpected.json']=b'{}'
        with self.assertRaises(ValueError):replay_prediction_files(bad)
        bad=dict(files);manifest2=json.loads(bad['manifest.json']);manifest2['steps'][0]['decision']='../decision.json';bad['manifest.json']=json.dumps(manifest2).encode()
        with self.assertRaises(ValueError):replay_prediction_files(bad)


if __name__=='__main__':unittest.main()
