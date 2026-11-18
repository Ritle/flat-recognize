import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import demo.app as api
from graph_service import GraphAnswerError, run_job
from graph_scope import validate_graph_scope
from test.test_review_graph_passages import fixture
from test.test_review_graph_doors import fixture as door_gap_fixture


class GraphServiceTest(unittest.TestCase):
    def test_saved_door_answer_on_rebuilt_graph_requires_the_same_question(self):
        graph, source, rgb, original = door_gap_fixture()
        gap = graph['diagnostics']['topology_gaps'][0]
        gap['reason'] = 'interior_raster_line_without_confirmed_type'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image, src, pred, answers = (root / name for name in
                                         ('plan.png', 'source.json', 'prediction.json', 'answers.json'))
            Image.fromarray(rgb).save(image)
            src.write_text(json.dumps(source), encoding='utf8')
            pred.write_text('{}', encoding='utf8')
            answer = {'id': 'gap:' + original['answers'][0]['gap_fingerprint'], 'choice': 'door'}
            payload = {'version': 1, 'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                       'answers': [answer]}
            answers.write_text(json.dumps(payload), encoding='utf8')
            with patch('graph_service.prepare_review_graph', return_value=(graph, [])):
                run_job(image, src, pred, root/'first', answers_path=answers, scale=5)
                project = json.loads((root/'first/project.json').read_text(encoding='utf8'))['levels'][0]
                self.assertEqual(len(project['areas']), 2)
                self.assertEqual([item['type'] for item in project['elements']], ['doorItem'])
                gap['local_width_px'] += 1
                with self.assertRaises(GraphAnswerError):
                    run_job(image, src, pred, root/'stale', answers_path=answers, scale=5)
                self.assertFalse((root/'stale/project.json').exists())

    def test_leaf_question_is_also_in_quality_report(self):
        graph, source, rgb, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); image = root / 'plan.png'
            Image.fromarray(np.uint8(rgb)).save(image)
            for name, obj in (('source.json', source), ('prediction.json', {}), ('graph.json', graph)):
                (root / name).write_text(json.dumps(obj), encoding='utf8')
            questions = {'questions': [{'id':'window:leaf', 'kind':'window', 'span':[[80,80],[120,80]],
                                        'reason':'possible_door_leaf_near_window'}]}
            with patch('graph_service.make_graph_questions', return_value=questions):
                quality = run_job(image, root/'source.json', root/'prediction.json', root/'out',
                                  graph_path=root/'graph.json')
            self.assertTrue(any(issue['code'] == 'graph_possible_door_leaf' for issue in quality['issues']))
            stored = json.loads((root/'out/graph.json').read_text(encoding='utf8'))
            self.assertEqual(stored['diagnostics']['opening_bridges'][0]['kind'], 'windows')

    def test_light_fill_flag_is_forwarded_only_when_enabled(self):
        with patch('demo.app.subprocess.run') as run:
            api.run_pipeline(Path('plan.png'), Path('project.json'), True, None, True)
            self.assertIn('--pipeline-v2', run.call_args.args[0])
            self.assertIn('--light-fill-cleanup', run.call_args.args[0])
            api.run_pipeline(Path('plan.png'), Path('project.json'), True)
            self.assertNotIn('--light-fill-cleanup', run.call_args.args[0])

    def test_scope_validation_and_no_changes_to_existing_graph(self):
        valid = {'exclude_polygons': [[[1, 1], [8, 1], [8, 8], [1, 8]]]}
        self.assertEqual(validate_graph_scope(valid, (10, 10)), valid['exclude_polygons'])
        invalid = [
            {'exclude_polygons': [[[1, 1], [10, 1], [1, 5]]]},
            {'exclude_polygons': [[[1, 1], [8, 8], [1, 8], [8, 1]]]},
            {'exclude_polygons': [[[float('nan'), 1], [8, 1], [1, 8]]]},
            {'exclude_polygons': [[]] * 33},
            {'exclude_polygons': [], 'unexpected': True},
        ]
        for value in invalid:
            with self.subTest(scope=value), self.assertRaises(ValueError):
                validate_graph_scope(value, (10, 10))

    def test_answered_passage_can_generate_followup_questions(self):
        graph, source, rgb, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image, src, prediction, graph_path = [root / name for name in
                ('plan.png', 'source.json', 'prediction.json', 'graph.json')]
            Image.fromarray(np.uint8(rgb)).save(image)
            for path, obj in ((src, source), (prediction, {}), (graph_path, graph)):
                path.write_text(json.dumps(obj), encoding='utf-8')
            first, second = root / 'first', root / 'second'
            quality = run_job(image, src, prediction, first, graph_path=graph_path)
            if quality['export_allowed']:
                self.assertTrue((first / 'graph_export.png').exists())
            questions = json.loads((first / 'graph_questions.json').read_text(encoding='utf-8'))
            question = questions['questions'][0]
            self.assertIn('open_passage', question['choices'])
            answers = root / 'answers.json'
            answers.write_text(json.dumps({'version': 1, 'image_sha256': questions['image_sha256'],
                                           'answers': [{'id': question['id'], 'choice': 'open_passage'}]}))
            run_job(image, src, prediction, second, graph_path=first / 'graph.json', answers_path=answers)
            reviewed = json.loads((second / 'graph.json').read_text(encoding='utf-8'))
            self.assertEqual(reviewed['diagnostics']['opening_bridges'][0]['kind'], 'passages')
            self.assertTrue((second / 'graph_questions.png').exists())

    def test_empty_rooms_cannot_be_exported(self):
        graph, source, rgb, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / 'plan.png'; Image.fromarray(np.uint8(rgb)).save(image)
            for name, obj in (('source.json', source), ('prediction.json', {}), ('graph.json', graph)):
                (root / name).write_text(json.dumps(obj), encoding='utf-8')
            quality = {'status': 'review', 'export_allowed': True, 'issues': [], 'errors': [], 'warnings': []}
            result = ({'levels': [{'areas': []}]}, {'quality': quality})
            with patch('graph_service.build_graph_project', return_value=result), \
                    patch('graph_service.make_graph_questions', return_value={'questions': []}):
                run_job(image, root / 'source.json', root / 'prediction.json', root / 'out',
                        graph_path=root / 'graph.json')
            self.assertFalse((root / 'out/project.json').exists())
            self.assertEqual(json.loads((root / 'out/quality.json').read_text())['status'], 'invalid')


class GraphServiceApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_upload_without_profile_flag_skips_legacy_geometry(self):
        png = b'known source bytes'
        digest = hashlib.sha256(png).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = dict(image_sha256=digest, scale_cm_per_px=3.9, wall_thickness_cm=20, scope={'exclude_polygons': []})
            (root/'recognition-profile.json').write_text(json.dumps(profile))
            (root/'opening-answers.json').write_text(json.dumps(dict(version=1, image_sha256=digest, answers=[])))
            def segmentation(image, project, v2, choices, light, *, graph_inputs_only):
                self.assertTrue(v2 and light and graph_inputs_only)
                self.assertIsNone(choices)
                for suffix in ('.json', '_classified.json'):
                    (root/(image.stem + suffix)).write_text('{}')
                return subprocess.CompletedProcess([], 0)
            async def build(parent, answers, scale, scope, *, saved_profile_sha256, wall_thickness_cm):
                self.assertEqual(saved_profile_sha256, digest)
                self.assertEqual(wall_thickness_cm, 20)
                self.assertEqual(scale, 3.9)
                self.assertTrue((root/parent/'classified.json').is_file())
                self.assertTrue((root/parent/'prediction.json').is_file())
                self.assertFalse((root/parent/'project.quality.json').exists())
                job = str(uuid.uuid4())
                (root/job).mkdir()
                return dict(job_id=job, parent_job_id=parent, image_sha256=digest, quality={'export_allowed':True})
            with patch.object(api, 'FOCUS_REVIEW', root), patch.object(api, 'JOBS_DIR', root), \
                    patch.object(api, 'OUTPUT_DIR', root), patch.object(api, 'run_pipeline', side_effect=segmentation), \
                    patch.object(api, 'build_graph_job', side_effect=build):
                for choices in (None, json.dumps(dict(version=1, image_sha256=digest, answers=[]))):
                    response = await api.recognize(UploadFile(filename='renamed.png', file=io.BytesIO(png)),
                                                   True, choices, True)
                    self.assertTrue(response['ok'] and response['saved_review'])
                    parent_meta = json.loads((root/response['parent_job_id']/'metadata.json').read_text())
                    self.assertEqual(parent_meta['graph_job_id'], response['job_id'])

    def test_saved_profile_is_bound_to_exact_image_digest(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(api, 'FOCUS_REVIEW', Path(directory)):
            profile = {'image_sha256': 'a'*64, 'scale_cm_per_px': 3.9}
            self.assertIsNone(api.saved_focus_profile('a'*64))
            (Path(directory)/'recognition-profile.json').write_text(json.dumps(profile), encoding='utf8')
            self.assertEqual(api.saved_focus_profile('a'*64), profile)
            self.assertIsNone(api.saved_focus_profile('b'*64))

    def test_profile_matches_only_identical_opaque_png_geometry(self):
        rgb = Image.new('RGB', (3, 2), 'red')
        profile = dict(image_sha256='a'*64, image_size=[3, 2],
                       opaque_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())

        def lookup(image, **options):
            stream = io.BytesIO()
            image.save(stream, format='PNG', **options)
            data = stream.getvalue()
            return api.saved_focus_profile(hashlib.sha256(data).hexdigest(), data)

        with tempfile.TemporaryDirectory() as directory, patch.object(api, 'FOCUS_REVIEW', Path(directory)):
            (Path(directory)/'recognition-profile.json').write_text(json.dumps(profile))
            self.assertEqual(lookup(rgb), profile)
            self.assertEqual(lookup(rgb.convert('RGBA')), profile)
            self.assertIsNone(lookup(rgb, transparency=(255, 0, 0)))
            alpha = rgb.convert('RGBA')
            alpha.putpixel((1, 1), (255, 0, 0, 254))
            self.assertIsNone(lookup(alpha))
            self.assertIsNone(lookup(rgb.resize((2, 3))))
            rotated = Image.Exif()
            rotated[274] = 6
            self.assertIsNone(lookup(rgb, exif=rotated))
            rgb.putpixel((1, 1), (254, 0, 0))
            self.assertIsNone(lookup(rgb))

    def test_only_reviews_from_matching_profile_can_be_rebound(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(api, 'FOCUS_REVIEW', Path(directory)):
            path = Path(directory)/'opening-answers.json'
            original = dict(version=1, image_sha256='a'*64, answers=[])
            path.write_text(json.dumps(original))
            updated = api.saved_review_payload(path.name, 'a'*64, 'b'*64)
            self.assertEqual(updated, {**original, 'image_sha256': 'b'*64})
            self.assertEqual(json.loads(path.read_text()), original)
            with self.assertRaises(HTTPException):
                api.saved_review_payload(path.name, 'c'*64, 'b'*64)

    async def test_light_fill_requires_v2(self):
        with self.assertRaises(HTTPException) as error:
            await api.recognize(None, False, None, True)
        self.assertEqual(error.exception.status_code, 400)

    async def test_initial_scope_is_passed_in_source_pixels_and_invalid_scope_rejected(self):
        graph, source, rgb, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); jobs = root / 'jobs'; output = root / 'output'; output.mkdir()
            parent_id = str(uuid.uuid4()); parent = jobs / parent_id; parent.mkdir(parents=True)
            Image.fromarray(np.uint8(rgb)).save(parent / f'{parent_id}.png')
            (parent / 'classified.json').write_text(json.dumps(source), encoding='utf-8')
            (parent / 'prediction.json').write_text('{}', encoding='utf-8')
            scope = {'exclude_polygons': [[[1, 1], [8, 1], [8, 8], [1, 8]]]}
            decisions = {'version': 1, 'image_sha256': hashlib.sha256(
                (parent / f'{parent_id}.png').read_bytes()).hexdigest(),
                'answers': [{'id': 'gap:saved', 'choice': 'door'}]}
            def pipeline(image, classified, prediction, destination, parent_graph, answers, scale, scope_path):
                self.assertIsNone(parent_graph)
                self.assertEqual(json.loads(answers.read_text()), decisions)
                self.assertEqual(json.loads(scope_path.read_text()), scope)
                for name, value in (
                    ('quality.json', {'status': 'invalid', 'export_allowed': False, 'warnings': []}),
                    ('graph_questions.json', {'questions': [], 'image_sha256': 'hash'}),
                    ('graph.json', {'scale_cm_per_px': scale}),
                ):
                    (destination / name).write_text(json.dumps(value), encoding='utf-8')
                return subprocess.CompletedProcess([], 0, stdout='', stderr='')
            with patch.object(api, 'JOBS_DIR', jobs), patch.object(api, 'OUTPUT_DIR', output), \
                    patch.object(api, 'run_graph_pipeline', side_effect=pipeline):
                response = await api.graph_review(parent_id, json.dumps(decisions), 3.9, json.dumps(scope))
                self.assertIsNone(response['download_url'])
                self.assertEqual(json.loads((jobs / response['job_id'] / 'graph_scope.json').read_text()), scope)
                with self.assertRaises(HTTPException) as error:
                    await api.graph_review(parent_id, None, 3.9,
                                           json.dumps({'exclude_polygons': [[[0, 0], [999, 0], [0, 1]]]}))
                self.assertEqual(error.exception.status_code, 400)

    async def test_endpoint_isolated_jobs_and_stale_answers(self):
        graph, source, rgb, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); jobs = root / 'jobs'; output = root / 'output'
            output.mkdir(); parent_id = str(uuid.uuid4()); parent = jobs / parent_id
            parent.mkdir(parents=True)
            image = parent / f'{parent_id}.png'; Image.fromarray(np.uint8(rgb)).save(image)
            (parent / 'classified.json').write_text(json.dumps(source), encoding='utf-8')
            (parent / 'graph.json').write_text(json.dumps(graph), encoding='utf-8')
            (parent / 'prediction.json').write_text('{}', encoding='utf-8')
            with patch.object(api, 'JOBS_DIR', jobs), patch.object(api, 'OUTPUT_DIR', output):
                first = await api.graph_review(parent_id, None, 3.9)
                self.assertNotEqual(first['job_id'], parent_id)
                self.assertEqual(first['parent_job_id'], parent_id)
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                self.assertEqual(first['image_sha256'], digest)
                self.assertTrue((jobs / first['job_id'] / 'graph_questions.png').exists())
                stale = {'version': 1, 'image_sha256': digest,
                         'answers': [{'id': 'stale', 'choice': 'door'}]}
                with self.assertRaises(HTTPException) as error:
                    await api.graph_review(first['job_id'], json.dumps(stale), 3.9)
                self.assertEqual(error.exception.status_code, 409)
                self.assertEqual(json.loads((parent / 'graph.json').read_text()), graph)
                with self.assertRaises(HTTPException) as error:
                    await api.graph_review(parent_id, None, float('nan'))
                self.assertEqual(error.exception.status_code, 400)
                with self.assertRaises(HTTPException) as error:
                    await api.graph_review(parent_id, None, 3.9, '{"exclude_polygons": []}')
                self.assertEqual(error.exception.status_code, 400)


if __name__ == '__main__':
    unittest.main()
