import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from review_graph_passages import passage_fingerprint
from review_graph_windows import confirm_graph_windows
from test.test_review_graph_passages import fixture


class GraphWindowConfirmationTest(unittest.TestCase):
    def test_confirmation_preserves_geometry_and_records_user_answer(self):
        graph, source, rgb, _ = fixture()
        graph['diagnostics']['opening_bridges'][0]['kind'] = 'recovered_window'
        before = copy.deepcopy((graph, source))
        fingerprint = passage_fingerprint(graph['diagnostics']['opening_bridges'][0])
        answers = {'version': 1, 'image_sha256': 'a'*64, 'answers': [
            {'fingerprint': fingerprint, 'choice': 'window', 'user_statement': 'небольшое окно'}]}
        result = confirm_graph_windows(graph, source, rgb.shape[1::-1], answers, 'a'*64)
        bridge = result['diagnostics']['opening_bridges'][0]
        self.assertEqual(bridge['kind'], 'recovered_window')
        self.assertEqual(bridge['window_review']['user_statement'], 'небольшое окно')
        self.assertEqual(result['edges'], graph['edges'])
        self.assertEqual((graph, source), before)
        self.assertEqual(confirm_graph_windows(result, source, rgb.shape[1::-1], answers, 'a'*64), result)

    def test_stale_wrong_type_duplicate_and_hash_are_rejected(self):
        for case in ('stale', 'type', 'duplicate', 'hash', 'size', 'empty_statement'):
            graph, source, rgb, _ = fixture()
            fingerprint = passage_fingerprint(graph['diagnostics']['opening_bridges'][0])
            answers = {'version': 1, 'image_sha256': 'a'*64, 'answers': [
                {'fingerprint': fingerprint, 'choice': 'window', 'user_statement': 'окно'}]}
            size = rgb.shape[1::-1]
            if case == 'stale': answers['answers'][0]['fingerprint'] = 'stale'
            elif case == 'type': graph['diagnostics']['opening_bridges'][0]['kind'] = 'doors'
            elif case == 'duplicate': answers['answers'] *= 2
            elif case == 'hash': answers['image_sha256'] = 'b'*64
            elif case == 'size': size = (1,1)
            else: answers['answers'][0]['user_statement'] = '  '
            with self.subTest(case=case), self.assertRaises(ValueError):
                confirm_graph_windows(graph, source, size, answers, 'a'*64)


if __name__ == '__main__': unittest.main()
