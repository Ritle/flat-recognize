import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from review_decisions import apply_review_decisions, make_review_questions


def opening(uid, kind, x, review=False):
    return {"id": uid, "type": kind,
            "outer": [[x, 10], [x + 8, 10], [x + 8, 50], [x, 50]],
            "classification": {"method": "jamb_leaf_and_arc",
                               "requires_review": review}}


class ReviewDecisionsTest(unittest.TestCase):
    def setUp(self):
        self.classified = {"meta": {"width": 100, "height": 80},
                           "doors": [opening(1, "door", 10, True)],
                           "windows": [opening(2, "window", 40)],
                           "rejected_openings": []}
        self.quality = {"issues": [{"code": "opening_perpendicular_wall", "severity": "error",
                                    "opening_kind": "windows", "opening_index": 1}]}

    def test_questions_identify_low_confidence_and_bad_binding(self):
        questions = make_review_questions(self.classified, self.quality, "a" * 64)
        self.assertEqual([q["opening_id"] for q in questions], [2, 1])
        self.assertEqual(questions[0]["reasons"], ["opening_perpendicular_wall"])
        self.assertEqual(questions[0]["center"], [44, 30])
        self.assertEqual(questions[0]["image_size"], [100, 80])

    def test_export_overhang_question_uses_opening_id(self):
        quality = {"issues": [{"code": "opening_exceeds_segment", "severity": "error",
                               "element_uuid": "exported-window", "opening_kind": "windows",
                               "opening_index": 1, "opening_id": 2,
                               "overhang_cm": 25, "opening_width_cm": 100,
                               "wall_length_cm": 60, "wall_chain_length_cm": 110,
                               "wall_chain_overhang_cm": 0}]}
        questions = make_review_questions(self.classified, quality, "a" * 64)
        self.assertEqual(questions[0]["opening_id"], 2)
        self.assertEqual(questions[0]["reasons"], ["opening_exceeds_segment"])
        self.assertEqual(questions[0]["severity"], "error")
        self.assertEqual(questions[0]["geometry"]["wall_chain_overhang_cm"], 0)

        quality["issues"][0]["opening_id"] = 999
        self.assertEqual([q["opening_id"] for q in
                          make_review_questions(self.classified, quality, "a" * 64)], [1])

    def test_choices_change_classification_without_mutating_source(self):
        original = copy.deepcopy(self.classified)
        questions = make_review_questions(self.classified, self.quality, "a" * 64)
        payload = {"version": 1, "image_sha256": "a" * 64,
                   "answers": [{"opening_id": q["opening_id"],
                                "fingerprint": q["fingerprint"],
                                "choice": "ignore" if q["opening_id"] == 2 else "window"}
                               for q in questions]}
        reviewed = apply_review_decisions(self.classified, payload, "a" * 64)
        self.assertEqual(self.classified, original)
        self.assertEqual(reviewed["doors"], [])
        self.assertEqual([x["id"] for x in reviewed["windows"]], [1])
        self.assertEqual([x["id"] for x in reviewed["rejected_openings"]], [2])
        self.assertTrue(reviewed["windows"][0]["classification"]["user_confirmed"])
        self.assertEqual(len(make_review_questions(reviewed, {}, "a" * 64)), 2)

    def test_stale_and_repeated_choices_are_rejected(self):
        q = make_review_questions(self.classified, self.quality, "a" * 64)[0]
        answer = {"opening_id": q["opening_id"], "fingerprint": q["fingerprint"],
                  "choice": "door"}
        payload = {"version": 1, "image_sha256": "a" * 64, "answers": [answer]}
        with self.assertRaisesRegex(ValueError, "another image"):
            apply_review_decisions(self.classified, payload, "b" * 64)
        with self.assertRaisesRegex(ValueError, "repeated"):
            apply_review_decisions(self.classified, {**payload, "answers": [answer, answer]}, "a" * 64)
        with self.assertRaisesRegex(ValueError, "changed"):
            apply_review_decisions(self.classified,
                                   {**payload, "answers": [{**answer, "fingerprint": "stale"}]},
                                   "a" * 64)


if __name__ == "__main__":
    unittest.main()
