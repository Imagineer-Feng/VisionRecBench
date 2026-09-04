import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_logic_audit import load_base_results
from source.agent import EVALUATION_PROTOCOL_VERSION
from source.logic_audit import (
    build_logic_audit_prompt,
    expected_logic_audit,
    parse_json_object,
    score_logic_audit,
)
from source.logic_audit_metrics import summarize_logic_audit


def _step(number, command, applied):
    return {
        "step": number,
        "command": {"step": number, "name": f"command-{number}", "delta": command},
        "applied_commands": {
            str(candidate): vector for candidate, vector in applied.items()
        },
    }


class LogicAuditTest(unittest.TestCase):
    def test_base_result_discovery_accepts_only_latest_protocol(self):
        common = {
            "dataset_name": "dataset",
            "dataset_content_sha256": "hash",
            "episode_signature": "signature",
            "model": "gpt-4o",
            "input_content_sha256": "input-hash",
            "answer_index": 1,
            "choice": 1,
            "valid": True,
            "correct": True,
            "response_text": "Choice: [1]",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            latest = {
                **common,
                "episode_id": "latest",
                "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            }
            old = {
                **common,
                "episode_id": "old",
                "evaluation_protocol_version": "eval_v2_compatible",
            }
            (root / "latest.json").write_text(json.dumps(latest))
            (root / "old.json").write_text(json.dumps(old))

            results = load_base_results(root, "gpt-4o", "dataset", "hash")

        self.assertEqual(set(results), {"latest"})

    def test_scene1_requires_every_candidate_step_relation(self):
        record = {
            "episode_id": "scene1-test",
            "scenario": "scene1_single_command_causality",
            "steps": [
                _step(1, [1, 0], {1: [1, 0], 2: [0, 1]}),
                _step(2, [0, 1], {1: [0, 1], 2: [0, 1]}),
            ],
        }
        expected = expected_logic_audit(record)
        self.assertEqual(expected["audit_type"], "step_match")
        self.assertEqual(expected["candidates"][1], ["match", "match"])
        self.assertEqual(expected["candidates"][2], ["mismatch", "match"])
        self.assertIn("exactly 2 relations", build_logic_audit_prompt(record))

        response = json.dumps(
            {
                "audit_type": "step_match",
                "candidates": [
                    {"candidate": 2, "step_relations": ["mismatch", "match"]},
                    {"candidate": 1, "step_relations": ["match", "match"]},
                ],
            }
        )
        score = score_logic_audit(record, response)
        self.assertTrue(score["response_valid"])
        self.assertTrue(score["exact_match"])
        self.assertEqual(score["correct_items"], 4)

        wrong = response.replace('"mismatch"', '"match"', 1)
        wrong_score = score_logic_audit(record, wrong)
        self.assertFalse(wrong_score["exact_match"])
        self.assertEqual(wrong_score["correct_items"], 3)

    def test_scene2_compares_every_cycle_with_cycle_one(self):
        record = {
            "episode_id": "scene2-test",
            "scenario": "scene2_single_scrambled_stability",
            "task": {"command_sequence": [{"delta": [1, 0]}, {"delta": [0, 1]}]},
            "steps": [
                _step(1, [1, 0], {1: [0, 1], 2: [0, 1]}),
                _step(2, [0, 1], {1: [1, 0], 2: [1, 0]}),
                _step(3, [1, 0], {1: [0, 1], 2: [1, 0]}),
                _step(4, [0, 1], {1: [1, 0], 2: [0, 1]}),
                _step(5, [1, 0], {1: [0, 1], 2: [0, 1]}),
                _step(6, [0, 1], {1: [1, 0], 2: [1, 0]}),
            ],
        }
        expected = expected_logic_audit(record)
        self.assertEqual(expected["audit_type"], "cycle_stability")
        self.assertEqual(expected["candidates"][1], ["same", "same", "same"])
        self.assertEqual(
            expected["candidates"][2],
            ["same", "different", "same"],
        )

        response = json.dumps(
            {
                "audit_type": "cycle_stability",
                "candidates": [
                    {
                        "candidate": 1,
                        "relative_to_cycle_1": ["same", "same", "same"],
                    },
                    {
                        "candidate": 2,
                        "relative_to_cycle_1": ["same", "different", "same"],
                    },
                ],
            }
        )
        self.assertTrue(score_logic_audit(record, response)["exact_match"])

    def test_scene3_requires_exact_lag_for_every_candidate(self):
        commands = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        delayed = [[0, 0], [0, 0], commands[0], commands[1]]
        record = {
            "episode_id": "scene3-test",
            "scenario": "scene3_dyad_causal_identification",
            "steps": [
                _step(
                    index + 1,
                    command,
                    {1: command, 2: delayed[index]},
                )
                for index, command in enumerate(commands)
            ],
        }
        expected = expected_logic_audit(record)
        self.assertEqual(expected["candidates"], {1: 0, 2: 2})

        response = (
            "```json\n"
            '{"audit_type":"temporal_lag","candidates":'
            '[{"candidate":1,"lag":0},{"candidate":2,"lag":2}]}\n'
            "```"
        )
        self.assertIsInstance(parse_json_object(response), dict)
        self.assertTrue(score_logic_audit(record, response)["exact_match"])

        missing = (
            '{"audit_type":"temporal_lag","candidates":'
            '[{"candidate":1,"lag":0}]}'
        )
        score = score_logic_audit(record, missing)
        self.assertFalse(score["response_valid"])
        self.assertFalse(score["exact_match"])

    def test_invalid_or_unclear_output_fails(self):
        record = {
            "episode_id": "scene1-invalid",
            "scenario": "scene1_single_command_causality",
            "steps": [_step(1, [1], {1: [1]})],
        }
        self.assertFalse(score_logic_audit(record, "not json")["exact_match"])
        unclear = (
            '{"audit_type":"step_match","candidates":'
            '[{"candidate":1,"step_relations":["unclear"]}]}'
        )
        self.assertFalse(score_logic_audit(record, unclear)["exact_match"])

    def test_summary_withholds_final_accuracy_until_all_audits_finish(self):
        base = [
            {"episode_id": "a", "correct": True},
            {"episode_id": "b", "correct": True},
            {"episode_id": "c", "correct": False},
        ]
        partial = [{"episode_id": "a", "logic_audit_pass": True}]
        metrics = summarize_logic_audit(base, partial)
        self.assertFalse(metrics["logic_audit_complete"])
        self.assertIsNone(metrics["logic_adjusted_accuracy"])
        self.assertAlmostEqual(
            metrics["logic_adjusted_accuracy_lower_bound"],
            1 / 3,
            places=6,
        )

        complete = partial + [{"episode_id": "b", "logic_audit_pass": False}]
        metrics = summarize_logic_audit(base, complete)
        self.assertTrue(metrics["logic_audit_complete"])
        self.assertAlmostEqual(metrics["logic_adjusted_accuracy"], 1 / 3, places=6)
        self.assertEqual(metrics["logic_audit_pass_rate_among_initially_correct"], 0.5)


if __name__ == "__main__":
    unittest.main()
