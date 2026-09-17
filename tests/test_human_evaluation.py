import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_human import (
    HUMAN_PROTOCOL_VERSION,
    HumanEvaluationStudy,
    build_balanced_plan,
    create_human_result,
    human_result_path,
    public_trial_payload,
    trial_token,
)


class HumanEvaluationTest(unittest.TestCase):
    def test_balanced_plan_is_deterministic_and_interleaves_cells(self):
        rows = []
        for scene in (1, 2):
            for index in range(4):
                rows.append(
                    {
                        "episode_id": f"scene-{scene}-{index}",
                        "scene": scene,
                        "difficulty_level": 1,
                        "test_type": "choice",
                        "nuisance_pair_id": f"pair-{index}",
                    }
                )
        first = build_balanced_plan(rows, seed=7)
        second = build_balanced_plan(rows, seed=7)

        self.assertEqual(
            [row["episode_id"] for row in first],
            [row["episode_id"] for row in second],
        )
        for offset in range(0, len(first), 2):
            self.assertEqual(
                {row["scene"] for row in first[offset:offset + 2]},
                {1, 2},
            )

    def test_public_payload_does_not_expose_ground_truth_or_episode_metadata(self):
        record = self._record()
        payload = public_trial_payload(record, "opaque-token", 0, 1, 36)
        serialized = json.dumps(payload)

        for secret in (
            record["episode_id"],
            record["scenario"],
            record["nuisance_pair_id"],
            "answer_index",
            "target_present",
            "difficulty_level",
            "nuisance_pair_id",
            "episode_id",
            "scenario",
        ):
            self.assertNotIn(str(secret), serialized)
        self.assertIn("/media/opaque-token/evidence/1", serialized)
        self.assertEqual(len(payload["answer_options"]), 2)

    def test_submission_is_atomic_and_resumes_from_next_trial(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            dataset.mkdir()
            records = [self._record("episode-1"), self._record("episode-2")]
            manifest = []
            for record in records:
                path = dataset / f"{record['episode_id']}.json"
                path.write_text(json.dumps(record), encoding="utf-8")
                manifest.append(
                    {
                        "episode_id": record["episode_id"],
                        "record_path": path.name,
                        "scenario": record["scenario"],
                        "difficulty_level": record["difficulty_level"],
                        "test_type": record["test_type"],
                    }
                )
            metadata = {
                "dataset_name": "dataset",
                "content_sha256": "dataset-hash",
            }
            plan = {
                "plan_id": "plan-hash",
                "participant_id": "human-001",
                "session_size": 1,
                "episode_ids": ["episode-1", "episode-2"],
            }
            output = root / "output"
            study = HumanEvaluationStudy(
                dataset,
                metadata,
                manifest,
                plan,
                output,
                input_hash_fn=lambda record: f"input-{record['episode_id']}",
            )
            token = trial_token("plan-hash", 1, "episode-1")
            status = study.submit(token, 1, 1200, 1100)

            self.assertEqual(status["completed"], 1)
            self.assertTrue(status["break_due"])
            saved = json.loads(
                human_result_path(output, records[0]).read_text(encoding="utf-8")
            )
            self.assertEqual(saved["model"], "human:human-001")
            self.assertEqual(saved["evaluation_protocol_version"], HUMAN_PROTOCOL_VERSION)
            self.assertEqual(saved["input_content_sha256"], "input-episode-1")
            self.assertTrue(saved["correct"])

            resumed = HumanEvaluationStudy(
                dataset,
                metadata,
                manifest,
                plan,
                output,
                input_hash_fn=lambda record: f"input-{record['episode_id']}",
            )
            self.assertEqual(resumed.completed_count(), 1)
            self.assertEqual(resumed.current_payload()["progress"]["current"], 2)

    def test_human_result_contains_fields_used_by_offline_summarizer(self):
        row = create_human_result(
            self._record(),
            {"dataset_name": "dataset", "content_sha256": "hash"},
            "human-001",
            "plan-id",
            1,
            36,
            1,
            1000,
            900,
            "input-hash",
        )
        required = {
            "dataset_content_sha256",
            "episode_id",
            "model",
            "scenario",
            "difficulty_level",
            "test_type",
            "correct",
            "valid",
        }
        self.assertTrue(required.issubset(row))

    @staticmethod
    def _record(episode_id="episode-1"):
        return {
            "episode_id": episode_id,
            "episode_signature": f"signature-{episode_id}",
            "scenario": "hidden-scenario",
            "scene": 1,
            "seed": 0,
            "difficulty_level": 1,
            "difficulty_name": "easy",
            "test_type": "choice",
            "nuisance_pair_id": "hidden-pair",
            "nuisance_signature": "hidden-signature",
            "environment_template": "lab",
            "arm_type": "arm",
            "camera_view": "front",
            "control_labels": ["axis_1"],
            "task": {"command_sequence": [{"delta": [1]}], "arm": {}},
            "steps": [
                {
                    "command": {"step": 1, "name": "move", "delta": [1]},
                    "evidence": {"path": "evidence.png"},
                    "observation": {"path": "observation.png"},
                }
            ],
            "answer_options": ["candidate 1", "candidate 2"],
            "answer_index": 1,
            "answer_text": "candidate 1",
            "target_present": True,
            "target_index": 1,
        }


if __name__ == "__main__":
    unittest.main()
