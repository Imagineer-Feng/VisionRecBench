import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from source.dataset_io import (
    DATASET_SCHEMA_VERSION,
    atomic_write_json,
    dataset_content_hash,
    image_descriptor,
    rebuild_manifest,
    validate_dataset,
)


class DatasetIOTest(unittest.TestCase):
    def _build_dataset(self, root):
        episode_id = "episode-00000-deadbeef"
        episode_dir = root / "episodes" / episode_id
        image_dir = episode_dir / "images"
        image_dir.mkdir(parents=True)
        initial_path = image_dir / "initial.png"
        observation_path = image_dir / "observation.png"
        evidence_path = image_dir / "evidence.png"
        Image.fromarray(np.full((16, 20, 3), 20, dtype=np.uint8)).save(initial_path)
        Image.fromarray(np.full((16, 20, 3), 40, dtype=np.uint8)).save(observation_path)
        Image.fromarray(np.full((16, 20, 3), 60, dtype=np.uint8)).save(evidence_path)

        record = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "episode_id": episode_id,
            "episode_index": 0,
            "episode_signature": "deadbeef",
            "scenario": "test_scenario",
            "scene": 1,
            "seed": 0,
            "difficulty_level": 1,
            "difficulty_name": "easy",
            "target_present": True,
            "target_index": 1,
            "answer_index": 1,
            "answer_text": "yes",
            "answer_options": ["yes", "no"],
            "task": {"episode_steps": 1},
            "initial_observation": image_descriptor(root, initial_path),
            "steps": [
                {
                    "step": 1,
                    "observation": image_descriptor(root, observation_path),
                    "evidence": image_descriptor(root, evidence_path),
                }
            ],
        }
        atomic_write_json(episode_dir / "episode.json", record)
        rows = rebuild_manifest(root)
        metadata = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_name": "test_dataset",
            "scenarios": ["test_scenario"],
            "difficulty_levels": [1],
            "episodes_per_scene_per_level": 1,
            "content_sha256": dataset_content_hash(rows),
        }
        atomic_write_json(root / "metadata.json", metadata)
        return record, evidence_path

    def test_complete_dataset_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._build_dataset(root)
            report = validate_dataset(root, verify_checksums=True)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["episode_count"], 1)

    def test_checksum_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, evidence_path = self._build_dataset(root)
            Image.fromarray(np.zeros((16, 20, 3), dtype=np.uint8)).save(evidence_path)
            report = validate_dataset(root, verify_checksums=True)
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("checksum mismatch" in error for error in report["errors"])
            )


if __name__ == "__main__":
    unittest.main()
