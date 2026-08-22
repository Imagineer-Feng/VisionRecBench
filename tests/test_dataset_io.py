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
    load_json,
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

    def _build_paired_dataset(self, root):
        pair_id = "test_scenario-base0-pair00000"
        signature = "shared-nuisance"
        record_paths = []
        for episode_index, target_present in enumerate((True, False)):
            episode_id = f"episode-{episode_index:05d}"
            episode_dir = root / "episodes" / episode_id
            image_dir = episode_dir / "images"
            image_dir.mkdir(parents=True)
            descriptors = {}
            for name, value in (
                ("initial", 20),
                ("observation", 40),
                ("evidence", 60),
            ):
                path = image_dir / f"{name}.png"
                Image.fromarray(
                    np.full((16, 20, 3), value, dtype=np.uint8)
                ).save(path)
                descriptors[name] = image_descriptor(root, path)

            answer_index = 1 if target_present else 2
            task = {
                "episode_steps": 1,
                "task_mode": "single_binary",
                "nuisance_pair_id": pair_id,
                "nuisance_signature": signature,
            }
            record = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "episode_id": episode_id,
                "episode_index": episode_index,
                "episode_signature": f"physical-{episode_index}",
                "scenario": "test_scenario",
                "scene": 1,
                "seed": episode_index,
                "difficulty_level": 1,
                "difficulty_name": "easy",
                "nuisance_pair_id": pair_id,
                "nuisance_signature": signature,
                "target_present": target_present,
                "target_index": 1 if target_present else None,
                "answer_index": answer_index,
                "answer_text": ["yes", "no"][answer_index - 1],
                "answer_options": ["yes", "no"],
                "task": task,
                "initial_observation": descriptors["initial"],
                "steps": [
                    {
                        "step": 1,
                        "observation": descriptors["observation"],
                        "evidence": descriptors["evidence"],
                    }
                ],
            }
            record_path = episode_dir / "episode.json"
            atomic_write_json(record_path, record)
            record_paths.append(record_path)

        rows = rebuild_manifest(root)
        metadata = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "dataset_name": "paired_test_dataset",
            "scenarios": ["test_scenario"],
            "difficulty_levels": [1],
            "episodes_per_scene_per_level": 2,
            "paired_nuisance": True,
            "nuisance_pair_size": 2,
            "content_sha256": dataset_content_hash(rows),
        }
        atomic_write_json(root / "metadata.json", metadata)
        return record_paths

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

    def test_paired_dataset_requires_pair_metadata_and_complete_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._build_dataset(root)
            metadata_path = root / "metadata.json"
            metadata = load_json(metadata_path)
            metadata["paired_nuisance"] = True
            metadata["nuisance_pair_size"] = 2
            atomic_write_json(metadata_path, metadata)

            report = validate_dataset(root, verify_checksums=False)
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("missing nuisance pair ID" in error for error in report["errors"])
            )

    def test_complete_paired_dataset_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._build_paired_dataset(root)
            report = validate_dataset(root, verify_checksums=True)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["nuisance_pair_groups"], 1)

    def test_pair_signature_mismatch_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record_paths = self._build_paired_dataset(root)
            record = load_json(record_paths[1])
            record["nuisance_signature"] = "different-nuisance"
            record["task"]["nuisance_signature"] = "different-nuisance"
            atomic_write_json(record_paths[1], record)
            rows = rebuild_manifest(root)
            metadata_path = root / "metadata.json"
            metadata = load_json(metadata_path)
            metadata["content_sha256"] = dataset_content_hash(rows)
            atomic_write_json(metadata_path, metadata)

            report = validate_dataset(root, verify_checksums=False)
            self.assertFalse(report["valid"])
            self.assertTrue(
                any(
                    "nuisance signature differs" in error
                    for error in report["errors"]
                )
            )


if __name__ == "__main__":
    unittest.main()
