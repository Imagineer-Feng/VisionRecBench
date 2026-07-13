import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.evaluate_dataset import (
    build_episode_content,
    content_sha256,
    evaluate_one,
)
from scripts.inference import get_control_labels
from source.agent import RandomIdentifier
from source.preprocess import construct
from source.task_logic import configure_binary_answers


class OfflineEvaluationTest(unittest.TestCase):
    def test_frozen_episode_builds_and_scores_without_isaac_sim(self):
        task = construct({"scenario": "scene1_single_command_causality"})
        task["seed"] = 0
        answer_options, answer_index = configure_binary_answers(
            task["answer_options"],
            target_present=True,
            seed=0,
            shuffle=True,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            steps = []
            for step_index in range(1, task["episode_steps"] + 1):
                observation_path = root / f"observation-{step_index}.png"
                evidence_path = root / f"evidence-{step_index}.png"
                Image.fromarray(
                    np.full((24, 24, 3), 20 + step_index, dtype=np.uint8)
                ).save(observation_path)
                Image.fromarray(
                    np.full((24, 24, 3), 40 + step_index, dtype=np.uint8)
                ).save(evidence_path)
                command = dict(
                    task["command_sequence"][
                        (step_index - 1) % len(task["command_sequence"])
                    ]
                )
                command["step"] = step_index
                steps.append(
                    {
                        "step": step_index,
                        "command": command,
                        "observation": {"path": observation_path.name},
                        "evidence": {"path": evidence_path.name},
                    }
                )

            record = {
                "episode_id": "offline-smoke-test",
                "episode_signature": "cafebabe",
                "scenario": task["name"],
                "scene": task["scene"],
                "seed": 0,
                "task": task,
                "control_labels": get_control_labels(task),
                "visual_history_mode": "motion_diffs",
                "target_present": True,
                "target_index": 1,
                "answer_index": answer_index,
                "answer_text": answer_options[answer_index - 1],
                "answer_options": answer_options,
                "steps": steps,
            }
            content = build_episode_content(
                record,
                level=1,
                max_image_history=-1,
                dataset_root=root,
            )
            image_count = sum(isinstance(item, np.ndarray) for item in content)
            self.assertEqual(image_count, task["episode_steps"] + 1)
            self.assertEqual(content_sha256(content), content_sha256(content))

            random.seed(0)
            result = evaluate_one(
                RandomIdentifier(),
                record,
                content,
                {
                    "dataset_name": "test",
                    "content_sha256": "1234",
                },
                model="random",
                level=1,
            )
            self.assertTrue(result["valid"])
            self.assertIn(result["choice"], (1, 2))
            self.assertEqual(result["input_content_sha256"], content_sha256(content))


if __name__ == "__main__":
    unittest.main()
