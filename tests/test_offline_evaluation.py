import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.evaluate_dataset import (
    build_episode_content,
    content_sha256,
    evaluate_one,
    result_path,
)
from source.agent import (
    DEFAULT_DECODING_PROFILE,
    EVALUATION_PROTOCOL_VERSION,
    IMAGE_DETAIL,
    OPENAI_GENERATION_PARAMETERS,
    OpenAIIdentifier,
    RandomIdentifier,
    evaluation_protocol_version,
    generation_parameters_for_profile,
    is_retryable_api_error,
    parse_choice,
)
from source.multimodal import build_prompts, get_control_labels
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
                "difficulty_level": 1,
                "difficulty_name": "easy",
                "test_type": "judgment",
                "task": task,
                "control_labels": get_control_labels(task),
                "visual_history_mode": "workspace_motion_panels",
                "target_present": True,
                "target_index": 1,
                "answer_index": answer_index,
                "answer_text": answer_options[answer_index - 1],
                "answer_options": answer_options,
                "steps": steps,
            }
            content = build_episode_content(
                record,
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
            )
            self.assertTrue(result["valid"])
            self.assertIn(result["choice"], (1, 2))
            self.assertEqual(result["input_content_sha256"], content_sha256(content))
            self.assertEqual(
                result["evaluation_protocol_version"],
                EVALUATION_PROTOCOL_VERSION,
            )
            self.assertIsNone(result["generation_parameters"])
            self.assertEqual(
                result["decoding_profile"],
                DEFAULT_DECODING_PROFILE,
            )

    def test_openai_images_request_explicit_high_detail(self):
        content = OpenAIIdentifier._to_openai_content(
            None,
            [np.zeros((24, 24, 3), dtype=np.uint8)],
        )

        self.assertEqual(content[0]["image_url"]["detail"], IMAGE_DETAIL)

    def test_option_wording_is_parsed_as_requested_by_prompt(self):
        self.assertEqual(parse_choice("Choice: [Option 1]"), 1)
        self.assertEqual(parse_choice("Choice: [2]"), 2)

    def test_prompt_requests_choice_before_explanation(self):
        _, suffix = build_prompts(["yes", "no"])

        self.assertLess(suffix.index("Choice:"), suffix.index("Thought:"))

    def test_compatible_request_and_response_metadata_are_frozen(self):
        captured = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    id="chatcmpl-test",
                    model="gpt-4o-2024-11-20",
                    created=123,
                    system_fingerprint="fp_test",
                    service_tier="default",
                    usage=SimpleNamespace(
                        model_dump=lambda mode: {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        }
                    ),
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="Choice: [1]"),
                            finish_reason="stop",
                        )
                    ],
                )

        identifier = object.__new__(OpenAIIdentifier)
        identifier.model = "gpt-4o"
        identifier.generation_parameters = dict(OPENAI_GENERATION_PARAMETERS)
        identifier.client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        )

        text, metadata = identifier._generate([], 2)

        self.assertEqual(text, "Choice: [1]")
        self.assertNotIn("temperature", captured)
        self.assertEqual(captured["max_completion_tokens"], 4096)
        self.assertEqual(metadata["response_model"], "gpt-4o-2024-11-20")
        self.assertEqual(metadata["system_fingerprint"], "fp_test")
        self.assertEqual(metadata["token_usage"]["total_tokens"], 15)

    def test_temperature_zero_is_an_explicit_separate_protocol(self):
        parameters = generation_parameters_for_profile("temperature_zero")

        self.assertEqual(parameters["temperature"], 0.0)
        self.assertEqual(parameters["max_completion_tokens"], 4096)
        self.assertNotEqual(
            evaluation_protocol_version("temperature_zero"),
            EVALUATION_PROTOCOL_VERSION,
        )

    def test_default_protocol_is_versioned_for_long_choice_first_responses(self):
        self.assertEqual(EVALUATION_PROTOCOL_VERSION, "eval_v3_compatible")

    def test_parameter_errors_fail_without_profile_fallback(self):
        self.assertFalse(
            is_retryable_api_error(SimpleNamespace(status_code=400))
        )
        self.assertFalse(
            is_retryable_api_error(SimpleNamespace(status_code=401))
        )
        self.assertTrue(
            is_retryable_api_error(SimpleNamespace(status_code=429))
        )
        self.assertTrue(
            is_retryable_api_error(SimpleNamespace(status_code=503))
        )

    def test_protocol_version_isolated_in_result_path(self):
        path = result_path(
            Path("results"),
            {"dataset_name": "dataset"},
            "gpt-4o",
            {
                "difficulty_level": 1,
                "test_type": "choice",
                "scenario": "scene1",
                "episode_id": "episode-1",
            },
        )

        self.assertIn(EVALUATION_PROTOCOL_VERSION, path.parts)


if __name__ == "__main__":
    unittest.main()
