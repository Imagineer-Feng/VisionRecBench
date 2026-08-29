import unittest
from collections import Counter

import numpy as np

from source.multimodal import build_prompts
from source.agent import parse_choice
from source.preprocess import construct
from source.task_logic import (
    build_mismatched_command_schedule,
    configure_binary_answers,
    select_behavior_option,
)


class Scene1LogicTest(unittest.TestCase):
    def setUp(self):
        self.task = construct({"scenario": "scene1_single_command_causality"})
        self.behavior_options = self.task["visible_arm_behavior_options"]
        self.commands = np.asarray(
            [item["delta"] for item in self.task["command_sequence"]],
            dtype=float,
        )

    def test_legacy_name_resolves_to_new_scene(self):
        task = construct({"scenario": "scene1_single_direct_or_random"})
        self.assertEqual(task["name"], "scene1_single_command_causality")
        self.assertEqual(
            task["requested_name"],
            "scene1_single_direct_or_random",
        )
        negative = construct({"scenario": "scene1_single_random"})
        self.assertEqual(negative["name"], "scene1_single_deranged")
        self.assertEqual(negative["requested_name"], "scene1_single_random")

    def test_four_consecutive_seeds_balance_condition_and_answer_position(self):
        combinations = []
        for seed in range(4):
            option_index = select_behavior_option(
                self.behavior_options,
                seed=seed,
                strategy=self.task["behavior_selection"],
            )
            target_present = self.behavior_options[option_index]["target_present"]
            _, answer_index = configure_binary_answers(
                self.task["answer_options"],
                target_present=target_present,
                seed=seed,
                shuffle=self.task["shuffle_answer_options"],
            )
            combinations.append((target_present, answer_index))

        self.assertCountEqual(
            combinations,
            [(True, 1), (True, 2), (False, 1), (False, 2)],
        )

    def test_difficulty_levels_have_exact_mismatch_counts(self):
        for level, expected_mismatches in ((1, 8), (2, 6), (3, 4)):
            task = construct(
                {"scenario": "scene1_single_command_causality", "level": level}
            )
            behavior = task["visible_arm_behavior_options"][1]["behavior"]
            self.assertEqual(behavior["mismatch_count"], expected_mismatches)
            for seed in range(16):
                schedule = build_mismatched_command_schedule(
                    self.commands,
                    episode_steps=task["episode_steps"],
                    seed=seed,
                    mismatch_count=behavior["mismatch_count"],
                )
                observed = sum(
                    not np.array_equal(expected, applied)
                    for expected, applied in zip(self.commands, schedule)
                )
                self.assertEqual(observed, expected_mismatches)

    def test_deranged_schedule_preserves_command_multiset_and_motion_budget(self):
        schedule = build_mismatched_command_schedule(
            self.commands,
            episode_steps=self.task["episode_steps"],
            seed=3,
            mismatch_count=6,
        )
        expected_counts = Counter(tuple(command) for command in self.commands)
        applied_counts = Counter(tuple(command) for command in schedule)
        self.assertEqual(applied_counts, expected_counts)
        np.testing.assert_array_equal(
            np.sum(schedule, axis=0),
            np.sum(self.commands, axis=0),
        )

    def test_scene_judges_once_after_complete_episode(self):
        self.assertEqual(self.task["episode_steps"], 8)
        self.assertEqual(self.task["judge_start_step"], 8)
        self.assertEqual(self.task["judge_interval_steps"], 8)
        self.assertEqual(
            self.task["visual_history_mode"],
            "workspace_motion_panels",
        )
        self.assertFalse(self.task["annotate_candidates"])

    def test_choice_form_contains_direct_and_mismatched_candidates(self):
        task = construct(
            {
                "scenario": "scene1_single_command_causality",
                "test_type": "choice",
                "level": 2,
            }
        )
        self.assertEqual(task["task_mode"], "multi_arm")
        self.assertEqual(task["num_arms"], 2)
        self.assertEqual(task["target_behavior"]["behavior"], "direct")
        self.assertEqual(task["distractors"][0]["behavior"], "sequence_mismatch")
        self.assertEqual(task["distractors"][0]["mismatch_count"], 6)
        self.assertTrue(task["annotate_candidates"])

    def test_prompt_is_neutral_and_has_no_fixed_option_number(self):
        prefix, suffix = build_prompts(
            list(reversed(self.task["answer_options"]))
        )
        prompt = prefix + suffix
        self.assertIn("complete motor-command trace", prompt)
        self.assertNotIn("mismatch", prompt.lower())
        self.assertNotIn("difficulty", prompt.lower())
        self.assertNotIn("Option 1 means", prompt)

    def test_markdown_choice_format_is_parseable(self):
        self.assertEqual(parse_choice("**Choice:** [1]"), 1)
        self.assertEqual(parse_choice("### Choice:\n[2]"), 2)


if __name__ == "__main__":
    unittest.main()
