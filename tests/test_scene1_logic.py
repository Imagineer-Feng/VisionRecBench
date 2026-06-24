import unittest
from collections import Counter

import numpy as np

from scripts.inference import build_prompts
from source.agent import parse_choice
from source.preprocess import construct
from source.task_logic import (
    build_deranged_command_schedule,
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

    def test_deranged_schedule_has_no_same_step_matches(self):
        for seed in range(32):
            schedule = build_deranged_command_schedule(
                self.commands,
                episode_steps=self.task["episode_steps"],
                seed=seed,
            )
            for expected, applied in zip(self.commands, schedule):
                self.assertFalse(np.array_equal(expected, applied))

    def test_deranged_schedule_preserves_command_multiset_and_motion_budget(self):
        schedule = build_deranged_command_schedule(
            self.commands,
            episode_steps=self.task["episode_steps"],
            seed=3,
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
        self.assertEqual(self.task["visual_history_mode"], "motion_diffs")
        self.assertFalse(self.task["annotate_candidates"])

    def test_prompt_is_neutral_and_has_no_fixed_option_number(self):
        prefix, suffix = build_prompts(
            1,
            self.task,
            max_image_history=8,
            answer_options=list(reversed(self.task["answer_options"])),
        )
        prompt = prefix + suffix
        self.assertIn("mismatched command stream", prompt)
        self.assertIn("same-step", prompt)
        self.assertIn("motion-difference images", prompt)
        self.assertNotIn("Option 1 means", prompt)

    def test_markdown_choice_format_is_parseable(self):
        self.assertEqual(parse_choice("**Choice:** [1]"), 1)
        self.assertEqual(parse_choice("### Choice:\n[2]"), 2)


if __name__ == "__main__":
    unittest.main()
