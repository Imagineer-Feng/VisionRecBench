import unittest

import numpy as np

from scripts.inference import build_prompts
from source.preprocess import construct
from source.task_logic import (
    apply_mapped_behavior,
    configure_binary_answers,
    materialize_mapping_behavior,
    select_behavior_option,
)


class Scene2LogicTest(unittest.TestCase):
    def setUp(self):
        self.task = construct({"scenario": "scene2_single_scrambled_stability"})
        self.behavior_options = self.task["visible_arm_behavior_options"]

    def test_legacy_name_resolves_to_new_scene(self):
        task = construct({"scenario": "scene2_single_scrambled_fixed"})
        self.assertEqual(task["name"], "scene2_single_scrambled_stability")
        self.assertEqual(task["requested_name"], "scene2_single_scrambled_fixed")

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

    def test_mapping_options_are_deranged_permutations(self):
        mappings = self.behavior_options[0]["behavior"]["mapping_options"]
        for mapping in mappings:
            matrix = np.asarray(mapping)
            np.testing.assert_array_equal(matrix.sum(axis=0), np.ones(4))
            np.testing.assert_array_equal(matrix.sum(axis=1), np.ones(4))
            np.testing.assert_array_equal(np.diag(matrix), np.zeros(4))

    def test_stable_mapping_repeats_same_response(self):
        behavior = materialize_mapping_behavior(
            self.behavior_options[0]["behavior"],
            seed=0,
            behavior_option_count=2,
        )
        command = np.array([1.0, 0.0, 0.0, 0.0])
        responses = [
            apply_mapped_behavior(behavior, command, step, command_dim=4)
            for step in (0, 4, 8)
        ]
        np.testing.assert_array_equal(responses[0], responses[1])
        np.testing.assert_array_equal(responses[1], responses[2])

    def test_unstable_mapping_changes_every_cycle(self):
        behavior = materialize_mapping_behavior(
            self.behavior_options[1]["behavior"],
            seed=1,
            behavior_option_count=2,
        )
        for axis in range(4):
            command = np.eye(4)[axis]
            responses = [
                tuple(apply_mapped_behavior(behavior, command, step, command_dim=4))
                for step in (0, 4, 8)
            ]
            self.assertEqual(len(set(responses)), 3)

    def test_each_cycle_has_same_total_motion_budget(self):
        for option_index, option in enumerate(self.behavior_options):
            behavior = materialize_mapping_behavior(
                option["behavior"],
                seed=option_index,
                behavior_option_count=2,
            )
            cycle_totals = []
            for cycle_index in range(3):
                responses = [
                    apply_mapped_behavior(
                        behavior,
                        np.eye(4)[axis],
                        command_index=cycle_index * 4 + axis,
                        command_dim=4,
                    )
                    for axis in range(4)
                ]
                cycle_totals.append(np.sum(responses, axis=0))

            for total in cycle_totals:
                np.testing.assert_array_equal(total, np.ones(4))

    def test_prompt_describes_both_conditions_without_fixed_answer_number(self):
        prefix, suffix = build_prompts(
            1,
            self.task,
            max_image_history=12,
            answer_options=list(reversed(self.task["answer_options"])),
        )
        prompt = prefix + suffix
        self.assertIn("may remain stable", prompt)
        self.assertIn("may change between cycles", prompt)
        self.assertIn("Non-self case", prompt)
        self.assertNotIn("Option 1 means", prompt)
        self.assertNotIn("should still count as your own body", prompt)
        self.assertIn("motion-difference images", prompt)

    def test_scene_judges_once_after_three_complete_four_axis_cycles(self):
        self.assertEqual(self.task["episode_steps"], 12)
        self.assertEqual(self.task["judge_start_step"], 12)
        self.assertEqual(self.task["judge_interval_steps"], 4)
        self.assertEqual(self.task["visual_history_mode"], "motion_diffs")
        self.assertFalse(self.task["annotate_candidates"])
        commands = np.asarray(
            [item["delta"] for item in self.task["command_sequence"]]
        )
        np.testing.assert_array_equal(commands, np.eye(4))


if __name__ == "__main__":
    unittest.main()
