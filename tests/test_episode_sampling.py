import copy
import unittest
from collections import Counter

import numpy as np

from source.difficulty import DIFFICULTY_LEVELS
from source.environment_config import ENVIRONMENT_TEMPLATE_IDS
from source.episode_sampling import (
    STANDARD_SCENARIOS,
    build_episode_task,
)
from source.preprocess import construct
from source.task_logic import (
    build_multi_arm_role_assignment,
    configure_binary_answers,
    materialize_mapping_behavior,
    select_behavior_option,
)


class EpisodeSamplingTest(unittest.TestCase):
    NUISANCE_FIELDS = (
        "command_sequence",
        "camera_eye",
        "camera_target",
        "camera_focal",
        "key_light_intensity",
        "key_light_rotation",
        "dome_light_intensity",
        "fill_light_intensity",
        "floor_color",
        "background_color",
        "environment",
    )

    def test_sampling_is_deterministic(self):
        first = build_episode_task(STANDARD_SCENARIOS[0], 7, base_seed=19)
        second = build_episode_task(STANDARD_SCENARIOS[0], 7, base_seed=19)
        self.assertEqual(first, second)

    def test_robust_split_has_unique_episode_signatures(self):
        for scenario in STANDARD_SCENARIOS:
            tasks = [
                build_episode_task(scenario, index, level=level, base_seed=0)
                for level in DIFFICULTY_LEVELS
                for index in range(48)
            ]
            signatures = {task["episode_signature"] for task in tasks}
            self.assertEqual(len(signatures), 144)

    def test_difficulty_levels_share_nuisance_variations(self):
        for scenario in STANDARD_SCENARIOS:
            tasks = [
                build_episode_task(scenario, 7, level=level, base_seed=0)
                for level in DIFFICULTY_LEVELS
            ]
            self.assertEqual(len({task["nuisance_signature"] for task in tasks}), 1)
            self.assertEqual(len({task["nuisance_pair_id"] for task in tasks}), 1)
            for key in self.NUISANCE_FIELDS:
                self.assertEqual(tasks[0][key], tasks[1][key])
                self.assertEqual(tasks[1][key], tasks[2][key])
            self.assertEqual(
                tasks[0]["arm"]["initial_joint_positions"],
                tasks[1]["arm"]["initial_joint_positions"],
            )
            self.assertEqual(
                tasks[1]["arm"]["initial_joint_positions"],
                tasks[2]["arm"]["initial_joint_positions"],
            )

    def test_opposite_conditions_share_exact_nuisance_pair(self):
        for scenario in STANDARD_SCENARIOS:
            for level in DIFFICULTY_LEVELS:
                first = build_episode_task(
                    scenario, 10, level=level, base_seed=19
                )
                second = build_episode_task(
                    scenario, 11, level=level, base_seed=19
                )
                self.assertEqual(first["nuisance_pair_id"], second["nuisance_pair_id"])
                self.assertEqual(
                    first["nuisance_signature"], second["nuisance_signature"]
                )
                self.assertEqual(
                    first["episode_variation"]["episode_seed"] // 2,
                    second["episode_variation"]["episode_seed"] // 2,
                )
                for key in self.NUISANCE_FIELDS:
                    self.assertEqual(first[key], second[key])
                self.assertEqual(
                    first["arm"]["initial_joint_positions"],
                    second["arm"]["initial_joint_positions"],
                )
                first_neutral = copy.deepcopy(first)
                second_neutral = copy.deepcopy(second)
                for key in (
                    "seed",
                    "episode_variation",
                    "nuisance_pair_id",
                    "nuisance_signature",
                    "episode_signature",
                    "episode_id",
                ):
                    first_neutral.pop(key)
                    second_neutral.pop(key)
                self.assertEqual(first_neutral, second_neutral)

                if first["task_mode"] == "single_binary":
                    target_presence = set()
                    answer_orders = []
                    sampled_mapping_options = []
                    for task in (first, second):
                        options = task["visible_arm_behavior_options"]
                        option_index = select_behavior_option(
                            options,
                            seed=task["seed"],
                            strategy=task["behavior_selection"],
                        )
                        target_presence.add(
                            bool(options[option_index]["target_present"])
                        )
                        answer_order, _ = configure_binary_answers(
                            task["answer_options"],
                            bool(options[option_index]["target_present"]),
                            task["seed"],
                            shuffle=True,
                        )
                        answer_orders.append(answer_order)
                        behavior = materialize_mapping_behavior(
                            options[option_index]["behavior"],
                            seed=task["seed"],
                            behavior_option_count=len(options),
                        )
                        if "sampled_mapping_option" in behavior:
                            sampled_mapping_options.append(
                                behavior["sampled_mapping_option"]
                            )
                    self.assertEqual(target_presence, {False, True})
                    self.assertEqual(answer_orders[0], answer_orders[1])
                    if sampled_mapping_options:
                        self.assertEqual(len(set(sampled_mapping_options)), 1)
                else:
                    target_positions = {
                        next(
                            assignment["index"]
                            for assignment in build_multi_arm_role_assignment(
                                task["num_arms"],
                                task["distractors"],
                                task["seed"],
                            )
                            if assignment["role"] == "target"
                        )
                        for task in (first, second)
                    }
                    self.assertEqual(target_positions, {1, 2})

    def test_different_pairs_use_different_nuisance_variations(self):
        for scenario in STANDARD_SCENARIOS:
            first = build_episode_task(scenario, 10, base_seed=19)
            next_pair = build_episode_task(scenario, 12, base_seed=19)
            self.assertNotEqual(
                first["nuisance_pair_id"], next_pair["nuisance_pair_id"]
            )
            self.assertNotEqual(
                first["nuisance_signature"], next_pair["nuisance_signature"]
            )

    def test_environment_templates_are_balanced_within_every_condition(self):
        for scenario in STANDARD_SCENARIOS:
            for level in DIFFICULTY_LEVELS:
                counts = Counter()
                for index in range(48):
                    task = build_episode_task(
                        scenario,
                        index,
                        level=level,
                        base_seed=19,
                    )
                    if task["task_mode"] == "single_binary":
                        options = task["visible_arm_behavior_options"]
                        selected = select_behavior_option(
                            options,
                            task["seed"],
                            task["behavior_selection"],
                        )
                        condition = bool(options[selected]["target_present"])
                    else:
                        assignments = build_multi_arm_role_assignment(
                            task["num_arms"],
                            task["distractors"],
                            task["seed"],
                        )
                        condition = next(
                            assignment["index"]
                            for assignment in assignments
                            if assignment["role"] == "target"
                        )
                    counts[(condition, task["environment"]["id"])] += 1

                self.assertEqual(
                    {template for _, template in counts},
                    set(ENVIRONMENT_TEMPLATE_IDS),
                )
                self.assertEqual(set(counts.values()), {8})

    def test_sampling_changes_commands_pose_camera_and_lighting(self):
        scenario = "scene3_dyad_causal_identification"
        base = construct({"scenario": scenario})
        sampled = build_episode_task(scenario, 0, base_seed=0)
        self.assertNotEqual(
            sampled["command_sequence"],
            base["command_sequence"],
        )
        self.assertNotEqual(
            sampled["arm"]["initial_joint_positions"],
            base["arm"]["initial_joint_positions"],
        )
        self.assertNotEqual(sampled["camera_eye"], base["camera_eye"])
        self.assertNotEqual(
            sampled["key_light_intensity"],
            base["key_light_intensity"],
        )

    def test_sampled_initial_poses_stay_inside_joint_limits(self):
        for scenario in STANDARD_SCENARIOS:
            for index in range(24):
                task = build_episode_task(scenario, index, base_seed=4)
                initial = np.asarray(
                    task["arm"]["initial_joint_positions"],
                    dtype=float,
                )
                limits = np.asarray(task["arm"]["joint_limits"], dtype=float)
                self.assertTrue(np.all(initial >= limits[:, 0]))
                self.assertTrue(np.all(initial <= limits[:, 1]))

    def test_binary_conditions_and_answer_positions_are_balanced(self):
        for scenario in STANDARD_SCENARIOS[:2]:
            for level in DIFFICULTY_LEVELS:
                combinations = Counter()
                mapping_options = Counter()
                for index in range(48):
                    task = build_episode_task(
                        scenario, index, level=level, base_seed=0
                    )
                    options = task["visible_arm_behavior_options"]
                    selected_index = select_behavior_option(
                        options,
                        seed=task["seed"],
                        strategy=task["behavior_selection"],
                    )
                    selected = options[selected_index]
                    _, answer_index = configure_binary_answers(
                        task["answer_options"],
                        selected["target_present"],
                        task["seed"],
                        shuffle=True,
                    )
                    combinations[(selected["target_present"], answer_index)] += 1
                    behavior = materialize_mapping_behavior(
                        selected["behavior"],
                        seed=task["seed"],
                        behavior_option_count=len(options),
                    )
                    if "sampled_mapping_option" in behavior:
                        mapping_options[
                            (
                                selected["target_present"],
                                behavior["sampled_mapping_option"],
                            )
                        ] += 1

                self.assertEqual(set(combinations.values()), {12})
                if mapping_options:
                    self.assertEqual(set(mapping_options.values()), {6})

    def test_scene3_target_and_distractor_roles_are_balanced(self):
        positions = Counter()
        behavior_positions = Counter()
        for level in DIFFICULTY_LEVELS:
            for index in range(48):
                task = build_episode_task(
                    "scene3_dyad_causal_identification",
                    index,
                    level=level,
                    base_seed=0,
                )
                assignments = build_multi_arm_role_assignment(
                    task["num_arms"],
                    task["distractors"],
                    task["seed"],
                )
                for assignment in assignments:
                    behavior = assignment["behavior"]["behavior"]
                    behavior_positions[
                        (level, behavior, assignment["index"])
                    ] += 1
                    if assignment["role"] == "target":
                        positions[(level, assignment["index"])] += 1

        self.assertEqual(set(positions.values()), {24})
        self.assertEqual(set(behavior_positions.values()), {24})

    def test_scene2_preserves_one_repeatable_four_command_cycle(self):
        task = build_episode_task(
            "scene2_single_scrambled_stability",
            11,
            base_seed=0,
        )
        self.assertEqual(len(task["command_sequence"]), 4)
        supports = []
        for command in task["command_sequence"]:
            nonzero = np.flatnonzero(np.asarray(command["delta"]))
            self.assertEqual(len(nonzero), 1)
            supports.append(int(nonzero[0]))
        self.assertCountEqual(supports, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
