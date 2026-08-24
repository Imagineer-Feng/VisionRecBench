import argparse
import copy
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from source.environment_config import ENVIRONMENT_TEMPLATE_IDS  # noqa: E402
from source.episode_sampling import STANDARD_SCENARIOS, build_episode_task  # noqa: E402
from source.multimodal import annotate_candidates, save_rgb  # noqa: E402
from source.camera_config import CAMERA_VIEW_IDS  # noqa: E402
from source.preprocess import ARM_CONFIG_IDS, TEST_TYPES  # noqa: E402
from source.render_config import RENDER_CONFIG  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render one initial-observation preview per background x robot x "
            "camera nuisance combination."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "previews" / "diversity_matrix",
    )
    parser.add_argument(
        "--scenario",
        choices=STANDARD_SCENARIOS,
        default="scene1_single_command_causality",
    )
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--test-type", choices=TEST_TYPES, default="choice")
    parser.add_argument(
        "--environment-template",
        choices=ENVIRONMENT_TEMPLATE_IDS,
        default=None,
        help="Optional preview-only background filter.",
    )
    parser.add_argument(
        "--arm-type",
        choices=ARM_CONFIG_IDS,
        default=None,
        help="Optional preview-only robot filter; does not change dataset sampling.",
    )
    parser.add_argument(
        "--camera-view",
        choices=CAMERA_VIEW_IDS,
        default=None,
        help="Optional preview-only view filter; does not change dataset sampling.",
    )
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def apply_render_config(task):
    task = copy.deepcopy(task)
    task["anti_aliasing_op"] = RENDER_CONFIG["anti_aliasing"]
    task["pathtracing_spp"] = RENDER_CONFIG["pathtracing_spp"]
    task["denoiser_enabled"] = RENDER_CONFIG["denoiser"]
    return task


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": RENDER_CONFIG["renderer"],
            "width": RENDER_CONFIG["resolution"],
            "height": RENDER_CONFIG["resolution"],
            "anti_aliasing": RENDER_CONFIG["anti_aliasing"],
            "denoiser": RENDER_CONFIG["denoiser"],
        }
    )
    from source.env import VisionRecBenchEnv

    expected_combinations = {
        (template, arm_type, camera_view)
        for template in ENVIRONMENT_TEMPLATE_IDS
        for arm_type in ARM_CONFIG_IDS
        for camera_view in CAMERA_VIEW_IDS
        if args.environment_template in (None, template)
        if args.arm_type in (None, arm_type)
        and args.camera_view in (None, camera_view)
    }
    rendered_combinations = set()
    try:
        pair_index = 0
        while rendered_combinations != expected_combinations:
            task = build_episode_task(
                args.scenario,
                episode_index=pair_index * 2,
                level=args.level,
                test_type=args.test_type,
                base_seed=args.base_seed,
            )
            template_id = task["environment"]["id"]
            combination = (
                template_id,
                task["arm_type"],
                task["camera_view"],
            )
            pair_index += 1
            if args.environment_template not in (None, template_id):
                continue
            if args.arm_type not in (None, task["arm_type"]):
                continue
            if args.camera_view not in (None, task["camera_view"]):
                continue
            if combination in rendered_combinations:
                continue

            task = apply_render_config(task)
            env = None
            try:
                env = VisionRecBenchEnv(simulation_app, task)
                image = env.reset()
                arm_positions = [
                    arm["articulation"].get_world_pose()[0].tolist()
                    for arm in env.arms
                ]
                print(
                    f"arm root positions for {combination}: {arm_positions}",
                    flush=True,
                )
                if task.get("annotate_candidates", True):
                    image = annotate_candidates(image, int(task["num_arms"]))
                output_path = args.output / (
                    f"{template_id}__{task['arm_type']}__"
                    f"{task['camera_view']}.png"
                )
                save_rgb(image, output_path)
                print(f"saved {output_path}", flush=True)
                rendered_combinations.add(combination)
            finally:
                if env is not None:
                    env.close(close_simulation_app=False)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
