import argparse
import copy
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from source.environment_config import ENVIRONMENT_TEMPLATE_IDS  # noqa: E402
from source.episode_sampling import STANDARD_SCENARIOS, build_episode_task  # noqa: E402
from source.multimodal import annotate_candidates, save_rgb  # noqa: E402
from source.render_config import RENDER_CONFIG  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render one initial-observation preview per environment template."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "previews" / "environment_templates",
    )
    parser.add_argument(
        "--scenario",
        choices=STANDARD_SCENARIOS,
        default="scene1_single_command_causality",
    )
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=1)
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

    rendered_templates = set()
    try:
        pair_index = 0
        while rendered_templates != set(ENVIRONMENT_TEMPLATE_IDS):
            task = build_episode_task(
                args.scenario,
                episode_index=pair_index * 2,
                level=args.level,
                base_seed=args.base_seed,
            )
            template_id = task["environment"]["id"]
            pair_index += 1
            if template_id in rendered_templates:
                continue

            task = apply_render_config(task)
            env = None
            try:
                env = VisionRecBenchEnv(simulation_app, task)
                image = env.reset()
                if task.get("annotate_candidates", True):
                    image = annotate_candidates(image, int(task["num_arms"]))
                output_path = args.output / f"{template_id}.png"
                save_rgb(image, output_path)
                print(f"saved {output_path}", flush=True)
                rendered_templates.add(template_id)
            finally:
                if env is not None:
                    env.close(close_simulation_app=False)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
