import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from source.action import options_string
from source.agent import create_identifier
from source.preprocess import construct
from source.prompts import PROMPTS


INSTRUCTION_DICT = {
    "Robot": (
        "You are playing the role of a simulated robotic agent performing a "
        "visual self-recognition task inside a 3D Isaac Sim environment."
    )
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="triad_delay_invert")
    parser.add_argument("--arm", type=str, default=None)
    parser.add_argument("--level", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--model", type=str, required=True, help="Model name to use, or random")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum episode steps, -1 for scenario default")
    parser.add_argument("--max_image_history", type=int, default=4, help="Maximum previous images to keep")
    parser.add_argument("--target_index", type=int, default=None, help="Optional 1-based target candidate index")
    parser.add_argument("--seed", type=int, default=None, help="Override the scenario random seed")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim in headless mode")
    parser.add_argument(
        "--renderer",
        type=str,
        default="RayTracedLighting",
        choices=["RayTracedLighting", "PathTracing"],
        help="Isaac Sim renderer. RayTracedLighting is faster and safer on newer GPUs.",
    )
    parser.add_argument("--resolution", type=int, default=768, help="Square camera resolution in pixels")
    parser.add_argument("--warmup_frames", type=int, default=12, help="Rendered frames during reset")
    parser.add_argument("--render_frames", type=int, default=6, help="Rendered frames per motor command")
    args = parser.parse_args()
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max_steps must be -1 or a positive integer")
    if args.max_image_history < 0:
        parser.error("--max_image_history must be non-negative")
    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    if args.warmup_frames <= 0:
        parser.error("--warmup_frames must be positive")
    if args.render_frames <= 0:
        parser.error("--render_frames must be positive")
    return args


def validate_api_key(model):
    if model == "random":
        return
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY"):
        return
    raise SystemExit(
        "OPENAI_API_KEY is not set. Run `source scripts/setup_env.sh` "
        "or export it before starting evaluation."
    )


def build_task(args):
    task_dict = construct({"scenario": args.scenario, "arm": args.arm})
    if args.seed is not None:
        task_dict["seed"] = args.seed
    if args.target_index is not None:
        task_dict["target_index"] = args.target_index
    return task_dict


def create_run_paths(args, tag):
    model_name = args.model.replace("/", "-")
    result_dir = BASE_DIR / "results" / f"level{args.level}" / model_name / args.scenario
    result_dir.mkdir(parents=True, exist_ok=True)

    log_dir = BASE_DIR / "logs" / tag
    log_dir.mkdir(parents=True, exist_ok=False)

    obs_dir = log_dir / "obs"
    obs_dir.mkdir(parents=True, exist_ok=False)

    return {
        "result_file": result_dir / f"{tag}.json",
        "log_dir": log_dir,
        "log_file": log_dir / "logs_env.txt",
        "log_file_agent": log_dir / "logs_agent.txt",
        "obs_dir": obs_dir,
        "args_file": log_dir / "args.json",
    }


def build_prompts(level, task_dict, max_image_history):
    track = task_dict["track"]
    distractor_summary = "\n".join(
        f"- {item['desc']}" for item in task_dict["distractors"]
    )
    prompt_prefix = PROMPTS[level][0].format(
        task=INSTRUCTION_DICT[track],
        arm_desc=task_dict["arm"]["desc"],
        num_arms=task_dict["num_arms"],
        num_distractors=task_dict["num_arms"] - 1,
        images=max_image_history,
    )
    prompt_suffix = PROMPTS[level][1].format(
        options=options_string(task_dict["num_arms"]),
        distractor_summary=distractor_summary,
        num_arms=task_dict["num_arms"],
        images=max_image_history,
    )
    return prompt_prefix, prompt_suffix


def format_command(command):
    return (
        f"Step {command['step']}: {command['name']} "
        f"(shoulder_delta={command['delta'][0]}, elbow_delta={command['delta'][1]})"
    )


def build_model_content(
    prompt_prefix,
    prompt_suffix,
    command,
    command_history,
    visual_history,
    motion_diff,
    current_obs,
):
    text_blocks = [
        prompt_prefix,
        "\nMotor-command trace sent to your own arm:\n",
    ]
    text_blocks.extend(f"- {format_command(item)}\n" for item in command_history)
    text_blocks.append(f"\nCurrent command to explain:\n- {format_command(command)}\n")
    text_blocks.append(
        "\nJudge this step from the command trace and images. "
        "Do not rely on any previous answer.\n"
    )

    content_items = ["".join(text_blocks)]
    if visual_history:
        content_items.append("\nVisual history, oldest to newest:\n")
        for image in visual_history:
            content_items.append(image)
    else:
        content_items.append("\nVisual history: none.\n")

    if motion_diff is not None:
        content_items.append("\nMotion-difference image from previous view to current view:\n")
        content_items.append(motion_diff)

    content_items.append("\nCurrent view after the current command:\n")
    content_items.append(current_obs)
    content_items.append(prompt_suffix)
    return content_items


def append_limited(history, item, max_items):
    history.append(item)
    if len(history) > max_items:
        del history[: len(history) - max_items]


def progress(message):
    print(f"[VisionRecBench] {message}", flush=True)


def save_rgb(image, path):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        raise ValueError(f"Expected RGB image with shape (H, W, C), got {image.shape}")

    image = image[:, :, :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            max_value = float(np.nanmax(image)) if image.size else 0.0
            if max_value <= 1.0:
                image = image * 255.0
        image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
        image = np.clip(image, 0, 255).astype(np.uint8)

    Image.fromarray(image, mode="RGB").save(path)
    return image


def annotate_candidates(image, num_candidates):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        return image

    pil_image = Image.fromarray(image[:, :, :3], mode="RGB")
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size
    y = max(10, int(height * 0.04))
    radius = max(14, width // 42)

    for index in range(1, num_candidates + 1):
        x = int(width * (index - 0.5) / num_candidates)
        box = [x - radius, y - radius, x + radius, y + radius]
        draw.ellipse(box, fill=(0, 0, 0), outline=(255, 255, 255), width=3)
        label = str(index)
        bbox = draw.textbbox((0, 0), label)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        draw.text(
            (x - text_w / 2, y - text_h / 2 - 1),
            label,
            fill=(255, 255, 255),
        )

    return np.asarray(pil_image)


def make_motion_diff(previous_image, current_image, num_candidates):
    previous = np.asarray(previous_image[:, :, :3], dtype=np.int16)
    current = np.asarray(current_image[:, :, :3], dtype=np.int16)
    if previous.shape != current.shape:
        return annotate_candidates(current_image, num_candidates)

    diff = np.abs(current - previous).max(axis=2)
    mask = diff > 18

    base = (current * 0.35).astype(np.uint8)
    overlay = np.zeros_like(base)
    overlay[:, :, 0] = 255
    overlay[:, :, 1] = 230
    overlay[:, :, 2] = 40
    base[mask] = overlay[mask]
    return annotate_candidates(base, num_candidates)


def save_args(args_file, args, task_dict, env, max_steps):
    with open(args_file, "w") as f:
        json.dump(
            {
                "scenario": args.scenario,
                "track": task_dict["track"],
                "arm": args.arm,
                "arm_desc": task_dict["arm"]["desc"],
                "level": args.level,
                "model": args.model.replace("/", "-"),
                "max_steps": max_steps,
                "max_image_history": args.max_image_history,
                "seed": task_dict.get("seed"),
                "target_index": env.target_index,
                "renderer": args.renderer,
                "resolution": args.resolution,
                "warmup_frames": args.warmup_frames,
                "render_frames": args.render_frames,
                "candidates": env.candidates,
            },
            f,
            indent=4,
        )


def append_step_log(log_file, step, command, identification, correct, applied_commands):
    with open(log_file, "a") as f:
        f.write(f"Step {step}:\n")
        f.write(
            "Motor Command: "
            f"{command['name']} "
            f"(shoulder_delta={command['delta'][0]}, elbow_delta={command['delta'][1]})\n"
        )
        f.write(f"Model Response:\n\n{identification.text}\n")
        f.write(f"Choice: {identification.choice}\n")
        f.write(f"Valid Choice: {identification.valid}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Applied Commands By Candidate: {json.dumps(applied_commands)}\n\n")


def calculate_metrics(tag, task_dict, env, predictions, bad_response_count):
    steps = len(predictions)
    correct_flags = [item["choice"] == env.target_index for item in predictions]
    valid_choices = [item["choice"] for item in predictions if item["choice"] > 0]
    majority_choice = Counter(valid_choices).most_common(1)[0][0] if valid_choices else None
    first_correct_step = next(
        (item["step"] for item, correct in zip(predictions, correct_flags) if correct),
        None,
    )

    return {
        "tag": tag,
        "scenario": task_dict["name"],
        "track": task_dict["track"],
        "target_index": env.target_index,
        "candidates": env.candidates,
        "steps": steps,
        "accuracy": sum(correct_flags) / steps if steps else 0.0,
        "final_correct": correct_flags[-1] if correct_flags else False,
        "majority_choice": majority_choice,
        "majority_correct": majority_choice == env.target_index,
        "first_correct_step": first_correct_step,
        "bad_response": bad_response_count,
        "predictions": predictions,
    }


def run_inference(args):
    validate_api_key(args.model)
    task_dict = build_task(args)

    tag = time.strftime("%Y%m%d-%H%M%S")
    paths = create_run_paths(args, tag)

    progress("starting Isaac Sim")
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": args.renderer,
            "width": args.resolution,
            "height": args.resolution,
        }
    )
    progress("Isaac Sim is ready; importing environment")

    from source.env import VisionRecBenchEnv

    progress("building VisionRecBench scene")
    env = None
    try:
        env = VisionRecBenchEnv(
            simulation_app,
            task_dict,
            renderer=args.renderer,
            resolution=(args.resolution, args.resolution),
            warmup_frames=args.warmup_frames,
            render_frames=args.render_frames,
        )
        progress("resetting scene and rendering initial observation")
        initial_obs = env.reset()
        progress(
            "initial observation returned "
            f"type={type(initial_obs).__name__}, "
            f"shape={getattr(initial_obs, 'shape', None)}"
        )
        initial_obs = save_rgb(initial_obs, paths["obs_dir"] / "0_initial_raw.png")
        initial_obs = annotate_candidates(initial_obs, task_dict["num_arms"])
        save_rgb(initial_obs, paths["obs_dir"] / "0_initial.png")
        progress(f"initial observation saved to {paths['obs_dir'] / '0_initial.png'}")

        max_steps = task_dict["episode_steps"] if args.max_steps < 0 else args.max_steps
        identifier = create_identifier(args.model, paths["log_file_agent"])
        prompt_prefix, prompt_suffix = build_prompts(
            args.level,
            task_dict,
            args.max_image_history,
        )
        save_args(paths["args_file"], args, task_dict, env, max_steps)

        predictions = []
        command_history = []
        visual_history = [initial_obs] if args.max_image_history > 0 else []
        bad_response_count = 0
        for step in range(1, max_steps + 1):
            command = env.get_command(step)
            progress(f"step {step}/{max_steps}: applying {command['name']}")
            obs, applied_commands = env.step(command)
            obs = save_rgb(obs, paths["obs_dir"] / f"{step}_{command['name']}_raw.png")
            obs = annotate_candidates(obs, task_dict["num_arms"])
            save_rgb(obs, paths["obs_dir"] / f"{step}_{command['name']}.png")
            motion_diff = make_motion_diff(visual_history[-1], obs, task_dict["num_arms"]) if visual_history else None
            if motion_diff is not None:
                save_rgb(motion_diff, paths["obs_dir"] / f"{step}_{command['name']}_motion.png")
            append_limited(command_history, command, max_steps)

            progress(
                f"Step {step} for {args.model} {args.scenario}, "
                f"target candidate {env.target_index}, tag {tag}"
            )
            content_items = build_model_content(
                prompt_prefix,
                prompt_suffix,
                command,
                command_history,
                visual_history,
                motion_diff,
                obs,
            )
            identification = identifier.identify(content_items, task_dict["num_arms"])
            if not identification.valid:
                bad_response_count += 1

            correct = identification.choice == env.target_index
            append_limited(visual_history, obs, args.max_image_history)
            predictions.append(
                {
                    "step": step,
                    "command": command,
                    "choice": identification.choice,
                    "valid": identification.valid,
                    "correct": correct,
                    "applied_commands": applied_commands,
                }
            )
            append_step_log(
                paths["log_file"],
                step,
                command,
                identification,
                correct,
                applied_commands,
            )

        with open(paths["result_file"], "w") as f:
            json.dump(
                calculate_metrics(
                    tag,
                    task_dict,
                    env,
                    predictions,
                    bad_response_count,
                ),
                f,
                indent=4,
            )
        progress(f"result saved to {paths['result_file']}")
    except Exception:
        error_text = traceback.format_exc()
        progress("run failed; traceback follows")
        print(error_text, file=sys.stderr, flush=True)
        with open(paths["log_dir"] / "error.txt", "w") as f:
            f.write(error_text)
        raise
    finally:
        if env is not None:
            env.close()
        else:
            simulation_app.close()


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
