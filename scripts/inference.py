import argparse
import copy
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
from source.render_config import RENDER_CONFIG


INSTRUCTION_DICT = {
    "Robot": (
        "You are playing the role of a simulated robotic agent performing a "
        "visual self-recognition task inside a 3D Isaac Sim environment."
    )
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="scene3_triad_delay_invert")
    parser.add_argument(
        "--arm",
        type=str,
        default=None,
        help=(
            "Advanced arm override from tasks/arm_repo.json. "
            "Standard benchmark scenarios are configured to use panda_arm."
        ),
    )
    parser.add_argument("--level", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--model", type=str, required=True, help="Model name to use, or random")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum episode steps, -1 for scenario default")
    parser.add_argument("--max_image_history", type=int, default=4, help="Maximum previous images to keep")
    parser.add_argument("--target_index", type=int, default=None, help="Optional 1-based target candidate index")
    parser.add_argument("--seed", type=int, default=None, help="Override the scenario random seed")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim in headless mode")
    args = parser.parse_args()
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max_steps must be -1 or a positive integer")
    if args.max_image_history < 0:
        parser.error("--max_image_history must be non-negative")
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
    task_dict["anti_aliasing_op"] = RENDER_CONFIG["anti_aliasing"]
    task_dict["pathtracing_spp"] = RENDER_CONFIG["pathtracing_spp"]
    task_dict["denoiser_enabled"] = RENDER_CONFIG["denoiser"]
    return task_dict


def create_run_paths(args, task_dict, tag):
    model_name = args.model.replace("/", "-")
    scene = task_dict.get("scene", 3)
    result_dir = (
        BASE_DIR
        / "results"
        / f"scene{scene}"
        / f"prompt_level{args.level}"
        / model_name
        / args.scenario
    )
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


def get_answer_options(task_dict):
    if "answer_options" in task_dict:
        return list(task_dict["answer_options"])
    return [
        f"candidate arm {i} from left to right"
        for i in range(1, int(task_dict["num_arms"]) + 1)
    ]


def get_control_labels(task_dict):
    delta_dim = len(task_dict["command_sequence"][0]["delta"])
    default_labels = (
        ["shoulder", "elbow"][:delta_dim]
        if delta_dim <= 2
        else [f"axis_{index}" for index in range(1, delta_dim + 1)]
    )
    labels = list(task_dict["arm"].get("control_labels", default_labels))
    if len(labels) != delta_dim:
        raise ValueError("arm control_labels length must match command delta dimension.")
    return labels


def format_delta(delta, labels):
    if len(delta) != len(labels):
        raise ValueError("command delta length must match control label length.")
    return ", ".join(
        f"{label}_delta={value:g}"
        for label, value in zip(labels, delta)
    )


def build_prompt_context(task_dict):
    num_arms = int(task_dict["num_arms"])
    task_mode = task_dict.get("task_mode", "multi_arm")
    scene = int(task_dict.get("scene", 3))
    control_labels = get_control_labels(task_dict)
    joint_names = ", ".join(control_labels)

    if task_mode == "single_binary" and scene == 1:
        task_setup = (
            "You observe one visible robotic arm in an Isaac Sim scene. "
            "The visible arm may follow each motor command at the same step, "
            "or it may execute a mismatched command stream containing comparable "
            "motions in a different order. Decide whether the visible motion is "
            "caused by the listed current command at each step."
        )
        short_task_setup = (
            "Decide whether the visible arm follows the listed command at each step."
        )
        behavior_summary = task_dict.get(
            "behavior_family_desc",
            (
                "- Self case: the visible arm follows the current motor command directly.\n"
                "- Non-self case: the visible arm executes a mismatched command stream."
            ),
        )
        behavior_rules = (
            "- Evaluate same-step command-to-motion correspondence across the complete episode.\n"
            "- Do not infer self merely because the arm moves often, uses the same action vocabulary, or reaches a plausible final pose.\n"
            "- Choose the self option only when the visible response consistently matches the command listed for that same step.\n"
            "- Choose the non-self option when the movements are systematically paired with different commands."
        )
        reasoning_steps = (
            "1. Inspect the motion-difference image associated with each step.\n"
            f"2. Compare the observed {joint_names} response with that step's command.\n"
            "3. Check the complete trace rather than judging from one coincidental movement or the final pose.\n"
            "4. Decide self for consistent same-step correspondence, or non-self for a reordered or mismatched stream."
        )
        answer_note = (
            "- The option order may vary between episodes; read the option text before choosing."
        )
    elif task_mode == "single_binary" and scene == 2:
        task_setup = (
            "You observe one visible robotic arm in a scrambled action space. "
            "Motor-command axes may map to different physical joints than their labels suggest. "
            "The hidden command-to-motion mapping may remain stable across repeated action cycles, "
            "or it may change between cycles. Decide whether the visible motion is controlled through "
            "one stable hidden mapping."
        )
        short_task_setup = (
            "Decide whether the visible arm follows one stable scrambled command-to-motion mapping."
        )
        behavior_summary = task_dict.get(
            "behavior_family_desc",
            (
                "- Self case: the visible arm uses a stable scrambled mapping from commands to physical joints.\n"
                "- Non-self case: the scrambled mapping changes between repeated action cycles."
            ),
        )
        behavior_rules = (
            "- Do not reject the arm only because the observed joint differs from the command label.\n"
            "- Under a scrambled action space, each command in the repeated cycle should produce the same mapped motion whenever it appears.\n"
            "- Choose the self option only if one mapping explains corresponding command responses across all observed cycles.\n"
            "- Choose the non-self option if corresponding commands change their physical response between cycles."
        )
        reasoning_steps = (
            "1. Track the physical response to each command in the four-command cycle.\n"
            "2. Compare each response with the same command in the other completed cycles.\n"
            "3. Ignore whether the hidden mapping matches the normal joint labels; test only whether it stays fixed.\n"
            "4. Decide self for one consistent mapping, or non-self when the mapping changes."
        )
        answer_note = (
            "- The option order may vary between episodes; read the option text before choosing."
        )
    else:
        task_setup = (
            f"You observe {num_arms} visually similar robotic arms in an Isaac Sim scene. "
            "Exactly one candidate arm is your own body. The other arms are distractors that imitate "
            "your motion with transformed command streams. Candidate arms are ordered from left to right "
            f"in the image: candidate 1 is the leftmost arm and candidate {num_arms} is the rightmost arm."
        )
        short_task_setup = (
            f"Identify which of the {num_arms} left-to-right candidate robotic arms is yourself."
        )
        distractor_summary = "\n".join(
            f"- {item['desc']}" for item in task_dict.get("distractors", [])
        )
        behavior_summary = task_dict.get("behavior_family_desc", distractor_summary)
        behavior_rules = (
            "- The target arm follows the current command directly at the same step.\n"
            "- A delayed distractor may move according to a previous command instead of the current one.\n"
            "- An inverted distractor moves in the opposite joint direction.\n"
            "- Other distractors may swap joints, smooth the command, or move randomly."
        )
        reasoning_steps = (
            "1. First inspect the motion-difference image; it is the most important image for this step.\n"
            "2. Evaluate every candidate separately instead of locking onto an earlier answer.\n"
            "3. Check whether each candidate moves immediately in the commanded joint direction.\n"
            "4. Reject candidates whose motion is delayed, inverted, random, swapped, or only partially follows the command."
        )
        answer_note = (
            "- The correct answer is the candidate whose visible motion is caused by the listed motor commands "
            "without delay or transformation."
        )

    return {
        "task_setup": task_setup,
        "short_task_setup": short_task_setup,
        "behavior_summary": behavior_summary,
        "behavior_rules": behavior_rules,
        "reasoning_steps": reasoning_steps,
        "answer_note": answer_note,
    }


def build_prompts(level, task_dict, max_image_history, answer_options=None):
    track = task_dict["track"]
    prompt_context = build_prompt_context(task_dict)
    if answer_options is None:
        answer_options = get_answer_options(task_dict)
    visual_history_mode = task_dict.get("visual_history_mode", "observations")
    if visual_history_mode == "motion_diffs":
        history_description = (
            f"up to {max_image_history} previous motion-difference images, "
            "each paired with its command and ordered from oldest to newest"
        )
    else:
        history_description = (
            f"a visual history of up to {max_image_history} previous observations, "
            "with visible arm numbers drawn above the arms"
        )
    prompt_prefix = PROMPTS[level][0].format(
        task=INSTRUCTION_DICT[track],
        arm_desc=task_dict["arm"]["desc"],
        num_arms=task_dict["num_arms"],
        num_distractors=task_dict["num_arms"] - 1,
        images=max_image_history,
        history_description=history_description,
        **prompt_context,
    )
    prompt_suffix = PROMPTS[level][1].format(
        options=options_string(labels=answer_options),
        distractor_summary=prompt_context["behavior_summary"],
        num_arms=task_dict["num_arms"],
        images=max_image_history,
        **prompt_context,
    )
    return prompt_prefix, prompt_suffix


def format_command(command, labels):
    return (
        f"Step {command['step']}: {command['name']} "
        f"({format_delta(command['delta'], labels)})"
    )


def build_model_content(
    prompt_prefix,
    prompt_suffix,
    control_labels,
    command,
    command_history,
    visual_history,
    motion_diff,
    current_obs,
    judgement_interval=1,
    visual_history_mode="observations",
):
    text_blocks = [
        prompt_prefix,
        "\nMotor-command trace available to the agent:\n",
    ]
    text_blocks.extend(f"- {format_command(item, control_labels)}\n" for item in command_history)
    if judgement_interval > 1:
        cycle_number = (int(command["step"]) + judgement_interval - 1) // judgement_interval
        text_blocks.append(
            f"\nCurrent command completes action cycle {cycle_number} "
            f"(cycle length: {judgement_interval} commands):\n"
            f"- {format_command(command, control_labels)}\n"
        )
        text_blocks.append(
            "\nJudge this completed action cycle from the command trace and images. "
            "Use evidence from prior cycles when available, and do not rely on any previous answer.\n"
        )
    else:
        text_blocks.append(f"\nCurrent command to explain:\n- {format_command(command, control_labels)}\n")
        text_blocks.append(
            "\nJudge this step from the command trace and images. "
            "Do not rely on any previous answer.\n"
        )

    content_items = ["".join(text_blocks)]
    if visual_history:
        if visual_history_mode == "motion_diffs":
            content_items.append(
                "\nMotion-difference history paired with its command, "
                "oldest to newest:\n"
            )
            for history_command, image in zip(command_history, visual_history):
                content_items.append(
                    f"\nMotion after {format_command(history_command, control_labels)}:\n"
                )
                content_items.append(image)
        else:
            content_items.append("\nVisual history, oldest to newest:\n")
            for image in visual_history:
                content_items.append(image)
    else:
        content_items.append("\nVisual evidence history: none.\n")

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


def make_motion_diff(
    previous_image,
    current_image,
    num_candidates,
    annotate=True,
):
    previous = np.asarray(previous_image[:, :, :3], dtype=np.int16)
    current = np.asarray(current_image[:, :, :3], dtype=np.int16)
    if previous.shape != current.shape:
        return (
            annotate_candidates(current_image, num_candidates)
            if annotate
            else current_image
        )

    diff = np.abs(current - previous).max(axis=2)
    mask = diff > 18

    base = (current * 0.35).astype(np.uint8)
    overlay = np.zeros_like(base)
    overlay[:, :, 0] = 255
    overlay[:, :, 1] = 230
    overlay[:, :, 2] = 40
    base[mask] = overlay[mask]
    return annotate_candidates(base, num_candidates) if annotate else base


def save_args(
    args_file,
    args,
    task_dict,
    env,
    max_steps,
    image_history_limit,
    judgement_interval,
    judgement_start_step,
):
    with open(args_file, "w") as f:
        json.dump(
            {
                "scenario": args.scenario,
                "track": task_dict["track"],
                "scene": task_dict.get("scene", 3),
                "task_mode": task_dict.get("task_mode", "multi_arm"),
                "arm": args.arm,
                "arm_desc": task_dict["arm"]["desc"],
                "control_joints": task_dict["arm"].get("control_joints"),
                "control_labels": get_control_labels(task_dict),
                "prompt_level": args.level,
                "model": args.model.replace("/", "-"),
                "max_steps": max_steps,
                "max_image_history": image_history_limit,
                "requested_max_image_history": args.max_image_history,
                "annotate_candidates": bool(
                    task_dict.get("annotate_candidates", True)
                ),
                "visual_history_mode": task_dict.get(
                    "visual_history_mode",
                    "observations",
                ),
                "judge_interval_steps": judgement_interval,
                "judge_start_step": judgement_start_step,
                "seed": task_dict.get("seed"),
                "target_index": env.target_index,
                "target_present": env.target_present,
                "answer_index": env.answer_index,
                "answer_text": env.answer_options[env.answer_index - 1],
                "answer_options": env.answer_options,
                "sampled_behavior_option": env.task_dict.get("sampled_behavior_option"),
                "visible_arm_behavior": env.task_dict.get("visible_arm_behavior"),
                "renderer": RENDER_CONFIG["renderer"],
                "resolution": RENDER_CONFIG["resolution"],
                "warmup_frames": RENDER_CONFIG["warmup_frames"],
                "render_frames": RENDER_CONFIG["render_frames"],
                "anti_aliasing": RENDER_CONFIG["anti_aliasing"],
                "pathtracing_spp": RENDER_CONFIG["pathtracing_spp"],
                "denoiser": RENDER_CONFIG["denoiser"],
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
            f"({format_delta(command['delta'], command['control_labels'])})\n"
        )
        f.write(f"Model Response:\n\n{identification.text}\n")
        f.write(f"Choice: {identification.choice}\n")
        f.write(f"Valid Choice: {identification.valid}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Applied Commands By Candidate: {json.dumps(applied_commands)}\n\n")


def calculate_metrics(
    tag,
    task_dict,
    env,
    predictions,
    action_trace,
    bad_response_count,
):
    prediction_steps = len(predictions)
    action_steps = predictions[-1]["step"] if predictions else 0
    correct_flags = [item["choice"] == env.answer_index for item in predictions]
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
        "scene": task_dict.get("scene", 3),
        "task_mode": task_dict.get("task_mode", "multi_arm"),
        "target_index": env.target_index,
        "target_present": env.target_present,
        "answer_index": env.answer_index,
        "answer_text": env.answer_options[env.answer_index - 1],
        "answer_options": env.answer_options,
        "sampled_behavior_option": env.task_dict.get("sampled_behavior_option"),
        "visible_arm_behavior": env.task_dict.get("visible_arm_behavior"),
        "judge_interval_steps": int(task_dict.get("judge_interval_steps", 1)),
        "judge_start_step": int(task_dict.get("judge_start_step", 1)),
        "candidates": env.candidates,
        "steps": action_steps,
        "prediction_steps": prediction_steps,
        "accuracy": sum(correct_flags) / prediction_steps if prediction_steps else 0.0,
        "final_correct": correct_flags[-1] if correct_flags else False,
        "majority_choice": majority_choice,
        "majority_correct": majority_choice == env.answer_index,
        "first_correct_step": first_correct_step,
        "bad_response": bad_response_count,
        "action_trace": action_trace,
        "predictions": predictions,
    }


def run_inference(args):
    validate_api_key(args.model)
    task_dict = build_task(args)

    tag = time.strftime("%Y%m%d-%H%M%S")
    paths = create_run_paths(args, task_dict, tag)

    progress("starting Isaac Sim")
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
    progress("Isaac Sim is ready; importing environment")

    from source.env import VisionRecBenchEnv

    progress("building VisionRecBench scene")
    env = None
    try:
        env = VisionRecBenchEnv(
            simulation_app,
            task_dict,
        )
        progress("resetting scene and rendering initial observation")
        initial_obs = env.reset()
        progress(
            "initial observation returned "
            f"type={type(initial_obs).__name__}, "
            f"shape={getattr(initial_obs, 'shape', None)}"
        )
        initial_obs = save_rgb(initial_obs, paths["obs_dir"] / "0_initial_raw.png")
        should_annotate_candidates = bool(
            task_dict.get("annotate_candidates", True)
        )
        if should_annotate_candidates:
            initial_obs = annotate_candidates(
                initial_obs,
                task_dict["num_arms"],
            )
        save_rgb(initial_obs, paths["obs_dir"] / "0_initial.png")
        progress(f"initial observation saved to {paths['obs_dir'] / '0_initial.png'}")

        max_steps = task_dict["episode_steps"] if args.max_steps < 0 else args.max_steps
        judgement_interval = int(task_dict.get("judge_interval_steps", 1))
        if judgement_interval < 1:
            raise ValueError("judge_interval_steps must be a positive integer.")
        judgement_start_step = int(task_dict.get("judge_start_step", 1))
        if not 1 <= judgement_start_step <= max_steps:
            raise ValueError("judge_start_step must be within the episode.")
        image_history_limit = args.max_image_history
        if judgement_interval > 1:
            image_history_limit = max(image_history_limit, max_steps)
        visual_history_mode = task_dict.get(
            "visual_history_mode",
            "observations",
        )
        if visual_history_mode not in {"observations", "motion_diffs"}:
            raise ValueError(
                "visual_history_mode must be observations or motion_diffs."
            )
        identifier = create_identifier(args.model, paths["log_file_agent"])
        prompt_prefix, prompt_suffix = build_prompts(
            args.level,
            task_dict,
            image_history_limit,
            answer_options=env.answer_options,
        )
        control_labels = get_control_labels(task_dict)
        save_args(
            paths["args_file"],
            args,
            task_dict,
            env,
            max_steps,
            image_history_limit,
            judgement_interval,
            judgement_start_step,
        )

        predictions = []
        action_trace = []
        command_history = []
        visual_history = (
            [initial_obs]
            if image_history_limit > 0 and visual_history_mode == "observations"
            else []
        )
        previous_obs = initial_obs
        bad_response_count = 0
        for step in range(1, max_steps + 1):
            command = env.get_command(step)
            command["control_labels"] = control_labels
            progress(f"step {step}/{max_steps}: applying {command['name']}")
            obs, applied_commands = env.step(command)
            action_trace.append(
                {
                    "step": step,
                    "command": copy.deepcopy(command),
                    "applied_commands": copy.deepcopy(applied_commands),
                }
            )
            obs = save_rgb(obs, paths["obs_dir"] / f"{step}_{command['name']}_raw.png")
            if should_annotate_candidates:
                obs = annotate_candidates(obs, task_dict["num_arms"])
            save_rgb(obs, paths["obs_dir"] / f"{step}_{command['name']}.png")
            motion_diff = make_motion_diff(
                previous_obs,
                obs,
                task_dict["num_arms"],
                annotate=should_annotate_candidates,
            )
            if motion_diff is not None:
                save_rgb(motion_diff, paths["obs_dir"] / f"{step}_{command['name']}_motion.png")
            append_limited(command_history, command, max_steps)
            should_identify = step >= judgement_start_step and (
                step % judgement_interval == 0 or step == max_steps
            )

            if should_identify:
                progress(
                    f"Step {step} for {args.model} {args.scenario}, "
                    f"target candidate {env.target_index}, tag {tag}"
                )
                content_items = build_model_content(
                    prompt_prefix,
                    prompt_suffix,
                    control_labels,
                    command,
                    command_history,
                    visual_history,
                    motion_diff,
                    obs,
                    judgement_interval=judgement_interval,
                    visual_history_mode=visual_history_mode,
                )
                identification = identifier.identify(content_items, len(env.answer_options))
                if not identification.valid:
                    bad_response_count += 1

                correct = identification.choice == env.answer_index
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
            else:
                progress(
                    f"step {step}/{max_steps}: deferring LLM judgement until the current action cycle is complete"
                )

            history_item = (
                motion_diff
                if visual_history_mode == "motion_diffs"
                else obs
            )
            if image_history_limit > 0 and history_item is not None:
                append_limited(
                    visual_history,
                    history_item,
                    image_history_limit,
                )
            previous_obs = obs

        with open(paths["result_file"], "w") as f:
            json.dump(
                calculate_metrics(
                    tag,
                    task_dict,
                    env,
                    predictions,
                    action_trace,
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
