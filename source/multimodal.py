from PIL import Image, ImageDraw
import numpy as np

from source.action import options_string
from source.prompts import PROMPT_PREFIX, PROMPT_SUFFIX


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


def format_command(command, labels):
    return (
        f"Step {command['step']}: {command['name']} "
        f"({format_delta(command['delta'], labels)})"
    )


def build_prompts(answer_options):
    return PROMPT_PREFIX, PROMPT_SUFFIX.format(
        options=options_string(labels=answer_options),
    )


def build_model_content(
    prompt_prefix,
    prompt_suffix,
    control_labels,
    command_history,
    visual_history,
    current_evidence,
    current_observation,
    visual_history_commands,
):
    if not command_history:
        raise ValueError("At least one motor command is required.")
    if len(visual_history_commands) != len(visual_history):
        raise ValueError(
            "visual_history_commands must align one-to-one with visual_history."
        )

    text_blocks = [prompt_prefix, "\nMotor-command trace:\n"]
    text_blocks.extend(
        f"- {format_command(command, control_labels)}\n"
        for command in command_history
    )
    content_items = ["".join(text_blocks)]

    content_items.append(
        "\nTime-ordered visual evidence paired with the corresponding command:\n"
    )
    for history_command, image in zip(
        visual_history_commands,
        visual_history,
    ):
        content_items.append(
            f"\nVisual evidence after {format_command(history_command, control_labels)}:\n"
        )
        content_items.append(image)

    final_command = command_history[-1]
    content_items.append(
        f"\nVisual evidence after {format_command(final_command, control_labels)}:\n"
    )
    content_items.append(current_evidence)
    content_items.append("\nFinal camera view after the complete command trace:\n")
    content_items.append(current_observation)
    content_items.append(prompt_suffix)
    return content_items


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


def _resize_image(image, size, nearest=False):
    resampling_class = getattr(Image, "Resampling", Image)
    resampling = resampling_class.NEAREST if nearest else resampling_class.BILINEAR
    return Image.fromarray(image[:, :, :3], mode="RGB").resize(size, resampling)


def _candidate_strip(image, candidate_index, num_candidates):
    image = np.asarray(image[:, :, :3], dtype=np.uint8)
    width = image.shape[1]
    left = int(round(width * (candidate_index - 1) / num_candidates))
    right = int(round(width * candidate_index / num_candidates))
    return image[:, left:right, :]


def _signed_change_image(previous, current):
    previous = np.asarray(previous[:, :, :3], dtype=np.int16)
    current = np.asarray(current[:, :, :3], dtype=np.int16)
    delta = current - previous
    abs_delta = np.abs(delta)
    strongest_channel = np.argmax(abs_delta, axis=2)
    signed_delta = np.take_along_axis(
        delta,
        strongest_channel[:, :, None],
        axis=2,
    ).squeeze(axis=2)
    motion_mask = abs_delta.max(axis=2) > 18

    base = np.clip(current * 0.34 + previous * 0.12, 0, 255).astype(np.uint8)
    newer_pixels = motion_mask & (signed_delta >= 0)
    older_pixels = motion_mask & (signed_delta < 0)
    base[newer_pixels] = np.array([255, 132, 36], dtype=np.uint8)
    base[older_pixels] = np.array([52, 128, 255], dtype=np.uint8)
    return base


def make_candidate_motion_panel(
    previous_image,
    current_image,
    num_candidates,
    annotate=True,
):
    del annotate
    previous = np.asarray(previous_image[:, :, :3], dtype=np.uint8)
    current = np.asarray(current_image[:, :, :3], dtype=np.uint8)
    if previous.shape != current.shape:
        return current_image

    height, width = current.shape[:2]
    row_label_width = max(96, width // 10)
    label_height = max(32, height // 28)
    header_height = label_height * 2
    tile_width = width // num_candidates
    tile_height = max(1, (height - header_height) // 3)
    panel_height = header_height + tile_height * 3

    panel = Image.new(
        "RGB",
        (row_label_width + tile_width * num_candidates, panel_height),
        (12, 18, 24),
    )
    draw = ImageDraw.Draw(panel)

    row_labels = ["BEFORE", "AFTER", "SIGNED CHANGE"]
    for candidate_index in range(1, num_candidates + 1):
        x = row_label_width + (candidate_index - 1) * tile_width
        draw.text(
            (x + 8, label_height + max(6, label_height // 4)),
            f"candidate {candidate_index}",
            fill=(255, 255, 255),
        )

        previous_crop = _candidate_strip(previous, candidate_index, num_candidates)
        current_crop = _candidate_strip(current, candidate_index, num_candidates)
        change_crop = _signed_change_image(previous_crop, current_crop)
        crops = [previous_crop, current_crop, change_crop]

        for row_index, crop in enumerate(crops):
            y = header_height + row_index * tile_height
            tile = _resize_image(
                crop,
                (tile_width, tile_height),
                nearest=row_index == 2,
            )
            panel.paste(tile, (x, y))

        draw.line(
            [(x, label_height), (x, panel_height)],
            fill=(255, 255, 255),
            width=1,
        )

    for row_index, label in enumerate(row_labels):
        y = header_height + row_index * tile_height
        draw.text(
            (8, y + max(6, tile_height // 2 - 6)),
            label,
            fill=(255, 255, 255),
        )
        draw.line(
            [(0, y), (panel.size[0], y)],
            fill=(255, 255, 255),
            width=1,
        )

    legend = "signed change: orange=new/current pixels, blue=old/previous pixels"
    legend_box = draw.textbbox((0, 0), legend)
    legend_width = legend_box[2] - legend_box[0]
    draw.rectangle(
        [
            4,
            4,
            min(panel.size[0] - 4, legend_width + 18),
            min(label_height - 4, 28),
        ],
        fill=(0, 0, 0),
    )
    draw.text(
        (10, 8),
        legend,
        fill=(255, 255, 255),
    )
    draw.line(
        [(0, label_height), (panel.size[0], label_height)],
        fill=(255, 255, 255),
        width=1,
    )

    return np.asarray(panel)
