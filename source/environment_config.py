import copy

import numpy as np


ENVIRONMENT_TEMPLATES = (
    {
        "id": "robotics_lab",
        "wall_color": [0.18, 0.24, 0.27],
        "ground_color": [0.10, 0.12, 0.14],
        "structure_color": [0.16, 0.18, 0.20],
        "panel_color": [0.54, 0.61, 0.64],
        "accent_color": [0.16, 0.58, 0.76],
        "secondary_accent_color": [0.94, 0.58, 0.16],
    },
    {
        "id": "assembly_cell",
        "wall_color": [0.27, 0.25, 0.21],
        "ground_color": [0.12, 0.12, 0.11],
        "structure_color": [0.13, 0.15, 0.16],
        "panel_color": [0.48, 0.49, 0.46],
        "accent_color": [0.92, 0.55, 0.10],
        "secondary_accent_color": [0.22, 0.48, 0.68],
    },
    {
        "id": "inspection_bay",
        "wall_color": [0.19, 0.21, 0.28],
        "ground_color": [0.09, 0.10, 0.14],
        "structure_color": [0.17, 0.18, 0.24],
        "panel_color": [0.62, 0.65, 0.70],
        "accent_color": [0.30, 0.72, 0.50],
        "secondary_accent_color": [0.78, 0.38, 0.56],
    },
)
ENVIRONMENT_TEMPLATE_IDS = tuple(
    template["id"] for template in ENVIRONMENT_TEMPLATES
)


def _jitter_color(color, rng, amount=0.025):
    varied = np.asarray(color, dtype=float) + rng.uniform(
        -amount,
        amount,
        size=3,
    )
    return [round(float(value), 6) for value in np.clip(varied, 0.03, 0.97)]


def sample_environment(rng, nuisance_pair_index, base_seed):
    """Return a deterministic, pair-shared procedural environment config."""
    template_index = (
        int(base_seed) + int(nuisance_pair_index)
    ) % len(ENVIRONMENT_TEMPLATES)
    environment = copy.deepcopy(ENVIRONMENT_TEMPLATES[template_index])
    environment["template_index"] = template_index
    environment["prop_shift"] = [
        round(float(rng.uniform(-0.10, 0.10)), 6),
        round(float(rng.uniform(-0.04, 0.04)), 6),
    ]
    environment["panel_variant"] = int(rng.integers(0, 3))
    environment["bin_color_order"] = [
        int(index) for index in rng.permutation(3)
    ]
    for color_key in (
        "wall_color",
        "ground_color",
        "structure_color",
        "panel_color",
        "accent_color",
        "secondary_accent_color",
    ):
        environment[color_key] = _jitter_color(
            environment[color_key],
            rng,
        )
    return environment
