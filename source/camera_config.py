import copy


CAMERA_VIEW_IDS = ("front", "overhead")

# Fixed, view-specific evidence crops expressed as normalized
# (left, top, right, bottom) coordinates.  They deliberately retain the full
# image width because the two robot workspaces can overlap the image centre.
# Keeping these boxes independent of episode content preserves the strict
# nuisance pairing while removing ceiling/floor pixels that carry no motion
# evidence.
EVIDENCE_WORKSPACE_CROPS = {
    "front": (0.0, 0.34, 1.0, 0.86),
    "overhead": (0.0, 0.27, 1.0, 0.79),
}

OVERHEAD_CAMERA = {
    "camera_eye": [0.0, -0.85, 5.30],
    "camera_target": [0.0, 0.04, 0.55],
    "camera_focal": 2.10,
}


def apply_camera_view(task, camera_view):
    """Apply a base camera profile before pair-shared random offsets."""
    if camera_view not in CAMERA_VIEW_IDS:
        raise ValueError(f"Unsupported camera view: {camera_view!r}")
    if camera_view == "overhead":
        task.update(copy.deepcopy(OVERHEAD_CAMERA))
    task["camera_view"] = camera_view
    return task
