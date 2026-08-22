import copy
import math
from pathlib import Path

import numpy as np

import carb
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import SingleArticulation, XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from source.render_config import RENDER_CONFIG
from source.task_logic import (
    apply_mapped_behavior,
    build_multi_arm_role_assignment,
    build_mismatched_command_schedule,
    configure_binary_answers,
    materialize_mapping_behavior,
    select_behavior_option,
)


BASE_DIR = Path(__file__).resolve().parents[1]


class VisionRecBenchEnv:
    def __init__(
        self,
        sim_app,
        task_dict,
    ):
        self.sim_app = sim_app
        self.task_dict = copy.deepcopy(task_dict)
        renderer = RENDER_CONFIG["renderer"]
        resolution = (RENDER_CONFIG["resolution"], RENDER_CONFIG["resolution"])

        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", renderer)
        background_color = self._cfg_vec("background_color", [0.78, 0.82, 0.86])
        settings.set("/rtx/post/backgroundZeroAlpha/enable", False)
        settings.set("/rtx/post/background/color", background_color)
        settings.set("/rtx/sceneDb/ambientLightIntensity", float(self.task_dict.get("ambient_light_intensity", 0.25)))
        settings.set("/rtx/post/aa/op", int(self.task_dict.get("anti_aliasing_op", 2)))
        settings.set("/rtx-transient/dlssg/enabled", False)
        settings.set("/rtx/post/motionblur/enabled", False)
        settings.set("/rtx/post/motionblur/numSamples", 0)
        settings.set("/rtx/denoising/enabled", bool(self.task_dict.get("denoiser_enabled", True)))
        if renderer == "PathTracing":
            pathtracing_spp = int(self.task_dict.get("pathtracing_spp", 16))
            settings.set("/rtx/pathtracing/spp", pathtracing_spp)
            settings.set("/rtx/pathtracing/totalSpp", pathtracing_spp)
            settings.set("/rtx/pathtracing/maxBounces", int(self.task_dict.get("pathtracing_max_bounces", 6)))
            settings.set("/rtx/pathtracing/maxSpecularAndTransmissionBounces", int(self.task_dict.get("pathtracing_max_specular_bounces", 4)))
            settings.set("/rtx/pathtracing/maxVolumeBounces", int(self.task_dict.get("pathtracing_max_volume_bounces", 2)))

        self.renderer = renderer
        self.resolution = tuple(resolution)
        self.warmup_frames = int(RENDER_CONFIG["warmup_frames"])
        self.render_frames = int(RENDER_CONFIG["render_frames"])
        self.arm_cfg = self.task_dict["arm"]
        self.arm_root = self.arm_cfg.get("root", "procedural")
        self.num_arms = int(self.task_dict["num_arms"])
        self.episode_steps = int(self.task_dict["episode_steps"])
        self.rng = np.random.default_rng(int(self.task_dict.get("seed", 0)))
        self.task_mode = self.task_dict.get("task_mode", "multi_arm")

        if self.task_mode == "single_binary":
            self._configure_single_binary_task()
        else:
            self._configure_multi_arm_task()

        self.world = World(stage_units_in_meters=1.0)
        self.stage = self.world.stage

        self.link_lengths = np.array(self.arm_cfg["link_lengths"], dtype=float)
        self.link_thickness = float(self.arm_cfg.get("link_thickness", 0.08))
        self.base_size = np.array(self.arm_cfg["base_size"], dtype=float)
        self.wrist_size = np.array(self.arm_cfg["wrist_size"], dtype=float)
        self.initial_joints = np.array(
            self.arm_cfg.get(
                "initial_joint_positions",
                self.arm_cfg.get("initial_joints_deg"),
            ),
            dtype=float,
        )
        self.joint_limits = np.array(
            self.arm_cfg.get(
                "joint_limits",
                self.arm_cfg.get("joint_limits_deg"),
            ),
            dtype=float,
        )
        self.command_step = float(
            self.arm_cfg.get(
                "command_step",
                self.arm_cfg.get("command_step_deg"),
            )
        )
        self.command_sequence = copy.deepcopy(self.task_dict["command_sequence"])
        self.command_dim = len(self.command_sequence[0]["delta"])
        default_labels = (
            ["shoulder", "elbow"][: self.command_dim]
            if self.command_dim <= 2
            else [f"axis_{index}" for index in range(1, self.command_dim + 1)]
        )
        self.control_labels = list(self.arm_cfg.get("control_labels", default_labels))
        if len(self.control_labels) != self.command_dim:
            raise ValueError(
                "arm control_labels length must match command delta dimension."
            )
        for item in self.command_sequence:
            if len(item["delta"]) != self.command_dim:
                raise ValueError(
                    "Every command delta in a scenario must have the same dimension."
                )
        self.command_library = [
            np.array(item["delta"], dtype=float) for item in self.command_sequence
        ]

        self.command_memory = []
        self.arms = []
        self._create_scene()
        self._create_arms()
        self._create_camera()

    def _configure_single_binary_task(self):
        behavior_options = self.task_dict.get("visible_arm_behavior_options")
        if behavior_options:
            option_index = select_behavior_option(
                behavior_options,
                seed=self.task_dict.get("seed", 0),
                strategy=self.task_dict.get("behavior_selection", "random"),
                rng=self.rng,
            )
            selected = copy.deepcopy(behavior_options[option_index])
            self.task_dict["visible_arm_behavior"] = materialize_mapping_behavior(
                selected["behavior"],
                seed=self.task_dict.get("seed", 0),
                behavior_option_count=len(behavior_options),
            )
            self.task_dict["target_present"] = bool(selected["target_present"])
            self.task_dict["sampled_behavior_option"] = option_index + 1
        else:
            self.task_dict["visible_arm_behavior"] = materialize_mapping_behavior(
                self.task_dict["visible_arm_behavior"],
                seed=self.task_dict.get("seed", 0),
            )

        self.target_present = bool(self.task_dict.get("target_present", True))
        self.target_index = 1 if self.target_present else None
        self.answer_options, self.answer_index = configure_binary_answers(
            self.task_dict.get(
                "answer_options",
                [
                    "yes, the visible arm is myself",
                    "no, the visible arm is not myself",
                ],
            ),
            target_present=self.target_present,
            seed=self.task_dict.get("seed", 0),
            shuffle=bool(self.task_dict.get("shuffle_answer_options", False)),
        )

    def _configure_multi_arm_task(self):
        strategy = self.task_dict.get("role_assignment_strategy", "seed_stratified")
        if strategy != "seed_stratified":
            raise ValueError(
                f"Unsupported multi-arm role_assignment_strategy: {strategy}"
            )
        self.role_assignments = build_multi_arm_role_assignment(
            self.num_arms,
            self.task_dict.get("distractors", []),
            seed=self.task_dict.get("seed", 0),
            target_index=self.task_dict.get("target_index"),
        )
        self.target_index = next(
            item["index"] for item in self.role_assignments if item["role"] == "target"
        )
        self.task_dict["target_index"] = self.target_index
        self.task_dict["role_assignments"] = copy.deepcopy(self.role_assignments)
        self.target_present = True
        self.answer_options = [
            f"candidate arm {i} from left to right"
            for i in range(1, self.num_arms + 1)
        ]
        self.answer_index = self.target_index

    def _normalize_rgb(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.size == 0:
            return None

        rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating):
                max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
                if max_value <= 1.0:
                    rgb = rgb * 255.0
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _capture_rgb(self, max_attempts=120):
        for _ in range(max_attempts):
            rgb = self._normalize_rgb(self.camera.get_rgb())
            if rgb is not None:
                return rgb
            self.world.step(render=True)
        raise RuntimeError("Camera did not return a valid RGB frame.")

    def _cfg_vec(self, name, default):
        return list(self.task_dict.get(name, default))

    def _create_scene(self):
        floor_width = max(
            float(self.task_dict.get("floor_width", 4.0)),
            self.num_arms * float(self.task_dict["layout_spacing"]) + 1.0,
        )
        floor_depth = float(self.task_dict.get("floor_depth", 2.4))
        floor_z = float(self.task_dict.get("floor_z", -0.015))
        FixedCuboid(
            prim_path="/World/Floor",
            name="floor",
            position=np.array([0.0, 0.25, floor_z]),
            size=1.0,
            scale=np.array([floor_width, floor_depth, 0.03]),
        )
        self._create_and_bind_material(
            "/World/Floor",
            "/World/Looks/FloorMaterial",
            color=self._cfg_vec("floor_color", [0.56, 0.59, 0.60]),
            metallic=0.0,
            roughness=float(self.task_dict.get("floor_roughness", 0.65)),
        )
        self._create_environment(floor_width, floor_depth)

        light = UsdLux.DistantLight.Define(self.stage, Sdf.Path("/World/KeyLight"))
        light.CreateIntensityAttr(float(self.task_dict.get("key_light_intensity", 4200.0)))
        light.CreateAngleAttr(float(self.task_dict.get("key_light_angle", 0.55)))
        key_rotation = self._cfg_vec("key_light_rotation", [-50.0, 0.0, 35.0])
        UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(*key_rotation))

        dome = UsdLux.DomeLight.Define(self.stage, Sdf.Path("/World/DomeLight"))
        dome.CreateIntensityAttr(float(self.task_dict.get("dome_light_intensity", 550.0)))
        dome.CreateColorAttr(Gf.Vec3f(*self._cfg_vec("dome_light_color", [0.86, 0.90, 0.96])))

        fill = UsdLux.SphereLight.Define(self.stage, Sdf.Path("/World/FillLight"))
        fill.CreateIntensityAttr(float(self.task_dict.get("fill_light_intensity", 900.0)))
        fill.CreateRadiusAttr(float(self.task_dict.get("fill_light_radius", 3.0)))
        fill.CreateColorAttr(Gf.Vec3f(*self._cfg_vec("fill_light_color", [0.90, 0.94, 1.0])))
        fill_position = self._cfg_vec("fill_light_position", [0.0, -2.2, 2.2])
        UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(*fill_position))

    def _create_visual_box(
        self,
        name,
        position,
        scale,
        color,
        metallic=0.0,
        roughness=0.62,
    ):
        prim_path = f"/World/Environment/{name}"
        cube = UsdGeom.Cube.Define(self.stage, Sdf.Path(prim_path))
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3f(*position))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))
        self._create_and_bind_material(
            prim_path,
            f"/World/Looks/Environment_{name}",
            color=color,
            metallic=metallic,
            roughness=roughness,
        )

    def _create_environment(self, floor_width, floor_depth):
        environment = self.task_dict.get("environment")
        if not environment:
            return

        UsdGeom.Xform.Define(self.stage, Sdf.Path("/World/Environment"))
        template_id = environment["id"]
        wall = environment["wall_color"]
        ground = environment["ground_color"]
        structure = environment["structure_color"]
        panel = environment["panel_color"]
        accent = environment["accent_color"]
        secondary = environment["secondary_accent_color"]
        shift_x, shift_y = environment.get("prop_shift", [0.0, 0.0])

        half_width = floor_width / 2.0
        back_y = 0.25 + floor_depth / 2.0 + 0.10
        self._create_visual_box(
            "GroundSlab",
            [0.0, 0.30, -0.14],
            [floor_width + 1.8, floor_depth + 2.5, 0.16],
            ground,
            roughness=0.88,
        )
        self._create_visual_box(
            "BackWall",
            [0.0, back_y + 0.07, 1.42],
            [floor_width + 1.7, 0.14, 2.85],
            wall,
            roughness=0.76,
        )
        self._create_visual_box(
            "BackWallLowerTrim",
            [0.0, back_y - 0.015, 0.22],
            [floor_width + 1.45, 0.06, 0.18],
            structure,
            metallic=0.25,
        )
        self._create_visual_box(
            "CeilingBeam",
            [0.0, back_y - 0.08, 2.55],
            [floor_width + 1.35, 0.20, 0.18],
            structure,
            metallic=0.35,
        )
        self._create_visual_box(
            "FloorSafetyLine",
            [0.0, -0.82, 0.007],
            [max(1.0, floor_width - 0.45), 0.075, 0.012],
            accent,
            roughness=0.72,
        )
        for index, x in enumerate((-half_width + 0.28, half_width - 0.28)):
            self._create_visual_box(
                f"SidePost_{index}",
                [x, back_y - 0.03, 1.30],
                [0.16, 0.18, 2.45],
                structure,
                metallic=0.30,
            )

        if template_id == "robotics_lab":
            self._create_robotics_lab(
                half_width,
                back_y,
                shift_x,
                shift_y,
                structure,
                panel,
                accent,
                secondary,
                environment,
            )
        elif template_id == "assembly_cell":
            self._create_assembly_cell(
                half_width,
                back_y,
                shift_x,
                shift_y,
                structure,
                panel,
                accent,
                secondary,
                environment,
            )
        elif template_id == "inspection_bay":
            self._create_inspection_bay(
                half_width,
                back_y,
                shift_x,
                shift_y,
                structure,
                panel,
                accent,
                secondary,
                environment,
            )
        else:
            raise ValueError(f"Unsupported environment template: {template_id}")

    def _create_robotics_lab(
        self,
        half_width,
        back_y,
        shift_x,
        shift_y,
        structure,
        panel,
        accent,
        secondary,
        environment,
    ):
        bench_width = max(1.6, half_width * 1.15)
        bench_y = back_y - 0.28 + shift_y
        self._create_visual_box(
            "LabBenchTop",
            [shift_x, bench_y, 0.58],
            [bench_width, 0.42, 0.10],
            panel,
            metallic=0.18,
        )
        for index, x in enumerate((-bench_width * 0.40, bench_width * 0.40)):
            self._create_visual_box(
                f"LabBenchLeg_{index}",
                [shift_x + x, bench_y, 0.29],
                [0.10, 0.30, 0.58],
                structure,
                metallic=0.32,
            )
        monitor_spacing = min(0.72, bench_width * 0.30)
        screen_colors = (accent, secondary)
        for index, x in enumerate((-monitor_spacing, monitor_spacing)):
            self._create_visual_box(
                f"MonitorFrame_{index}",
                [shift_x + x, bench_y - 0.10, 1.05],
                [0.58, 0.10, 0.42],
                structure,
                metallic=0.22,
            )
            self._create_visual_box(
                f"MonitorScreen_{index}",
                [shift_x + x, bench_y - 0.158, 1.05],
                [0.47, 0.018, 0.31],
                screen_colors[index],
                roughness=0.32,
            )
        cabinet_x = half_width + 0.22
        for index, x in enumerate((-cabinet_x, cabinet_x)):
            self._create_visual_box(
                f"LabCabinet_{index}",
                [x, back_y - 0.15, 0.68],
                [0.46, 0.42, 1.35],
                structure,
                metallic=0.28,
            )
            for drawer in range(3):
                self._create_visual_box(
                    f"LabDrawer_{index}_{drawer}",
                    [x, back_y - 0.372, 0.35 + drawer * 0.31],
                    [0.34, 0.018, 0.21],
                    panel if drawer % 2 == 0 else accent,
                    roughness=0.55,
                )
        self._create_wall_bins(
            "LabBin",
            back_y,
            shift_x,
            1.72,
            panel,
            accent,
            secondary,
            environment,
        )

    def _create_assembly_cell(
        self,
        half_width,
        back_y,
        shift_x,
        shift_y,
        structure,
        panel,
        accent,
        secondary,
        environment,
    ):
        fence_y = back_y - 0.30 + shift_y
        fence_width = half_width * 1.45
        for index, x in enumerate(
            np.linspace(-fence_width / 2.0, fence_width / 2.0, 5)
        ):
            self._create_visual_box(
                f"FencePost_{index}",
                [float(x) + shift_x, fence_y, 1.12],
                [0.065, 0.075, 2.05],
                accent,
                metallic=0.35,
            )
        for index, z in enumerate((0.50, 1.18, 1.82)):
            self._create_visual_box(
                f"FenceRail_{index}",
                [shift_x, fence_y, z],
                [fence_width, 0.065, 0.055],
                accent,
                metallic=0.35,
            )

        rack_x = -half_width - 0.20
        for index, x in enumerate((rack_x - 0.24, rack_x + 0.24)):
            self._create_visual_box(
                f"RackPost_{index}",
                [x, back_y - 0.12, 1.05],
                [0.08, 0.34, 1.95],
                structure,
                metallic=0.30,
            )
        for index, z in enumerate((0.30, 0.87, 1.45, 2.02)):
            self._create_visual_box(
                f"RackShelf_{index}",
                [rack_x, back_y - 0.12, z],
                [0.62, 0.42, 0.07],
                panel,
                metallic=0.18,
            )
        crate_x = half_width + 0.18
        crate_colors = (panel, accent, secondary)
        for index in range(3):
            self._create_visual_box(
                f"AssemblyCrate_{index}",
                [
                    crate_x + (index % 2) * 0.12,
                    back_y - 0.26 - (index % 2) * 0.04,
                    0.20 + index * 0.36,
                ],
                [0.54, 0.46, 0.31],
                crate_colors[environment["bin_color_order"][index]],
                roughness=0.78,
            )

    def _create_inspection_bay(
        self,
        half_width,
        back_y,
        shift_x,
        shift_y,
        structure,
        panel,
        accent,
        secondary,
        environment,
    ):
        board_y = back_y - 0.086
        self._create_visual_box(
            "InspectionBoard",
            [shift_x, board_y, 1.48],
            [1.72, 0.035, 1.20],
            structure,
            metallic=0.12,
        )
        tile_colors = (panel, accent, secondary)
        variant = int(environment.get("panel_variant", 0))
        for row in range(3):
            for column in range(4):
                color = tile_colors[(row + column + variant) % len(tile_colors)]
                self._create_visual_box(
                    f"InspectionTile_{row}_{column}",
                    [
                        shift_x - 0.60 + column * 0.40,
                        board_y - 0.025,
                        1.12 + row * 0.36,
                    ],
                    [0.31, 0.018, 0.27],
                    color,
                    roughness=0.48,
                )

        console_x = half_width + 0.16
        self._create_visual_box(
            "InspectionConsole",
            [console_x, back_y - 0.28 + shift_y, 0.60],
            [0.55, 0.48, 1.18],
            structure,
            metallic=0.22,
        )
        self._create_visual_box(
            "InspectionConsoleScreen",
            [console_x, back_y - 0.528 + shift_y, 0.77],
            [0.38, 0.018, 0.32],
            accent,
            roughness=0.30,
        )
        tower_x = -half_width - 0.14
        self._create_visual_box(
            "InspectionTower",
            [tower_x, back_y - 0.17, 0.85],
            [0.20, 0.25, 1.55],
            structure,
            metallic=0.30,
        )
        for index, color in enumerate((accent, secondary, panel)):
            self._create_visual_box(
                f"InspectionStatus_{index}",
                [tower_x, back_y - 0.305, 1.20 + index * 0.22],
                [0.12, 0.025, 0.12],
                color,
                roughness=0.35,
            )

    def _create_wall_bins(
        self,
        prefix,
        back_y,
        shift_x,
        z,
        panel,
        accent,
        secondary,
        environment,
    ):
        colors = (panel, accent, secondary)
        for index, x in enumerate((-0.62, 0.0, 0.62)):
            color_index = environment["bin_color_order"][index]
            self._create_visual_box(
                f"{prefix}_{index}",
                [shift_x + x, back_y - 0.09, z],
                [0.42, 0.18, 0.30],
                colors[color_index],
                roughness=0.72,
            )

    def _create_arms(self):
        spacing = float(self.task_dict["layout_spacing"])
        x_offset = (self.num_arms - 1) * spacing / 2.0

        for index in range(1, self.num_arms + 1):
            if self.task_mode == "single_binary":
                behavior = copy.deepcopy(self.task_dict["visible_arm_behavior"])
                role = "target" if self.target_present else "non_target"
            else:
                assignment = self.role_assignments[index - 1]
                behavior = copy.deepcopy(assignment["behavior"])
                role = assignment["role"]

            base_pos = np.array([(index - 1) * spacing - x_offset, 0.0, 0.0])
            arm = {
                "index": index,
                "role": role,
                "behavior": behavior,
                "base_pos": base_pos,
                "joints": self.initial_joints.copy(),
                "smooth_command": np.zeros(self.command_dim),
                "command_schedule": None,
                "xforms": {},
                "articulation": None,
                "control_indices": None,
            }
            self._create_single_arm(arm)
            self.arms.append(arm)

        self.candidates = [
            {
                "index": arm["index"],
                "role": arm["role"],
                "behavior": arm["behavior"]["behavior"],
                "desc": arm["behavior"].get("desc", ""),
            }
            for arm in self.arms
        ]

    def _create_single_arm(self, arm):
        if self.arm_root == "panda":
            self._create_panda_arm(arm)
        elif self.arm_root == "usd":
            self._create_usd_arm(arm)
        else:
            self._create_procedural_arm(arm)

    def _resolve_usd_path(self, asset_path):
        asset_path = str(asset_path)
        if asset_path.startswith("isaac://"):
            from isaacsim.storage.native import get_assets_root_path

            assets_root = get_assets_root_path()
            if not assets_root:
                raise RuntimeError("Could not resolve Isaac Sim assets root.")
            return assets_root.rstrip("/") + "/" + asset_path[len("isaac://") :].strip("/")

        path = Path(asset_path)
        if not path.is_absolute():
            path = BASE_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"USD arm asset not found: {path}")
        return str(path)

    def _create_panda_arm(self, arm):
        prefix = f"/World/Arm_{arm['index']}"
        asset_path = self._resolve_usd_path(self.arm_cfg["asset_path"])
        add_reference_to_stage(usd_path=asset_path, prim_path=prefix)

        orientation = np.array(self.arm_cfg.get("orientation", [1.0, 0.0, 0.0, 0.0]))
        articulation = SingleArticulation(
            prim_path=prefix,
            name=f"panda_arm_{arm['index']}",
            position=arm["base_pos"],
            orientation=orientation,
        )
        arm["articulation"] = self.world.scene.add(articulation)

    def _create_usd_arm(self, arm):
        prefix = f"/World/Arm_{arm['index']}"
        asset_path = self._resolve_usd_path(self.arm_cfg["asset_path"])

        root_prim = UsdGeom.Xform.Define(self.stage, Sdf.Path(prefix)).GetPrim()
        asset_prim_path = self.arm_cfg.get("asset_prim_path")
        if asset_prim_path:
            root_prim.GetReferences().AddReference(
                str(asset_path),
                Sdf.Path(str(asset_prim_path)),
            )
        else:
            root_prim.GetReferences().AddReference(str(asset_path))

        part_paths = self.arm_cfg.get("part_paths", {})
        required_parts = ["base", "shoulder", "link1", "elbow", "link2", "wrist"]
        missing = [name for name in required_parts if name not in part_paths]
        if missing:
            raise ValueError(
                "USD arm config is missing part_paths for: "
                f"{', '.join(missing)}"
            )

        for part_name in required_parts:
            rel_path = str(part_paths[part_name]).strip("/")
            prim_path = f"{prefix}/{rel_path}"
            if not self.stage.GetPrimAtPath(prim_path).IsValid():
                raise ValueError(
                    f"USD arm part '{part_name}' does not exist at {prim_path}. "
                    "Update tasks/arm_repo.json part_paths to match the USD asset."
                )
            arm["xforms"][part_name] = XFormPrim(prim_paths_expr=prim_path)

        self._update_arm_pose(arm)

    def _create_procedural_arm(self, arm):
        prefix = f"/World/Arm_{arm['index']}"
        UsdGeom.Xform.Define(self.stage, Sdf.Path(prefix))
        colors = {
            "base": self.arm_cfg["base_color"],
            "link": self.arm_cfg["link_color"],
            "joint": self.arm_cfg["joint_color"],
            "wrist": self.arm_cfg["wrist_color"],
        }

        parts = [
            ("base", np.array(arm["base_pos"]) + np.array([0.0, 0.0, self.base_size[2] / 2]), self.base_size, colors["base"]),
            ("shoulder", np.zeros(3), np.array([self.wrist_size[0], self.wrist_size[1], self.wrist_size[2]]), colors["joint"]),
            ("link1", np.zeros(3), np.array([self.link_thickness, self.link_lengths[0], self.link_thickness]), colors["link"]),
            ("elbow", np.zeros(3), np.array([self.wrist_size[0], self.wrist_size[1], self.wrist_size[2]]), colors["joint"]),
            ("link2", np.zeros(3), np.array([self.link_thickness, self.link_lengths[1], self.link_thickness]), colors["link"]),
            ("wrist", np.zeros(3), self.wrist_size, colors["wrist"]),
        ]

        for part_name, position, scale, color in parts:
            prim_path = f"{prefix}/{part_name}"
            FixedCuboid(
                prim_path=prim_path,
                name=f"arm_{arm['index']}_{part_name}",
                position=position,
                size=1.0,
                scale=scale,
            )
            self._create_and_bind_material(
                prim_path,
                f"/World/Looks/Arm{arm['index']}_{part_name}",
                color=color,
                metallic=0.0,
                roughness=0.45,
            )
            arm["xforms"][part_name] = XFormPrim(prim_paths_expr=prim_path)

        self._update_arm_pose(arm)

    def _create_camera(self):
        camera_eye = self._cfg_vec("camera_eye", [0.0, -3.9, 2.15])
        camera_target = self._cfg_vec("camera_target", [0.0, 0.25, 0.55])
        self.camera = Camera(
            prim_path="/World/Camera",
            translation=np.array(camera_eye),
            frequency=20,
            resolution=self.resolution,
        )
        set_camera_view(
            eye=camera_eye,
            target=camera_target,
            camera_prim_path="/World/Camera",
        )
        self.camera.set_focal_length(float(self.task_dict.get("camera_focal", 2.8)))

    def _create_and_bind_material(
        self,
        prim_path,
        mat_path,
        color,
        metallic=0.0,
        roughness=0.5,
    ):
        mat = UsdShade.Material.Define(self.stage, Sdf.Path(mat_path))
        shader = UsdShade.Shader.Define(self.stage, Sdf.Path(mat_path).AppendChild("PreviewSurface"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(self.stage.GetPrimAtPath(prim_path)).Bind(mat)

    def _quat_x(self, angle_deg):
        return rot_utils.euler_angles_to_quats(
            np.array([[angle_deg, 0.0, 0.0]]),
            degrees=True,
            extrinsic=False,
        )

    def _update_arm_pose(self, arm):
        if self.arm_root == "panda":
            arm["articulation"].set_joint_positions(arm["joints"])
            return

        shoulder = arm["base_pos"] + np.array([0.0, 0.0, self.base_size[2]])
        theta1 = math.radians(float(arm["joints"][0]))
        theta2 = math.radians(float(arm["joints"][0] + arm["joints"][1]))

        elbow = shoulder + np.array(
            [0.0, self.link_lengths[0] * math.cos(theta1), self.link_lengths[0] * math.sin(theta1)]
        )
        wrist = elbow + np.array(
            [0.0, self.link_lengths[1] * math.cos(theta2), self.link_lengths[1] * math.sin(theta2)]
        )

        link1_mid = (shoulder + elbow) / 2.0
        link2_mid = (elbow + wrist) / 2.0

        arm["xforms"]["shoulder"].set_world_poses(positions=np.array([shoulder]))
        arm["xforms"]["elbow"].set_world_poses(positions=np.array([elbow]))
        arm["xforms"]["wrist"].set_world_poses(positions=np.array([wrist]))
        arm["xforms"]["link1"].set_world_poses(
            positions=np.array([link1_mid]),
            orientations=self._quat_x(float(arm["joints"][0])),
        )
        arm["xforms"]["link2"].set_world_poses(
            positions=np.array([link2_mid]),
            orientations=self._quat_x(float(arm["joints"][0] + arm["joints"][1])),
        )

    def reset(self):
        self.command_memory = []
        self.world.reset()
        for arm in self.arms:
            arm["joints"] = self.initial_joints.copy()
            arm["smooth_command"] = np.zeros(self.command_dim)
            behavior_name = arm["behavior"]["behavior"]
            if behavior_name in {"sequence_derangement", "sequence_mismatch"}:
                mismatch_count = (
                    self.episode_steps
                    if behavior_name == "sequence_derangement"
                    else int(arm["behavior"]["mismatch_count"])
                )
                arm["command_schedule"] = build_mismatched_command_schedule(
                    self.command_library,
                    episode_steps=self.episode_steps,
                    seed=int(self.task_dict.get("seed", 0)),
                    mismatch_count=mismatch_count,
                )
            else:
                arm["command_schedule"] = None
            if self.arm_root == "panda":
                self._initialize_panda_controls(arm)
            self._update_arm_pose(arm)

        self.camera.initialize()
        for _ in range(self.warmup_frames):
            self.world.step(render=True)
        return self._capture_rgb()

    def get_command(self, step):
        command = copy.deepcopy(self.command_sequence[(step - 1) % len(self.command_sequence)])
        command["delta"] = [float(value) for value in command["delta"]]
        command["step"] = int(step)
        return command

    def step(self, command, n_render_steps=None):
        if n_render_steps is None:
            n_render_steps = self.render_frames
        target_delta = np.array(command["delta"], dtype=float)
        self.command_memory.append(target_delta)
        applied_commands = {}

        for arm in self.arms:
            applied = self._apply_behavior(arm, target_delta)
            if len(applied) != self.command_dim:
                raise ValueError(
                    "Applied command dimension must match scenario command dimension."
                )
            self._advance_joints(arm, applied)
            applied_commands[str(arm["index"])] = applied.tolist()
            self._update_arm_pose(arm)

        for _ in range(n_render_steps):
            self.world.step(render=True)

        return self._capture_rgb(), applied_commands

    def _initialize_panda_controls(self, arm):
        if arm["control_indices"] is not None:
            return

        control_joints = self.arm_cfg.get(
            "control_joints",
            ["panda_joint2", "panda_joint4"],
        )
        if len(control_joints) != self.command_dim:
            raise ValueError(
                "Panda control_joints length must match command delta dimension."
            )
        arm["control_indices"] = np.array(
            [arm["articulation"].get_dof_index(name) for name in control_joints],
            dtype=int,
        )

    def _advance_joints(self, arm, applied):
        if self.arm_root == "panda":
            self._initialize_panda_controls(arm)
            for command_axis, joint_index in enumerate(arm["control_indices"]):
                arm["joints"][joint_index] += float(applied[command_axis]) * self.command_step
            arm["joints"] = np.clip(
                arm["joints"],
                self.joint_limits[:, 0],
                self.joint_limits[:, 1],
            )
            return

        arm["joints"] = np.clip(
            arm["joints"] + applied * self.command_step,
            self.joint_limits[:, 0],
            self.joint_limits[:, 1],
        )

    def _apply_behavior(self, arm, target_delta):
        behavior = arm["behavior"]["behavior"]
        if behavior == "direct":
            return target_delta

        if behavior == "delay":
            delay = int(arm["behavior"].get("delay", 1))
            if len(self.command_memory) <= delay:
                return np.zeros(self.command_dim)
            return self.command_memory[-1 - delay]

        if behavior == "invert":
            return -target_delta

        if behavior == "axis_swap":
            permutation = arm["behavior"].get("permutation")
            if permutation is None:
                permutation = list(range(self.command_dim))
                for index in range(0, self.command_dim - 1, 2):
                    permutation[index], permutation[index + 1] = (
                        permutation[index + 1],
                        permutation[index],
                    )
            if len(permutation) != self.command_dim:
                raise ValueError("axis_swap permutation must match command dimension.")
            return target_delta[np.array(permutation, dtype=int)]

        if behavior in {"mapped_direct", "mapped_cycle_switch"}:
            return apply_mapped_behavior(
                arm["behavior"],
                target_delta,
                command_index=len(self.command_memory) - 1,
                command_dim=self.command_dim,
            )

        if behavior == "smooth":
            alpha = float(arm["behavior"].get("alpha", 0.5))
            arm["smooth_command"] = alpha * target_delta + (1.0 - alpha) * arm["smooth_command"]
            return arm["smooth_command"]

        if behavior == "random":
            random_index = int(self.rng.integers(0, len(self.command_library)))
            return self.command_library[random_index]

        if behavior in {"sequence_derangement", "sequence_mismatch"}:
            schedule = arm.get("command_schedule")
            if not schedule:
                raise RuntimeError(
                    f"{behavior} command schedule was not initialized."
                )
            command_index = len(self.command_memory) - 1
            return schedule[command_index % len(schedule)]

        raise ValueError(f"Unsupported distractor behavior: {behavior}")

    def close(self, close_simulation_app=True):
        """Release this episode while optionally keeping Isaac Sim alive.

        Dataset generation reuses one SimulationApp for many independent
        episodes. Clearing the World singleton prevents prims, callbacks, and
        physics state from leaking into the next episode.
        """
        try:
            if getattr(self, "world", None) is not None:
                self.world.stop()
                self.world.clear()
                World.clear_instance()
        finally:
            if close_simulation_app:
                self.sim_app.close()
