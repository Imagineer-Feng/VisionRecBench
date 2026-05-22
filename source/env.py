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
            requested_target = self.task_dict.get("target_index")
            if requested_target is None:
                self.target_index = int(self.rng.integers(1, self.num_arms + 1))
            else:
                self.target_index = int(requested_target)
            if not 1 <= self.target_index <= self.num_arms:
                raise ValueError("target_index must be within [1, num_arms].")
            self.target_present = True
            self.answer_options = [
                f"candidate arm {i} from left to right"
                for i in range(1, self.num_arms + 1)
            ]
            self.answer_index = self.target_index

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
            option_index = int(self.rng.integers(0, len(behavior_options)))
            selected = copy.deepcopy(behavior_options[option_index])
            self.task_dict["visible_arm_behavior"] = selected["behavior"]
            self.task_dict["target_present"] = bool(selected["target_present"])
            self.task_dict["sampled_behavior_option"] = option_index + 1

        self.target_present = bool(self.task_dict.get("target_present", True))
        self.target_index = 1 if self.target_present else None
        self.answer_options = list(
            self.task_dict.get(
                "answer_options",
                [
                    "yes, the visible arm is myself",
                    "no, the visible arm is not myself",
                ],
            )
        )
        if len(self.answer_options) != 2:
            raise ValueError("single_binary scenarios must define exactly two answer options.")
        self.answer_index = 1 if self.target_present else 2

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

    def _create_arms(self):
        spacing = float(self.task_dict["layout_spacing"])
        x_offset = (self.num_arms - 1) * spacing / 2.0
        distractors = iter(self.task_dict.get("distractors", []))

        for index in range(1, self.num_arms + 1):
            if self.task_mode == "single_binary":
                behavior = copy.deepcopy(self.task_dict["visible_arm_behavior"])
                role = "target" if self.target_present else "non_target"
            elif index == self.target_index:
                behavior = {
                    "behavior": "direct",
                    "desc": "the target arm that follows the motor command directly",
                }
                role = "target"
            else:
                behavior = copy.deepcopy(next(distractors))
                role = "distractor"

            base_pos = np.array([(index - 1) * spacing - x_offset, 0.0, 0.0])
            arm = {
                "index": index,
                "role": role,
                "behavior": behavior,
                "base_pos": base_pos,
                "joints": self.initial_joints.copy(),
                "smooth_command": np.zeros(2),
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
            arm["smooth_command"] = np.zeros(2)
            if self.arm_root == "panda":
                self._initialize_panda_controls(arm)
            self._update_arm_pose(arm)

        self.camera.initialize()
        for _ in range(self.warmup_frames):
            self.world.step(render=True)
        return self._capture_rgb()

    def get_command(self, step):
        command = copy.deepcopy(self.command_sequence[(step - 1) % len(self.command_sequence)])
        command["delta"] = [float(command["delta"][0]), float(command["delta"][1])]
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
                return np.zeros(2)
            return self.command_memory[-1 - delay]

        if behavior == "invert":
            return -target_delta

        if behavior == "axis_swap":
            return np.array([target_delta[1], target_delta[0]], dtype=float)

        if behavior == "mapped_direct":
            mapping = np.array(arm["behavior"]["mapping"], dtype=float)
            if mapping.shape != (2, 2):
                raise ValueError("mapped_direct behavior requires a 2x2 mapping matrix.")
            return mapping @ target_delta

        if behavior == "smooth":
            alpha = float(arm["behavior"].get("alpha", 0.5))
            arm["smooth_command"] = alpha * target_delta + (1.0 - alpha) * arm["smooth_command"]
            return arm["smooth_command"]

        if behavior == "random":
            random_index = int(self.rng.integers(0, len(self.command_library)))
            return self.command_library[random_index]

        raise ValueError(f"Unsupported distractor behavior: {behavior}")

    def close(self):
        self.sim_app.close()
