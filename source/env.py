import copy
import math

import numpy as np

import carb
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


class VisionRecBenchEnv:
    def __init__(
        self,
        sim_app,
        task_dict,
        renderer="RayTracedLighting",
        resolution=(768, 768),
        warmup_frames=12,
        render_frames=6,
    ):
        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", renderer)
        if renderer == "PathTracing":
            settings.set("/rtx/pathtracing/spp", 1)

        self.sim_app = sim_app
        self.task_dict = copy.deepcopy(task_dict)
        self.renderer = renderer
        self.resolution = tuple(resolution)
        self.warmup_frames = int(warmup_frames)
        self.render_frames = int(render_frames)
        self.arm_cfg = self.task_dict["arm"]
        self.num_arms = int(self.task_dict["num_arms"])
        self.episode_steps = int(self.task_dict["episode_steps"])
        self.rng = np.random.default_rng(int(self.task_dict.get("seed", 0)))

        requested_target = self.task_dict.get("target_index")
        if requested_target is None:
            self.target_index = int(self.rng.integers(1, self.num_arms + 1))
        else:
            self.target_index = int(requested_target)
        if not 1 <= self.target_index <= self.num_arms:
            raise ValueError("target_index must be within [1, num_arms].")

        self.world = World(stage_units_in_meters=1.0)
        self.stage = self.world.stage

        self.link_lengths = np.array(self.arm_cfg["link_lengths"], dtype=float)
        self.link_thickness = float(self.arm_cfg["link_thickness"])
        self.base_size = np.array(self.arm_cfg["base_size"], dtype=float)
        self.wrist_size = np.array(self.arm_cfg["wrist_size"], dtype=float)
        self.initial_joints = np.array(self.arm_cfg["initial_joints_deg"], dtype=float)
        self.joint_limits = np.array(self.arm_cfg["joint_limits_deg"], dtype=float)
        self.command_step_deg = float(self.arm_cfg["command_step_deg"])
        self.command_sequence = copy.deepcopy(self.task_dict["command_sequence"])
        self.command_library = [
            np.array(item["delta"], dtype=float) for item in self.command_sequence
        ]

        self.command_memory = []
        self.arms = []
        self._create_scene()
        self._create_arms()
        self._create_camera()

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

    def _create_scene(self):
        floor_width = max(4.0, self.num_arms * float(self.task_dict["layout_spacing"]) + 1.0)
        FixedCuboid(
            prim_path="/World/Floor",
            name="floor",
            position=np.array([0.0, 0.25, -0.015]),
            size=1.0,
            scale=np.array([floor_width, 2.4, 0.03]),
        )
        self._create_and_bind_material(
            "/World/Floor",
            "/World/Looks/FloorMaterial",
            color=[0.52, 0.55, 0.56],
            metallic=0.0,
            roughness=0.7,
        )

        light = UsdLux.DistantLight.Define(self.stage, Sdf.Path("/World/KeyLight"))
        light.CreateIntensityAttr(4500.0)
        light.CreateAngleAttr(0.45)
        UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 35.0))

    def _create_arms(self):
        spacing = float(self.task_dict["layout_spacing"])
        x_offset = (self.num_arms - 1) * spacing / 2.0
        distractors = iter(self.task_dict["distractors"])

        for index in range(1, self.num_arms + 1):
            if index == self.target_index:
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
        self.camera = Camera(
            prim_path="/World/Camera",
            translation=np.array([0.0, -4.0, 2.3]),
            frequency=20,
            resolution=self.resolution,
        )
        set_camera_view(
            eye=[0.0, -3.9, 2.15],
            target=[0.0, 0.25, 0.55],
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
            arm["joints"] = np.clip(
                arm["joints"] + applied * self.command_step_deg,
                self.joint_limits[:, 0],
                self.joint_limits[:, 1],
            )
            applied_commands[str(arm["index"])] = applied.tolist()
            self._update_arm_pose(arm)

        for _ in range(n_render_steps):
            self.world.step(render=True)

        return self._capture_rgb(), applied_commands

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
