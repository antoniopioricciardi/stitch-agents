"""ManiSkill3 PickCube with visual / task / robot variation axes.

make_env(visual=(cam, look, light), task, robot). Index 0 on every visual axis is the ManiSkill default.
Visual variants never change the set of actors, so a state saved in one visual variant can be
re-rendered in any other (env.set_state(s) + render): this gives the paired frames for SAPS.
"""
import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

# visual axis 1: sensor camera (eye, target). Robot base is at x = -0.615, cubes spawn in [-0.1, 0.1]^2.
CAMERAS = [
    ([0.3, 0.0, 0.6], [-0.1, 0.0, 0.1]),     # 0 default: in front of the robot, above
    ([0.0, 0.6, 0.45], [-0.05, 0.0, 0.1]),   # 1 left side
    ([0.55, -0.2, 0.25], [-0.1, 0.0, 0.15]), # 2 low front-right
]

# visual axis 2: look = colours / textures of cube, table, floor. None = ManiSkill default
# (red cube, wooden table, grid floor). 'checker' = procedural checkerboard texture.
LOOKS = [
    None,
    dict(cube=[1.0, 0.85, 0.0, 1], table=[0.25, 0.25, 0.28, 1], floor=[0.85, 0.8, 0.7, 1]),
    dict(cube=[0.0, 0.9, 1.0, 1], table="checker", floor=[0.3, 0.2, 0.35, 1]),
]

# visual axis 3: lighting. 0 = ManiSkill default, 1 = dim, warm, low light from the left with shadows.
N_LIGHTS = 2

# task axis. Cube friction/density: ManiSkill's default (0.3 static/dynamic, 1000 kg/m^3 -> 0.064 kg).
# arm_scale multiplies the EE-delta controller's action bounds (pos and rot, default +-0.1): with 0.5 the same
# normalized action moves the arm half as far, so the expert needs different action values for the same motion.
# A cube-physics variant (friction 0.1, 10x mass) was dropped: the open-loop planner's actions did not change
# (EXPERIMENTS.md, Step 1).
TASKS = {
    "default": dict(friction=0.3, density=1000.0, arm_scale=1.0),
    "actuation": dict(friction=0.3, density=1000.0, arm_scale=0.5),
    "goal": dict(friction=0.3, density=1000.0, arm_scale=1.0),  # goal region moved to the robot's left, see below
}
# goal variant: goal y in [0.15, 0.25] instead of [-0.1, 0.1] (disjoint from the default), x and z as default.
GOAL_Y = (0.15, 0.25)

ROBOTS = {"panda": "panda", "xarm6": "xarm6_robotiq"}

# goal marker "lollipop": sphere of half the goal threshold (0.025 m) on a vertical pole. Magenta: distinct from the
# three cube colours (red, yellow, cyan) and from every table/floor colour.
LOLLIPOP = dict(radius=0.0125, pole_radius=0.006, pole_length=0.4, color=[1.0, 0.0, 1.0, 1])  # 12 mm pole: 8 mm aliased to dashes in cam1


def checker_texture(n=(24, 48), px=8):
    # (n0*px, n1*px, 4) uint8 checkerboard, light/dark grey. The box UVs span a whole face, so on the
    # 2.42 x 1.21 m table top 48 x 24 squares are ~5 cm each. Mipmaps limit aliasing at 128x128.
    board = (np.indices(n).sum(0) % 2).repeat(px, 0).repeat(px, 1)
    g = np.where(board, 200, 60).astype(np.uint8)
    return sapien.render.RenderTexture2D(np.stack([g, g, g, np.full_like(g, 255)], -1), "R8G8B8A8Unorm", mipmap_levels=6)


def material(spec):
    if isinstance(spec, str):  # 'checker'
        m = sapien.render.RenderMaterial(base_color=[1, 1, 1, 1])
        m.set_base_color_texture(checker_texture())
        return m
    return sapien.render.RenderMaterial(base_color=spec)


def replace_visual(actor, half_size, local_p, spec):
    # swap the actor's render body for a box. Editing the loaded glb material in place has no effect.
    for obj in actor._objs:
        obj.remove_component(obj.find_component_by_type(sapien.render.RenderBodyComponent))
        shape = sapien.render.RenderShapeBox(half_size, material(spec))
        shape.local_pose = sapien.Pose(p=local_p)
        body = sapien.render.RenderBodyComponent()
        body.attach(shape)
        obj.add_component(body)


class StitchPickCubeEnv(PickCubeEnv):
    def __init__(self, *args, cam=0, look=0, light=0, task="default", goal_marker="sphere", goal_in_state=True, grasp_in_state=True, **kwargs):
        # set before super().__init__, which builds the scene.
        # goal_marker: "sphere" = PickCube's green goal sphere, visible to the camera; "hidden" = as PickCube
        # (sensor cameras don't render it; diagnostic for the Step 2c gap); "lollipop" = see LOLLIPOP
        # goal_in_state=False / grasp_in_state=False drop goal_pos / is_grasped from the observation's "extra" (PickCube
        # puts them there; without goal_pos the goal must be read from pixels). ManiSkill's DP baseline takes its state
        # from agent + extra.
        self.cam, self.look, self.light, self.task, self.goal_marker = cam, look, light, task, goal_marker
        self.goal_in_state, self.grasp_in_state = goal_in_state, grasp_in_state
        super().__init__(*args, **kwargs)
        if "ee_delta" in self.control_mode:
            # normalized action a in [-1, 1] -> delta = a * bound; the replay conversion reads the same bounds
            arm = self.agent.controller.controllers["arm"]
            arm.action_space_low *= TASKS[task]["arm_scale"]
            arm.action_space_high *= TASKS[task]["arm_scale"]

    @property
    def _default_sensor_configs(self):
        eye, target = CAMERAS[self.cam]
        pose = sapien_utils.look_at(eye=eye, target=target)
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    def _load_lighting(self, options):
        if self.light == 0:
            return super()._load_lighting(options)
        self.scene.set_ambient_light([0.08, 0.07, 0.06])
        self.scene.add_directional_light([0.3, -1.0, -0.4], [1.6, 1.1, 0.6], shadow=True, shadow_scale=5, shadow_map_size=2048)

    def _load_scene(self, options):
        # same as PickCubeEnv._load_scene, plus cube physics, look, and a goal sphere visible to the camera
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        look = LOOKS[self.look]
        phys = TASKS[self.task]

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=[self.cube_half_size] * 3,
            material=sapien.physx.PhysxMaterial(phys["friction"], phys["friction"], 0.0),
            density=phys["density"],
        )
        builder.add_box_visual(half_size=[self.cube_half_size] * 3, material=material(look["cube"] if look else [1, 0, 0, 1]))
        builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
        self.cube = builder.build(name="cube")

        if self.goal_marker == "lollipop":
            # small sphere at the goal on a thin pole down to the table: marks the goal's height without covering
            # the cube (the full-size sphere covered cube/gripper pixels in ~37% of cam0 frames, Step 2e).
            # One kinematic actor; the pole is a fixed LOLLIPOP["pole_length"] long, the part below the table top
            # is hidden inside the table. No collision.
            builder = self.scene.create_actor_builder()
            mat = sapien.render.RenderMaterial(base_color=LOLLIPOP["color"])
            builder.add_sphere_visual(radius=LOLLIPOP["radius"], material=mat)
            half = LOLLIPOP["pole_length"] / 2
            # sapien cylinders lie along x: rotate 90 deg about y to make the pole vertical
            builder.add_cylinder_visual(radius=LOLLIPOP["pole_radius"], half_length=half, material=mat,
                                        pose=sapien.Pose(p=[0, 0, -half], q=[np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]))
            builder.initial_pose = sapien.Pose()
            self.goal_site = builder.build_kinematic(name="goal_site")
        else:
            self.goal_site = actors.build_sphere(
                self.scene, radius=self.goal_thresh, color=[0, 1, 0, 1], name="goal_site",
                body_type="kinematic", add_collision=False, initial_pose=sapien.Pose(),
            )
        if self.goal_marker == "hidden":
            self._hidden_objects.append(self.goal_site)

        if look:
            # table actor frame is rotated 90 deg about z; box matches the table's collision box
            replace_visual(self.table_scene.table, [1.209, 0.6045, 0.4598], [0, 0, 0.4598], look["table"])
            # ground actor sits at the origin; its mesh carries the altitude (-table_height)
            replace_visual(self.table_scene.ground, [10, 10, 0.01], [0, 0, -self.table_scene.table_height - 0.01], look["floor"])

    def _get_obs_extra(self, info):
        extra = super()._get_obs_extra(info)
        if not self.goal_in_state:
            del extra["goal_pos"]
        if not self.grasp_in_state:
            del extra["is_grasped"]
        return extra

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if self.task == "goal":
            with torch.device(self.device):
                b = len(env_idx)
                p = self.goal_site.pose.p.clone()
                p[:, 1] = torch.rand(b) * (GOAL_Y[1] - GOAL_Y[0]) + GOAL_Y[0]
                self.goal_site.set_pose(Pose.create_from_pq(p))


# 160 steps: ~2x the mean converted demo length (~78 steps; ManiSkill's advice for imitation learning)
register_env("StitchPickCube-v1", max_episode_steps=160)(StitchPickCubeEnv)
# same env with the goal sphere hidden from the camera: an env id, because the DP baseline builds envs from the id alone
register_env("StitchPickCubeHiddenGoal-v1", max_episode_steps=160, goal_marker="hidden")(StitchPickCubeEnv)
register_env("StitchPickCubeLollipop-v1", max_episode_steps=160, goal_marker="lollipop")(StitchPickCubeEnv)
register_env("StitchPickCubeLollipopNoGoalState-v1", max_episode_steps=160, goal_marker="lollipop", goal_in_state=False, grasp_in_state=False)(StitchPickCubeEnv)
# core oracle setup (Step 2g): goal position in the state, no is_grasped
register_env("StitchPickCubeLollipopNoGrasp-v1", max_episode_steps=160, goal_marker="lollipop", grasp_in_state=False)(StitchPickCubeEnv)
# the same core setup seen from cam1 (left side; Step 1b)
register_env("StitchPickCubeLollipopNoGraspCam1-v1", max_episode_steps=160, goal_marker="lollipop", grasp_in_state=False, cam=1)(StitchPickCubeEnv)
register_env("StitchPickCubeLollipopNoGraspCam2-v1", max_episode_steps=160, goal_marker="lollipop", grasp_in_state=False, cam=2)(StitchPickCubeEnv)
register_env("StitchPickCubeLollipopNoGraspLook1-v1", max_episode_steps=160, goal_marker="lollipop", grasp_in_state=False, look=1)(StitchPickCubeEnv)
# cam2 with the goal variant (goal y in GOAL_Y, disjoint from the cube area): "go to goal_pos" cannot find the cube
register_env("StitchPickCubeLollipopNoGraspCam2Goal-v1", max_episode_steps=160, goal_marker="lollipop", grasp_in_state=False, cam=2, task="goal")(StitchPickCubeEnv)


def make_env(visual=(0, 0, 0), task="default", robot="panda", obs_mode="rgb", control_mode="pd_joint_pos", goal_marker="sphere"):
    cam, look, light = visual
    return gym.make(
        "StitchPickCube-v1", cam=cam, look=look, light=light, task=task, robot_uids=ROBOTS[robot], goal_marker=goal_marker,
        obs_mode=obs_mode, control_mode=control_mode, sim_backend="cpu", render_mode="rgb_array",
    )


def proprio(env):
    # (P,) robot-only proprioception: qpos, qvel, tcp pose (7). Panda: P = 9 + 9 + 7 = 25.
    # No goal position: the encoder has to read the goal from pixels.
    a = env.unwrapped.agent
    return torch.cat([a.robot.get_qpos()[0], a.robot.get_qvel()[0], a.tcp.pose.raw_pose[0]]).cpu().numpy()
