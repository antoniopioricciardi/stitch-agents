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
    dict(cube=[0.1, 0.3, 1.0, 1], table="checker", floor=[0.3, 0.2, 0.35, 1]),
]

# visual axis 3: lighting. 0 = ManiSkill default, 1 = dim, warm, low light from the left with shadows.
N_LIGHTS = 2

# task axis. Cube friction/density: default is ManiSkill's (0.3 static/dynamic, 1000 kg/m^3 -> 0.064 kg).
# Note: Panda fingers have friction 2.0 and PhysX averages the two materials, so cube friction mostly
# matters for cube-table contact.
TASKS = {
    "default": dict(friction=0.3, density=1000.0),
    "physics": dict(friction=0.1, density=10000.0),  # slippery and 10x heavier (0.64 kg)
    "goal": dict(friction=0.3, density=1000.0),      # goal region moved to the robot's left, see below
}
# goal variant: goal y in [0.15, 0.25] instead of [-0.1, 0.1] (disjoint from the default), x and z as default.
GOAL_Y = (0.15, 0.25)

ROBOTS = {"panda": "panda", "xarm6": "xarm6_robotiq"}


def checker_texture(n=8, px=32):
    # (n*px, n*px, 4) uint8 checkerboard, light/dark grey
    board = (np.indices((n, n)).sum(0) % 2).repeat(px, 0).repeat(px, 1)
    g = np.where(board, 200, 60).astype(np.uint8)
    return sapien.render.RenderTexture2D(np.stack([g, g, g, np.full_like(g, 255)], -1), "R8G8B8A8Unorm")


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
    def __init__(self, *args, cam=0, look=0, light=0, task="default", **kwargs):
        # set before super().__init__, which builds the scene
        self.cam, self.look, self.light, self.task = cam, look, light, task
        super().__init__(*args, **kwargs)

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

        self.goal_site = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[0, 1, 0, 1], name="goal_site",
            body_type="kinematic", add_collision=False, initial_pose=sapien.Pose(),
        )
        # not added to self._hidden_objects: the encoder has to see the goal

        if look:
            # table actor frame is rotated 90 deg about z; box matches the table's collision box
            replace_visual(self.table_scene.table, [1.209, 0.6045, 0.4598], [0, 0, 0.4598], look["table"])
            # ground actor sits at the origin; its mesh carries the altitude (-table_height)
            replace_visual(self.table_scene.ground, [10, 10, 0.01], [0, 0, -self.table_scene.table_height - 0.01], look["floor"])

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if self.task == "goal":
            with torch.device(self.device):
                b = len(env_idx)
                p = self.goal_site.pose.p.clone()
                p[:, 1] = torch.rand(b) * (GOAL_Y[1] - GOAL_Y[0]) + GOAL_Y[0]
                self.goal_site.set_pose(Pose.create_from_pq(p))


# 100 steps: motion-planning demos take ~85-90 control steps, more than PickCube's default 50
register_env("StitchPickCube-v1", max_episode_steps=100)(StitchPickCubeEnv)


def make_env(visual=(0, 0, 0), task="default", robot="panda", obs_mode="rgb", control_mode="pd_joint_pos"):
    cam, look, light = visual
    return gym.make(
        "StitchPickCube-v1", cam=cam, look=look, light=light, task=task, robot_uids=ROBOTS[robot],
        obs_mode=obs_mode, control_mode=control_mode, sim_backend="cpu", render_mode="rgb_array",
    )
