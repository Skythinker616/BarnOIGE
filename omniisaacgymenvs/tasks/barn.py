import math

import numpy as np
import torch
import carb
from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrim, RigidPrimView, RigidContactView
from omni.isaac.core.prims import XFormPrim, XFormPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse, get_euler_xyz
from omni.isaac.core.utils.torch.maths import torch_rand_float
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.range_sensor import _range_sensor
from omni.isaac.sensor import _sensor

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.sensor")   # required by OIGE
from typing import Optional, Tuple # type: ignore
from omni.isaac.sensor import RotatingLidarPhysX, ContactSensor, LidarRtx

import time

# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt # use to draw lidar[0] scan

''' unused
class RotatingLidar(RotatingLidarPhysX):
    def __init__(
        self,
        prim_path: str,
        name: str = "rotating_lidar_physX",
        rotation_frequency: Optional[float] = None,
        rotation_dt: Optional[float] = None,
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        fov: Optional[Tuple[float, float]] = None,
        resolution: Optional[Tuple[float, float]] = None,
        valid_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(prim_path, name, rotation_frequency, rotation_dt, position, translation, orientation, fov, resolution, valid_range)
        self._hori_fov = fov[0]
        self._hori_num = int(fov[0] / resolution[0])
        self._hori_start_angle = ((360 - fov[0]) / 2 - 180) * math.pi / 180
        self._hori_end_angle = (180 - (360 - fov[0]) / 2) * math.pi / 180
        self._depth_buffer = np.zeros((self._hori_num, ), dtype=np.float32)

    def _data_acquisition_callback(self, step_size: float) -> None:
        self._current_time += step_size
        self._number_of_physics_steps += 1
        if not self._pause:
            depth = self._lidar_sensor_interface.get_linear_depth_data(self.prim_path).reshape(-1)

            if len(depth) > 0:
                azimuth = self._lidar_sensor_interface.get_azimuth_data(self.prim_path)

                start_index = int(self._hori_num * (azimuth[0] - self._hori_start_angle) / (self._hori_fov * math.pi / 180))
            
                if start_index + len(depth) > self._hori_num:
                    self._depth_buffer[start_index:] = depth[:self._hori_num - start_index]
                    self._depth_buffer[:len(depth) - (self._hori_num - start_index)] = depth[self._hori_num - start_index:]
                else:
                    self._depth_buffer[start_index:start_index + len(depth)] = depth

                self._current_frame["linear_depth"] = self._backend_utils.create_tensor_from_list(
                    self._depth_buffer, dtype="float32", device=self._device
                )

            self._current_frame["physics_step"] = self._number_of_physics_steps
            self._current_frame["time"] = self._current_time
        return

class LidarView:
    def __init__(self,
                 prim_paths: list[str],
                 name: Optional[str] = "LidarView") -> None:
        self._lidars = []
        for prim_path in prim_paths:
            lidar = RotatingLidar(prim_path=prim_path, name=name, rotation_frequency=100, fov=(270, 10), resolution=(1.5, 0.4), valid_range=(0.05, 10.0))
            lidar.add_linear_depth_data_to_frame()
            lidar.add_azimuth_data_to_frame()
            lidar.enable_visualization(high_lod=False,
                                       draw_points=False,
                                       draw_lines=True)
            lidar.initialize()
            self._lidars.append(lidar)
        
    def get_observation(self):
        observations = None
        for lidar in self._lidars:
            if observations is None:
                observations = lidar.get_current_frame()["linear_depth"].unsqueeze(-1)
            else:
                observations = torch.cat((observations, lidar.get_current_frame()["linear_depth"].unsqueeze(-1)), dim=-1)
        return observations.T
'''

class RTXLidar(LidarRtx):
    def __init__(self, prim_path: str, name: Optional[str] = "RTXLidar") -> None:
        super().__init__(prim_path=prim_path, name=name)
        
    def _data_acquisition_callback(self, event: carb.events.IEvent):
        super()._data_acquisition_callback(event)
        self._current_frame["linear_depth_data"] = self._backend_utils.create_tensor_from_list(
            self._current_frame["linear_depth_data"], dtype="float32", device=self._device
        )

class RTXLidarView:
    def __init__(self,
                 prim_paths: list[str],
                 div_num: int,
                 name: Optional[str] = "RTXLidarView") -> None:
        self._lidars = []
        for prim_path in prim_paths:
            lidar = RTXLidar(prim_path=prim_path)
            lidar.add_linear_depth_data_to_frame()
            # lidar.enable_visualization()
            lidar.initialize()
            self._lidars.append(lidar)
        self._div_num = div_num

    def _get_single_observation(self, lidar):
        obs = lidar.get_current_frame()["linear_depth_data"]
        raw_len = obs.shape[0]
        if raw_len == 0:
            return torch.zeros((self._div_num, ), dtype=torch.float32, device=lidar._device)
        div_len = raw_len // self._div_num
        obs = obs[:div_len * self._div_num].view(self._div_num, div_len).min(dim=1).values
        return obs

    def get_observation(self):
        observations = None
        for lidar in self._lidars:
            if observations is None:
                observations = self._get_single_observation(lidar).unsqueeze(-1)
            else:
                observations = torch.cat((observations, self._get_single_observation(lidar).unsqueeze(-1)), dim=-1)
        return observations.T

class ContactSensorView:
    def __init__(self,
                 prim_paths: list[str],
                 name: Optional[str] = "ContactSensorView") -> None:
        self._contact_sensors = []
        for prim_path in prim_paths:
            contact_sensor = ContactSensor(prim_path=prim_path, name=name)
            contact_sensor.add_raw_contact_data_to_frame()
            contact_sensor.initialize()
            self._contact_sensors.append(contact_sensor)
        self._contact_nums = np.zeros((len(self._contact_sensors), ), dtype=np.int32)
        
    def get_observation(self):
        for i in range(len(self._contact_sensors)):
            readings = self._contact_sensors[i]._contact_sensor_interface.get_sensor_reading(self._contact_sensors[i].prim_path)
            if readings.is_valid and readings.in_contact:
                self._contact_nums[i] = 1
            else:
                self._contact_nums[i] = 0
        return self._contact_sensors[0]._backend_utils.create_tensor_from_list(
            self._contact_nums, dtype="int32", device=self._contact_sensors[0]._device
        )

class BarnTask(RLTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        self._max_episode_length = 300
        self._lidar_div_num = 256

        self._num_observations = self._lidar_div_num + 2 + 2
        self._num_actions = 2

        RLTask.__init__(self, name, env)

        # plt.ion()
        return
    
    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_vel_forward = self._task_cfg["env"]["maxVelForward"]
        self._max_vel_spin = self._task_cfg["env"]["maxVelSpin"]

        self._target_pos = list(self._task_cfg["env"]["targetPos"])

        self._jackal_usd = r"../barn_utils/usd/jackal.usd"
        self._world_usd = r"../barn_utils/usd/worlds/out_world_%d.usd"

    def get_jackal(self):
        prim_path = self.default_zero_env_path + "/jackal"
        add_reference_to_stage(self._jackal_usd, prim_path)
        jackal = Robot(
            prim_path = prim_path,
            name = "jackal",
            translation = np.array([-2.5, 2.5, 0.05]),
            orientation = np.array([0.707, 0, 0, 0.707]),
        )
        self._sim_config.apply_articulation_settings(
            "jackal", get_prim_at_path(jackal.prim_path), self._sim_config.parse_actor_config("jackal")
        )

    def get_lidar_view(self):
        prim_paths = []
        for i in range(self._num_envs):
            prim_paths.append(f"/World/envs/env_{i}/jackal/base_link/sick_lms1xx_lidar_frame/rtx_lidar1")
        self._lidars = RTXLidarView(
            prim_paths = prim_paths,
            div_num = self._lidar_div_num,
        )

    def get_contact_sensor_view(self):
        prim_paths = []
        for i in range(self._num_envs):
            prim_paths.append(f"/World/envs/env_{i}/jackal/base_link/visuals/mesh_0/contact_sensor")
        self._contact_sensors = ContactSensorView(
            prim_paths = prim_paths,
            name = "contact_sensor",
        )

    def get_world(self, world_index = 0):
        prim_path = self.default_zero_env_path + "/world"
        add_reference_to_stage(self._world_usd.replace("%d", str(world_index)), prim_path)
        world = XFormPrim(
            prim_path = prim_path,
            name = "world",
        )

    def get_worlds(self):
        for index in range(0, self._num_envs):
            prim_path = f"/World/worlds/world_{index}"
            add_reference_to_stage(self._world_usd.replace("%d", str(index)), prim_path)
            world = XFormPrim(
                prim_path = prim_path,
                name = "world" + str(index),
                translation = np.array(self._env_pos[index].cpu()),
            )

    def set_up_scene(self, scene) -> None:
        self.get_jackal()

        # self.get_world(0) # uncomment to clone the world to each env
        super().set_up_scene(scene)
        self.get_worlds() # uncomment to set world_0, world_1, ... to each env

        self._jackals = ArticulationView(
            prim_paths_expr="/World/envs/.*/jackal", name="jackal_view", reset_xform_properties=False
        )
        scene.add(self._jackals)
        self.get_lidar_view()
        self.get_contact_sensor_view()

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("jackal_view"):
            scene.remove_object("jackal_view", registry_only=True)
        self._jackals = ArticulationView(
            prim_paths_expr="/World/envs/.*/jackal", name="jackal_view", reset_xform_properties=False
        )
        scene.add(self._jackals)
        self.get_lidar_view()
        self.get_contact_sensor_view()

    def get_observations(self) -> dict:
        self._lidar_obs = self._lidars.get_observation()
        # print("lidar[0] obs shape:", self._lidar_obs[0].shape)
        self._contact_obs = self._contact_sensors.get_observation()

        target_pos = torch.tensor(self._target_pos, dtype=torch.float32, device=self._device)
        world_pos, world_rot = self._jackals.get_world_poses(clone=False)
        local_pos = world_pos - self._env_pos
        world_pos_err = target_pos - local_pos
        local_pos_err = quat_rotate_inverse(world_rot, world_pos_err)
        local_pos_err_norm = torch.norm(local_pos_err, dim=-1)
        local_pos_err_angle = torch.atan2(local_pos_err[:, 1], local_pos_err[:, 0])
        self._pos_err = torch.stack((local_pos_err_norm, local_pos_err_angle), dim=-1)
        # print("pos_err:", self._pos_err[0].tolist())

        if self._last_pos_err is not None:
            self._pos_err_diff[:, 0] = self._pos_err[:, 0] - self._last_pos_err[:, 0]
            self._pos_err_diff[:, 1] = torch.abs(self._pos_err[:, 1]) - torch.abs(self._last_pos_err[:, 1])
        else:
            self._pos_err_diff = torch.zeros((self._num_envs, 2), dtype=torch.float32, device=self._device)
        self._last_pos_err = self._pos_err

        roll, pitch, yaw = get_euler_xyz(world_rot)
        roll = torch.where(roll > math.pi, roll - 2 * math.pi, roll)
        pitch = torch.where(pitch > math.pi, pitch - 2 * math.pi, pitch)
        self._euler = torch.stack((roll, pitch, yaw), dim=-1)

        # print("error:", self._pos_err[0].tolist(), "euler:", (self._euler[0] * 180 / math.pi).tolist())

        # plot lidar[0] scan
        # plt.clf()
        # plt.subplot(111, polar=True)
        # plt.scatter(np.linspace(0, 2 * math.pi, self._lidar_obs[0].shape[0]), self._lidar_obs[0].cpu().numpy(), s=1)
        # plt.scatter(self._pos_err.cpu().numpy()[0, 1], self._pos_err.cpu().numpy()[0, 0], c="r")
        # plt.pause(0.05)

        # print lidar[0] scan
        # print("lidar_size:", self._lidars._lidars[0]._current_frame["linear_depth_data"].shape, self._lidars._lidars[0]._current_frame["azimuth"].shape)
        # print("lidar:", self._lidars._lidars[0]._current_frame["linear_depth_data"], self._lidars._lidars[0]._current_frame["azimuth"])

        # return torch.zeros((self._num_envs, self._num_observations), dtype=torch.float32, device=self._device) # for testing

        obs = torch.cat(
            (
                self._lidar_obs * 0.1,
                local_pos_err[:, 0].unsqueeze(-1) * 0.1,
                local_pos_err[:, 1].unsqueeze(-1) / math.pi,
                self._last_action,
            ),
            dim=-1,
        )

        self.obs_buf[:] = obs
        observations = {self._jackals.name: {"obs_buf": self.obs_buf}}
        return observations
    
    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)
        actions = torch.clip(actions, -1, 1)

        actions[:, 0] = (actions[:, 0] - (-1)) / (1 - (-1)) * 0.9 + 0.1

        # actions = torch.zeros((self._num_envs, 2), dtype=torch.float32, device=self._device) # for testing
        # actions[:, 0] = 1
        # actions[:, 1] = 0.5

        speed = torch.cat(
            (
                actions[:, 0].unsqueeze(-1) * self._max_vel_forward - actions[:, 1].unsqueeze(-1) * self._max_vel_spin,
                actions[:, 0].unsqueeze(-1) * self._max_vel_forward + actions[:, 1].unsqueeze(-1) * self._max_vel_spin,
                actions[:, 0].unsqueeze(-1) * self._max_vel_forward - actions[:, 1].unsqueeze(-1) * self._max_vel_spin,
                actions[:, 0].unsqueeze(-1) * self._max_vel_forward + actions[:, 1].unsqueeze(-1) * self._max_vel_spin,
            ),
            dim=-1,
        )

        self._last_action = actions

        indices = torch.arange(self._jackals.count, dtype=torch.int32, device=self._device)
        self._jackals.set_joint_velocities(speed, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        indices = env_ids.to(dtype=torch.int32)
        self._jackals.set_world_poses(
            self.initial_root_pos[env_ids].clone(), self.initial_root_rot[env_ids].clone(), indices
        )
        root_vel = torch.zeros((num_resets, 6), device=self._device)
        self._jackals.set_velocities(root_vel, indices)
        dof_vel = torch.zeros((num_resets, 4), device=self._device)
        self._jackals.set_joint_velocities(dof_vel, indices)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self._lf_dof_idx = self._jackals.get_dof_index("front_left_wheel_joint")
        self._rf_dof_idx = self._jackals.get_dof_index("front_right_wheel_joint")
        self._lr_dof_idx = self._jackals.get_dof_index("rear_left_wheel_joint")
        self._rr_dof_idx = self._jackals.get_dof_index("rear_right_wheel_joint")

        self.initial_root_pos, self.initial_root_rot = self._jackals.get_world_poses()

        self._last_action = torch.zeros((self._num_envs, self._num_actions), dtype=torch.float32, device=self._device)
        self._last_pos_err = None

        indices = torch.arange(self._jackals.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        rewards = - self._pos_err_diff[:, 0] * 0.1 # - self._pos_err_diff[:, 1] * 0.05
        rewards = rewards + torch.clip(self._last_action[:, 0], -1, 0) * 0.01
        rewards = torch.where(self._contact_obs > 0, -0.2, rewards)
        rewards = torch.where((torch.abs(self._euler[:, 0]) > math.pi / 6) | (torch.abs(self._euler[:, 1]) > math.pi / 6), -0.2, rewards)
        rewards = torch.where(self._pos_err[:, 0] < 1, 1, rewards)
        rewards = rewards - 0.001
        # resets = torch.where(self.progress_buf >= self._max_episode_length, -0.2, resets)
        # print("reward:", rewards[0].item())
        self.rew_buf[:] = rewards

    def is_done(self) -> None:
        resets = torch.where(self._contact_obs > 0, 1, 0)
        resets = torch.where(self._pos_err[:, 0] < 1, 1, resets)
        resets = torch.where((torch.abs(self._euler[:, 0]) > math.pi / 6) | (torch.abs(self._euler[:, 1]) > math.pi / 6), 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets