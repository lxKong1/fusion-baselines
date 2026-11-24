from typing import Union, Dict, AnyStr

from metadrive.envs.top_down_env import TopDownMetaDrive
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.utils import Config
from metadrive.constants import DEFAULT_AGENT, TerminationState
from metadrive.obs.observation_base import ObservationBase


import numpy as np
import gym

from fusion.costs import (
    binary_safety_cost,
    safety_cost_from_record,
    terminate_after_safety_event,
)
from fusion.rewards import (
    capped_speed_reward_from_velocity,
    DRIVING_REWARD_WEIGHT,
    GOAL_REWARD,
    METADRIVE_BASE_SPEED_REWARD_WEIGHT,
)
from .expert_controller import Expert

STATE_DIM = 35
LIDAR_DIM = 240
BEV_CHANNELS = 5
BEV_RESOLUTION = 84


def _state_lidar_from_combined_observation(combined):
    combined = np.asarray(combined, dtype=np.float32).reshape(-1)
    minimum_size = STATE_DIM + LIDAR_DIM
    if combined.size < minimum_size:
        raise ValueError(
            "Expected at least {} state/lidar values, got {}".format(
                minimum_size, combined.size
            )
        )
    # The first 35 values are MetaDrive's compact state. The policy uses the
    # 240 ray distances at the end and intentionally excludes every BEV pixel.
    return combined[:STATE_DIM], combined[-LIDAR_DIM:]


class StateLidarObservation(ObservationBase):
    """Dictionary observation containing only policy state and lidar."""

    def __init__(self, vehicle_config):
        super(StateLidarObservation, self).__init__(vehicle_config)
        self._lidar_observation = LidarStateObservation(vehicle_config)

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            "state": gym.spaces.Box(
                -0.0, 1.0, shape=(STATE_DIM,), dtype=np.float32
            ),
            "lidar": gym.spaces.Box(
                -0.0, 1.0, shape=(LIDAR_DIM,), dtype=np.float32
            ),
            "expert": gym.spaces.Box(
                -1.0, 1.0, shape=(2,), dtype=np.float32
            ),
        })

    def observe(self, vehicle):
        state, lidar = _state_lidar_from_combined_observation(
            self._lidar_observation.observe(vehicle)
        )
        return {
            "state": state,
            "lidar": lidar,
            "expert": np.zeros(2, dtype=np.float32),
        }

    def reset(self, env, vehicle=None):
        return self._lidar_observation.reset(env, vehicle)


class StateLidarBevObservation(ObservationBase):
    """One aligned observation object for state, lidar, and semantic BEV."""

    def __init__(
        self,
        vehicle_config,
        resolution=BEV_RESOLUTION,
        frame_stack=3,
        post_stack=5,
        frame_skip=5,
        max_distance=20.0,
    ):
        super(StateLidarBevObservation, self).__init__(vehicle_config)
        self._state_lidar_observation = StateLidarObservation(vehicle_config)
        self._bev_observation = TopDownMultiChannel(
            vehicle_config,
            False,
            True,
            frame_stack=int(frame_stack),
            post_stack=int(post_stack),
            frame_skip=int(frame_skip),
            resolution=(int(resolution), int(resolution)),
            max_distance=float(max_distance),
        )
        self.bev_shape = (
            int(self._bev_observation.num_stacks),
            int(resolution),
            int(resolution),
        )

    @property
    def observation_space(self):
        spaces = dict(self._state_lidar_observation.observation_space.spaces)
        spaces["bev"] = gym.spaces.Box(
            -0.0, 1.0, shape=self.bev_shape, dtype=np.float32
        )
        return gym.spaces.Dict(spaces)

    def observe(self, vehicle):
        observation = self._state_lidar_observation.observe(vehicle)
        bev_hwc = self._bev_observation.observe(vehicle)
        bev_chw = np.ascontiguousarray(
            np.asarray(bev_hwc, dtype=np.float32).transpose(2, 0, 1)
        )
        if bev_chw.shape != self.bev_shape:
            raise ValueError(
                "Expected online BEV shape {}, got {}".format(
                    self.bev_shape, bev_chw.shape
                )
            )
        observation["bev"] = bev_chw
        return observation

    def reset(self, env, vehicle=None):
        self._state_lidar_observation.reset(env, vehicle)
        self._bev_observation.reset(env, vehicle)


class State_TopDownMetaDriveEnv(TopDownMetaDrive):
    @classmethod
    def default_config(cls) -> Config:
        config = TopDownMetaDrive.default_config()
        config.update({
            # Keep Panda3D's offscreen buffer available for optional video
            # capture. FUSION's semantic BEV is generated separately.
            "offscreen_render": True,
            "accident_prob": 0.0,
            "safe_rl_env": True,
            "crash_vehicle_cost": 1,
            "crash_vehicle_penalty": 0,
            "crash_object_cost": 1,
            "crash_object_penalty": 0,
            "out_of_road_cost": 1.,  # only give penalty for out_of_road
            "out_of_road_penalty": 0,
            # MetaDrive's built-in speed term is linear. Keep it disabled and
            # add the canonical capped term in ``reward_function`` so online
            # evaluation exactly matches the in-memory offline relabeling.
            "driving_reward": DRIVING_REWARD_WEIGHT,
            "speed_reward": METADRIVE_BASE_SPEED_REWARD_WEIGHT,
            "success_reward": GOAL_REWARD,
            "use_lateral_reward": False,
            "use_lateral": False,
            "distance": 20, # same with offline data
            "obs_noise_scale": 0.0,
            "idm_target_speed": 30,
            "idm_acc_factor": 1.0,
            "idm_deacc_factor": -5.0,
            # Multimodal policy settings. They exactly match the derived BEV
            # replay manifest and remain dormant for vector-only checkpoints.
            "fusion_bev_observation": False,
            "fusion_bev_resolution": BEV_RESOLUTION,
            "fusion_bev_frame_stack": 3,
            "fusion_bev_post_stack": 5,
            "fusion_bev_frame_skip": 5,
            "fusion_bev_max_distance": 20.0,
        })
        return config

    def get_single_observation(self, _=None):
        if self.config["fusion_bev_observation"]:
            return StateLidarBevObservation(
                self.config["vehicle_config"],
                resolution=self.config["fusion_bev_resolution"],
                frame_stack=self.config["fusion_bev_frame_stack"],
                post_stack=self.config["fusion_bev_post_stack"],
                frame_skip=self.config["fusion_bev_frame_skip"],
                max_distance=self.config["fusion_bev_max_distance"],
            )
        return StateLidarObservation(self.config["vehicle_config"])

    def __init__(self, config):
        super(State_TopDownMetaDriveEnv, self).__init__(config)
        self.episode_cost = 0
        self.count_out_road = 0

    def cost_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()
        collision = bool(
            vehicle.crash_vehicle
            or vehicle.crash_object
            or getattr(vehicle, "crash_building", False)
        )
        step_info["cost_sparse"] = binary_safety_cost(
            collision=collision,
            out_of_road=self._is_out_of_road(vehicle),
        )
        # BaseEnv may query ``cost_function`` more than once around reset or
        # diagnostics, so cumulative episode cost is maintained in ``step``
        # after the simulator has produced exactly one transition.
        step_info["cost"] = step_info["cost_sparse"]
        return step_info['cost_sparse'], step_info

    def reward_function(self, vehicle_id: str):
        """Apply the canonical reward and let severe safety override success."""
        reward, step_info = super(
            State_TopDownMetaDriveEnv, self
        ).reward_function(vehicle_id)
        vehicle = self.vehicles[vehicle_id]
        collision = bool(
            vehicle.crash_vehicle
            or vehicle.crash_object
            or getattr(vehicle, "crash_building", False)
        )
        safety_event = bool(
            binary_safety_cost(collision, self._is_out_of_road(vehicle))
        )
        speed_reward = 0.0
        if safety_event:
            reward = 0.0
        elif not vehicle.arrive_destination:
            speed_reward = capped_speed_reward_from_velocity(vehicle.speed)
            reward += speed_reward
        step_info["capped_speed_reward"] = float(speed_reward)
        step_info["task_reward"] = float(reward)
        return reward, step_info

    def done_function(self, vehicle_id: str):
        done, done_info = super(State_TopDownMetaDriveEnv, self).done_function(vehicle_id)
        vehicle = self.vehicles[vehicle_id]
        try:
            combined_observation = self.lidar_state_obs.observe(vehicle)
        except AttributeError:
            self.lidar_state_obs = LidarStateObservation(self.config["vehicle_config"])
            combined_observation = self.lidar_state_obs.observe(vehicle)

        state, lidar = _state_lidar_from_combined_observation(
            combined_observation
        )

        current_lane = vehicle.navigation._get_current_lane(vehicle)[1]
        done_info.update({"true_state": state,
                          "lidar_state": lidar,
                          'current_lane': current_lane})

        # A severe safety event is absorbing for this research environment.
        # MetaDrive may expose several overlapping crash flags; their logical
        # OR still produces exactly one terminal event and one unit of cost.
        done = terminate_after_safety_event(done, done_info)

        # overwrite any potential change when it needs to be truncated
        if done_info[TerminationState.MAX_STEP]:
            done = True

        return done, done_info

    def step(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]):
        if self.expert is None:
            self.expert = Expert(self.vehicles[DEFAULT_AGENT], speed=100)
            self.expert_action = self.expert.get_action(0)

        o, r, d, i = super(State_TopDownMetaDriveEnv, self).step(actions)
        # Proximity remains diagnostic-only. Speed is intentionally excluded
        # from the safety cost.
        lidar_dist = np.min(i['lidar_state'])
        i['proximity_cost'] = 1e-1*max(0, 0.4-lidar_dist)**2
        # Retain overspeed telemetry for analysis, but never add it to safety
        # cost or CTG. This is the upstream diagnostic threshold, not a
        # constraint in the current experiment.
        i["velocity_cost"] = 1e-2 * max(0.0, float(i["velocity"]) - 15.0)
        i['cost_sparse'] = safety_cost_from_record(i)
        i['cost'] = i['cost_sparse']
        self.episode_cost += i['cost']
        i['total_cost'] = self.episode_cost
        self.last_dist = lidar_dist

        i['last_state'] = self.last_state["state"]
        i['last_lidar'] = self.last_state["lidar"]
        o.update({'state': i['true_state'], 'lidar': i['lidar_state']})
        self.last_state = o

        self.expert_action = self.expert.get_action(o['state'])

        o.update({'expert': np.clip(self.expert_action, -np.ones(2,), np.ones(2))})


        return o, r, d, i


    def reset(self, *args, **kwargs):
        self.episode_cost = 0
        self.count_out_road = 0
        self.count_crash = 0
        self.expert = None
        self.last_dist = 1.
        self.expert_action = np.zeros((2,), dtype=np.float32)
        o = super(State_TopDownMetaDriveEnv, self).reset(*args, **kwargs)
        self.lidar_state_obs = LidarStateObservation(self.config["vehicle_config"])
        o.update({'expert': self.expert_action})
        self.last_state = o

        return o
