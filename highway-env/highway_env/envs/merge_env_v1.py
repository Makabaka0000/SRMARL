"""
This environment is built on HighwayEnv with one main road and one merging lane.
Dong Chen: chendon9@msu.edu
Date: 01/05/2021
"""
import numpy as np
from gym.envs.registration import register
from typing import Tuple

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    n_a = 5
    n_s = 25

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"},
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True},
            "controlled_vehicles": 1,
            "screen_width": 600,
            "screen_height": 120,
            "centering_position": [0.3, 0.5],
            "scaling": 3,
            "simulation_frequency": 15,  # [Hz]
            "duration": 20,  # time step
            "policy_frequency": 5,  # [Hz]
            "reward_speed_range": [10, 30],
            "COLLISION_REWARD": 200,  # default=200
            "COLLISION_COST": 200,  # default=200
            "TTC_COST": 10,  
            "MERGE_SUCCESS_REWARD":0,
            "MERGE_REACH_REWARD":0,
            "HIGH_SPEED_REWARD": 1,  # default=0.5
            "HEADWAY_COST": 10,  # default=1
            "HEADWAY_TIME": 1.2,  # default=1.2[s]
            "MERGING_LANE_COST": 10,  # default=4
            "traffic_density": 1,  # easy or hard modes
        })
        return config

    def _reward(self, action: int) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
            The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
            But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
            :param action: the action performed
            :return: the reward of the state-action transition
       """
        # the optimal reward is 0
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # compute cost for staying on the merging lane
        if vehicle.lane_index == ("b", "c", 1):
            Merging_lane_cost = - np.exp(-(vehicle.position[0] - sum(self.ends[:3])) ** 2 / (
                    10 * self.ends[2]))
        else:
            Merging_lane_cost = 0


        ramp_lanes = [
            ("j", "k", 0),
            ("k", "b", 0),
            ("b", "c", 2)
        ]

        # 主路车道
        main_lanes = [
            ("a", "b", 0),
            ("b", "c", 0),
            ("c", "d", 0),
            ("a", "b", 1),
            ("b", "c", 1),
            ("c", "d", 1)
        ]
        old_flag = vehicle.road_flag
        if vehicle.lane_index in ramp_lanes:
            new_flag= 0
        elif vehicle.lane_index in main_lanes:
            new_flag = 1
        if new_flag-old_flag == 1:
            vehicle.road_flag = new_flag

        # compute headway cost
        headway_distance = self._compute_headway_distance(vehicle)
        Headway_cost = np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0
        # compute overall reward
        reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
                 + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
                 + self.config["MERGING_LANE_COST"] * Merging_lane_cost \
                 + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)
        return reward

    def _agent_cost(self, action: int, vehicle: Vehicle) -> float:
        """
            The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
            But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
            :param action: the action performed
            :return: the reward of the state-action transition
       """
        
        # compute ttc cost
        ttc_cost = self._compute_headway_ttc(vehicle)
        # compute overall cost
        cost = self.config["COLLISION_COST"] * (1 * vehicle.crashed)+ self.config["TTC_COST"] * ttc_cost
        return cost
    
    def _compute_headway_ttc(self, vehicle):
        front_distance = float('inf')
        rear_distance = float('inf')
        min_ttc=0.1
        alpha = 1.0   # 前车
        beta  = 1.5   # 后车，惩罚更重
        max_ttc_clip = 100.0  # 最大TTC值，超过该值认为安全
        front_vehicle = None
        rear_vehicle = None

        for v in self.road.vehicles:
            if v == vehicle:
                continue

            if v.lane_index == vehicle.lane_index:
                delta_pos = v.position[0] - vehicle.position[0]

                # 前车
                if delta_pos > 0 and delta_pos < front_distance:
                    front_distance = delta_pos
                    front_vehicle = v

                # 后车
                elif delta_pos < 0 and abs(delta_pos) < rear_distance:
                    rear_distance = abs(delta_pos)
                    rear_vehicle = v

        # 计算前车 TTC cost
        if front_vehicle:
            delta_v_front = vehicle.speed - front_vehicle.speed
            if delta_v_front <= 0:
                TTC_front = float('inf')
            else:
                TTC_front = front_distance / max(delta_v_front, min_ttc)
        else:
            TTC_front = float('inf')

        # 计算后车 TTC cost
        if rear_vehicle:
            delta_v_rear = rear_vehicle.speed - vehicle.speed
            if delta_v_rear <= 0:
                TTC_rear = float('inf')
            else:
                TTC_rear = rear_distance / max(delta_v_rear, min_ttc)
        else:
            TTC_rear = float('inf')

        # 映射成 dense cost
        ttc_cost_front = 1.0 / min(TTC_front, max_ttc_clip)
        ttc_cost_rear  = 1.0 / min(TTC_rear, max_ttc_clip)

        ttc_cost = alpha * ttc_cost_front + beta * ttc_cost_rear

        return ttc_cost
    
    def _regional_reward(self):
        for vehicle in self.controlled_vehicles:
            neighbor_vehicle = []

            # vehicle is on the main road
            if vehicle.lane_index == ("a", "b", 0) or vehicle.lane_index == ("b", "c", 0) or vehicle.lane_index == (
                    "c", "d", 0):
                v_fl, v_rl = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
                # assume we can observe the ramp on this road
                elif vehicle.lane_index == ("a", "b", 0) and vehicle.position[0] > self.ends[0]:
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle, ("k", "b", 0))
                else:
                    v_fr, v_rr = None, None
            else:
                # vehicle is on the ramp
                v_fr, v_rr = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
                # assume we can observe the straight road on the ramp
                elif vehicle.lane_index == ("k", "b", 0):
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle, ("a", "b", 0))
                else:
                    v_fl, v_rl = None, None
            for v in [v_fl, v_fr, vehicle, v_rl, v_rr]:
                if type(v) is MDPVehicle and v is not None:
                    neighbor_vehicle.append(v)
            regional_reward = sum(v.local_reward for v in neighbor_vehicle)
            vehicle.regional_reward = regional_reward / sum(1 for _ in filter(None.__ne__, neighbor_vehicle))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = []
        obs, reward, done, info = super().step(action)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
        info["agents_info"] = agent_info

        for vehicle in self.controlled_vehicles:
            vehicle.local_reward = self._agent_reward(action, vehicle)
            vehicle.cost = self._agent_cost(action, vehicle)
        # local reward
        info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        info["agents_costs"] = tuple(vehicle.cost for vehicle in self.controlled_vehicles)
        info["agent_reach"] = sum(vehicle.road_flag for vehicle in self.controlled_vehicles)
        # regional reward
        self._regional_reward()
        info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)
        info["agent_crashed"] = sum(vehicle.crashed for vehicle in self.controlled_vehicles)
        obs = np.asarray(obs).reshape((len(obs), -1))
        return obs, reward, done, info

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _reset(self, is_training,num_CAV=0) -> None:
        self._make_road()

        if not is_training:
            if self.config["traffic_density"] == 1:
                # easy mode: 1-3 CAVs + 1-3 HDVs
                if num_CAV == 0:
                    num_CAV = np.random.choice(np.arange(1, 4), 1)[0]
                else:
                    num_CAV = num_CAV
                num_HDV = np.random.choice(np.arange(1, 4), 1)[0]
            elif self.config["traffic_density"] == 2:
                # hard mode: 2-4 CAVs + 2-4 HDVs
                if num_CAV == 0:
                    num_CAV = np.random.choice(np.arange(2, 5), 1)[0]
                else:
                    num_CAV = num_CAV
                num_HDV = np.random.choice(np.arange(2, 5), 1)[0]

            elif self.config["traffic_density"] == 3:
                # hard mode: 4-6 CAVs + 3-5 HDVs
                if num_CAV == 0:
                    num_CAV = np.random.choice(np.arange(4, 6), 1)[0]
                else:
                    num_CAV = num_CAV
                num_HDV = np.random.choice(np.arange(3, 6), 1)[0]
        else:
            if self.config["traffic_density"] == 1:
                num_CAV = 3
                num_HDV = 2
            elif self.config["traffic_density"] == 2:
                # hard mode: 2-4 CAVs + 2-4 HDVs
                num_CAV = 4
                num_HDV = 3
            elif self.config["traffic_density"] == 3:
                # hard mode: 4-6 CAVs + 3-5 HDVs
                num_CAV = 5
                num_HDV = 4

        self._make_vehicles(num_CAV, num_HDV)
        self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])
        
        
    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()
        
        # Highway lanes
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        #print("ends",self.ends)
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(self.ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(self.ends[:2]), y[i]],
                    [sum(self.ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(self.ends[:3]), y[i]], [sum(self.ends), y[i]], line_types=line_type[i]
                ),
            )

        # Merging lane
        
        amplitude = 3.25
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [self.ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True  # se acrescentar mais um 4 vai aumentar o numero de faixas?
        )
        lkb = SineLane(
            ljk.position(self.ends[0], -amplitude),
            ljk.position(sum(self.ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * self.ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(self.ends[1], 0),
            lkb.position(self.ends[1], 0) + [self.ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(self.ends[2], 0)))
        
        self.road = road
        
    """
    # codigo de 1 faixa
    
    def _make_road(self, ) -> None:
        
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        
        net = RoadNetwork()

        # Highway lanes
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        net.add_lane("a", "b", StraightLane([0, 0], [sum(self.ends[:2]), 0], line_types=[c, c]))
        net.add_lane("b", "c",
                     StraightLane([sum(self.ends[:2]), 0], [sum(self.ends[:3]), 0], line_types=[c, s]))
        net.add_lane("c", "d", StraightLane([sum(self.ends[:3]), 0], [sum(self.ends), 0], line_types=[c, c]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4], [self.ends[0], 6.5 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(self.ends[0], -amplitude), ljk.position(sum(self.ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2 * self.ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(self.ends[1], 0), lkb.position(self.ends[1], 0) + [self.ends[2], 0],
                           line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(self.ends[2], 0)))
        self.road = road
    """
    
    # preparado para duas faixas

    def _make_vehicles(self, num_CAV=4, num_HDV=3) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        spawn_points_s = [10, 50, 90, 130, 170, 210]
        spawn_points_m = [5, 45, 85, 125, 165, 205]

        """Spawn points for CAV"""
        # spawn point indexes on the straight road
        spawn_point_s_c = np.random.choice(spawn_points_s, num_CAV // 2, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_c = np.random.choice(spawn_points_m, num_CAV - num_CAV // 2,
                                        replace=False)
        spawn_point_s_c = list(spawn_point_s_c)
        spawn_point_m_c = list(spawn_point_m_c)
        # remove the points to avoid duplicate
        for a in spawn_point_s_c:
            spawn_points_s.remove(a)
        for b in spawn_point_m_c:
            spawn_points_m.remove(b)

        """Spawn points for HDV"""
        # spawn point indexes on the straight road
        spawn_point_s_h = np.random.choice(spawn_points_s, num_HDV // 2, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_h = np.random.choice(spawn_points_m, num_HDV - num_HDV // 2,
                                        replace=False)
        spawn_point_s_h = list(spawn_point_s_h)
        spawn_point_m_h = list(spawn_point_m_h)

        # initial speed with noise and location noise
        initial_speed = np.random.rand(num_CAV + num_HDV) * 2 + 25  # range from [25, 27]
        loc_noise = np.random.rand(num_CAV + num_HDV) * 3 - 1.5  # range from [-1.5, 1.5]
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)

        """spawn the CAV on the straight road first"""
        for _ in range(num_CAV // 2):
            
            ego_vehicle = self.action_type.vehicle_class(
                road, 
                road.network.get_lane(("a", "b", np.random.randint(2))).position(
                    spawn_point_s_c.pop(0) + loc_noise.pop(0), 0),
                speed=initial_speed.pop(0)
            )
            ego_vehicle.road_flag = 1  # main road
            ego_vehicle.reach = 0
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        """spawn the rest CAV on the merging road"""
        for _ in range(num_CAV - num_CAV // 2):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("j", "k", 0)).position(
                spawn_point_m_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            ego_vehicle.road_flag = 1  # main road
            ego_vehicle.reach = 0
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        """spawn the HDV on the main road first"""
        for _ in range(num_HDV // 2):
            
            road.vehicles.append(
                other_vehicles_type(
                    road, 
                    road.network.get_lane(("a", "b", np.random.randint(2))).position(
                        spawn_point_s_h.pop(0) + loc_noise.pop(0), 0),
                    speed=initial_speed.pop(0)
                )
            )

        """spawn the rest HDV on the merging road"""
        for _ in range(num_HDV - num_HDV // 2):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(
                    spawn_point_m_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0)))


        """
        
        def _make_vehicles(self, num_CAV=4, num_HDV=3) -> None:
        
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        spawn_points_s = [10, 50, 90, 130, 170, 210]
        spawn_points_m = [5, 45, 85, 125, 165, 205]

       
        # spawn point indexes on the straight road
        spawn_point_s_c = np.random.choice(spawn_points_s, num_CAV // 2, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_c = np.random.choice(spawn_points_m, num_CAV - num_CAV // 2,
                                           replace=False)
        spawn_point_s_c = list(spawn_point_s_c)
        spawn_point_m_c = list(spawn_point_m_c)
        # remove the points to avoid duplicate
        for a in spawn_point_s_c:
            spawn_points_s.remove(a)
        for b in spawn_point_m_c:
            spawn_points_m.remove(b)

        # spawn point indexes on the straight road
        spawn_point_s_h = np.random.choice(spawn_points_s, num_HDV // 2, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_h = np.random.choice(spawn_points_m, num_HDV - num_HDV // 2,
                                           replace=False)
        spawn_point_s_h = list(spawn_point_s_h)
        spawn_point_m_h = list(spawn_point_m_h)

        # initial speed with noise and location noise
        initial_speed = np.random.rand(num_CAV + num_HDV) * 2 + 25  # range from [25, 27]
        loc_noise = np.random.rand(num_CAV + num_HDV) * 3 - 1.5  # range from [-1.5, 1.5]
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)

        
        for _ in range(num_CAV // 2):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 0)).position(
                spawn_point_s_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)
        
        for _ in range(num_CAV - num_CAV // 2):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("j", "k", 0)).position(
                spawn_point_m_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        
        for _ in range(num_HDV // 2):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(
                    spawn_point_s_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0)))

        
        for _ in range(num_HDV - num_HDV // 2):
            road.vehicles.append(
                other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(
                    spawn_point_m_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0)))
        
        
        """
    
    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds


class MergeEnvMARL(MergeEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }},
            "controlled_vehicles": 4
        })
        return config
    


register(
    id='merge-v1',
    entry_point='highway_env.envs:MergeEnv',
)

register(
    id='merge-multi-agent-v0',
    entry_point='highway_env.envs:MergeEnvMARL',
)