from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, SineLane, StraightLane, CircularLane, AbstractLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.objects import Obstacle


class HMRIEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "collision_reward": -1,
                "right_lane_reward": 0.1,
                "high_speed_reward_highway": 0.4,
                "high_speed_reward_merge": 0.2,
                "reward_speed_range": [20, 30],
                "merging_speed_reward": -0.5,
                "lane_change_reward": -0.05,
                "lanes_count": 2,
                "high_speed_reward_roundabout": 0.1,
                "roundabout_progress_reward": 0.35, # prioritizing getting thru the roundabout over speed (>0.1)
                "high_speed_reward_intersection": 0.1,  # optional, minor reward for speed
                "intersection_progress_reward": 0.4,    # main dense reward for moving through
                # "intersection_lane_reward": 0.05   
                }
        )
        return cfg

    def _reward(self, action: int) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions

        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.

        :param action: the action performed
        :return: the reward of the state-action transition
        """
        if self.vehicle is None:
            return 0.0

        segment = self._get_segment()

        reward = self._segment_rewards(segment, action)

        return reward
    
    
    def _get_segment(self) -> str:
        """
        Identify which stitched environment segment
        the ego vehicle is currently in.
        """
        if self.vehicle is None:
            return "unknown"

        if self.vehicle.lane_index is None:
            return "unknown"

        start, end, _ = self.vehicle.lane_index

        if start.startswith("h"):
            return "highway"

        if start in ["a", "b", "c", "j", "k"]:
            return "merge"

        if start in [
            "se","ex","ee","nx",
            "ne","wx","we","sx",
            "ser","ses","sxs","sxr",
            "eer","ees","exs","exr",
            "ner","nes","nxs","nxr",
            "wer","wes","wxs","wxr"
        ]:
            return "roundabout"

        return "unknown"
    
    def _segment_rewards(self, segment: str, action: int) -> float:

        cfg = self.config

        scaled_speed = utils.lmap(
            self.vehicle.speed,
            cfg["reward_speed_range"],
            [0, 1],
        )

        right_lane = (
            self.vehicle.lane_index[2]
            / max(cfg["lanes_count"] - 1, 1)
        )

        lane_change = action in [
            self.action_type.actions_indexes["LANE_LEFT"],
            self.action_type.actions_indexes["LANE_RIGHT"],
        ]

        collision = float(self.vehicle.crashed)

        # HIGHWAY
        if segment == "highway":

            reward = (
            cfg["collision_reward"] * collision
            + cfg["right_lane_reward"] * right_lane
            + cfg["high_speed_reward_highway"] * scaled_speed
            )

            min_r = cfg["collision_reward"]
            max_r = (
                cfg["right_lane_reward"]
                + cfg["high_speed_reward_highway"]
            )

            return utils.lmap(reward, [min_r, max_r], [0, 1]) #(-1.05 to 0.5)

        # MERGE
        elif segment == "merge":

            merge_penalty = self._merge_altruism()

            reward = (
            cfg["collision_reward"] * collision
            + cfg["merging_speed_reward"] * merge_penalty
            + cfg["lane_change_reward"] * lane_change
            + cfg["right_lane_reward"] * right_lane
            + cfg["high_speed_reward_merge"] * scaled_speed
            )

            min_r = (
                cfg["collision_reward"]
                + cfg["merging_speed_reward"]
                + cfg["lane_change_reward"]
            )

            max_r = (
                cfg["right_lane_reward"]
                + cfg["high_speed_reward_merge"]
            )

            return utils.lmap(reward, [min_r, max_r], [0, 1]) #(-1.55 to 0.3)

        # ROUNDABOUT
        elif segment == "roundabout":

            progress = self._roundabout_progress()

            reward = (
            cfg["collision_reward"] * collision
            + cfg["high_speed_reward_roundabout"] * scaled_speed
            + cfg["right_lane_reward"] * right_lane
            + cfg["lane_change_reward"] * lane_change
            + cfg["roundabout_progress_reward"] * progress
            )

            min_r = (cfg["collision_reward"] + cfg["lane_change_reward"])
            max_r = (cfg["high_speed_reward_roundabout"] + cfg["right_lane_reward"] + cfg["roundabout_progress_reward"])

            return utils.lmap(reward, [min_r, max_r], [0, 1]) #(-1.05 to 0.55)

        # INTERSECTION
        elif segment == "intersection":

            progress = self._intersection_progress()

            reward = (
                cfg["collision_reward"] * collision
                + cfg["high_speed_reward_intersection"] * scaled_speed
                + cfg["intersection_progress_reward"] * progress
                # + cfg["intersection_lane_reward"] * right_lane
                + cfg["lane_change_reward"] * lane_change
            )

            min_r = cfg["collision_reward"] + cfg["lane_change_reward"]  # worst case
            max_r = (
                cfg["high_speed_reward_intersection"]
                + cfg["intersection_progress_reward"]
                # + cfg["intersection_lane_reward"]
            ) 

            return utils.lmap(reward, [min_r, max_r], [0, 1])

        else:
            return 0.0

    
    def _merge_altruism(self):

        return sum(
            (vehicle.target_speed - vehicle.speed)
            / vehicle.target_speed

            for vehicle in self.road.vehicles

            if vehicle.lane_index == ("b", "c", 2)
            and isinstance(vehicle, ControlledVehicle)
        )
    
    def _roundabout_progress(self):

        lane = self.road.network.get_lane(self.vehicle.lane_index)
        progress = lane.local_coordinates(self.vehicle.position)[0]
        return progress / lane.length

    def _intersection_progress(self):

        lane = self.road.network.get_lane(self.vehicle.lane_index)
        progress = lane.local_coordinates(self.vehicle.position)[0]
        return progress / lane.length

    # def _rewards(self, action: int) -> dict[str, float]:
    #     scaled_speed = utils.lmap(
    #         self.vehicle.speed, self.config["reward_speed_range"], [0, 1]
    #     )
    #     return {
    #         "collision_reward": self.vehicle.crashed,
    #         "right_lane_reward": self.vehicle.lane_index[2] / 1,
    #         "high_speed_reward": scaled_speed,
    #         "lane_change_reward": action in [0, 2],
    #         "merging_speed_reward": sum(  # Altruistic penalty
    #             (vehicle.target_speed - vehicle.speed) / vehicle.target_speed
    #             for vehicle in self.road.vehicles
    #             if vehicle.lane_index == ("b", "c", 2)
    #             and isinstance(vehicle, ControlledVehicle)
    #         ),
    #     }

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        if self.vehicle.crashed:
            return True

        if self.vehicle is None:
            return False

        return self.vehicle.position[0] > self.exit_position[0]

    def _is_truncated(self) -> bool:
        return False

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        #-------------------------------------------------------HIGHWAY------------------------------------------------------

        endsh = [200, 60, 30]

        n_start = 4
        n_end = 2

        y_start = [i * StraightLane.DEFAULT_WIDTH for i in range(n_start)]
        y_end = [(i+1) * StraightLane.DEFAULT_WIDTH for i in range(n_end)]

        line_types_4 = []
        line_types_2 = []

        for i in range(n_start):

            if i == 0:
               line_types_4.append([c, s])

            elif i == n_start - 1:
                line_types_4.append([n, c])

            else:
                line_types_4.append([n, s])

        for i in range(n_end):

                if i == 0:
                  line_types_2.append([c, s])

                else:
                    line_types_2.append([n, c])

        for i in range(n_start):

            net.add_lane(
                "h0",
                "h1",
                StraightLane(
                    [0, y_start[i]],
                    [endsh[0], y_start[i]],
                    line_types=line_types_4[i],
                ),
            )


        # -------------------------------------------------
    # 2) CENTER MERGE USING SINELANE
    # -------------------------------------------------

        amplitude = StraightLane.DEFAULT_WIDTH / 2
        pulsation = np.pi / endsh[1]

    # lane 0 merges upward

        net.add_lane(
            "h1",
            "h2",
            SineLane(
                [endsh[0], y_start[0]],
                [sum(endsh[:2]), y_end[0]],
                amplitude=amplitude,
                pulsation=pulsation,
                phase=0,
                line_types=[c, n],
            ),
        )

    # lane 1 continues → center-left

        net.add_lane(
            "h1",
            "h2",
            StraightLane(
                [endsh[0], y_start[1]],
                [sum(endsh[:2]), y_end[0]],
                line_types=[s, s],
            ),
        )

    # lane 2 continues → center-right

        net.add_lane(
            "h1",
            "h2",
            StraightLane(
                [endsh[0], y_start[2]],
                [sum(endsh[:2]), y_end[1]],
                line_types=[n, s],
            ),
        )

    # lane 3 merges downward

        net.add_lane(
            "h1",
            "h2",
            SineLane(
                [endsh[0], y_start[3]],
                [sum(endsh[:2]), y_end[1]],
                amplitude=-amplitude,
                pulsation=pulsation,
                phase=0,
                line_types=[n, c],
            ),
        )

    # -------------------------------------------------
    # 3) STRAIGHT 2-LANE HIGHWAY
    # -------------------------------------------------

        for i in range(n_end):

            net.add_lane(
               "h2",
                "a",  # connects to your existing merge
                StraightLane(
                    [sum(endsh[:2]), y_end[i]],
                    [sum(endsh), y_end[i]],
                    line_types=line_types_2[i],
               ),
            )


        #-------------------------------------------------------MERGE------------------------------------------------------
        # Highway lanes
        ends = [50, 80, 80, 50]  # Before, converging, merge, after
        n_lanes = self.config["lanes_count"]
        y = [(i+1)*StraightLane.DEFAULT_WIDTH for i in range(n_lanes)]
        line_type = []
        line_type_merge = []

        for i in range(n_lanes):
            if i == 0:
                line_type.append([c, s])
                line_type_merge.append([c, s])
            elif i == n_lanes - 1:
                line_type.append([n, c])
                line_type_merge.append([n, s])
            else:
                line_type.append([n, s])
                line_type_merge.append([n, s])

        for i in range(n_lanes):
            net.add_lane("a","b",StraightLane([sum(endsh), y[i]], [sum(endsh) + sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            net.add_lane("b", "c", StraightLane([sum(endsh) + sum(ends[:2]), y[i]], [sum(endsh) + sum(ends[:3]), y[i]], line_types=line_type_merge[i],),
            )
            net.add_lane("c","wer", StraightLane([sum(endsh) + sum(ends[:3]), y[i]], [sum(endsh) + sum(ends), y[i]], line_types=line_type[i]),
            )
            

        # Merging lane
        road_width = n_lanes * StraightLane.DEFAULT_WIDTH
        merge_y = road_width + 10.5

        amplitude = 3.25
        ljk = StraightLane(
                            [sum(endsh), merge_y],
                            [sum(endsh) + ends[0], merge_y],
                            line_types=[c, c],
                            forbidden=True
                            )
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)

        #-------------------------------------------------------ROUNDABOUT------------------------------------------------------

        # Access lanes: (r)oad/(s)ine
        access = 170  # [m]
        dev = 85  # [m]
        a = 5 # [m]
        delta_st = 0.2 * dev  # [m]
        delta_en = dev - delta_st
        w = 2 * np.pi / dev


        # Circular Lanes
        center = [sum(endsh) + sum(ends) + access, StraightLane.DEFAULT_WIDTH + StraightLane.DEFAULT_WIDTH/2]  # [m]
        #center = [0,0]
        radius = 20  # [m]
        alpha = 24  # [deg]

        radii = [radius, radius + 4]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]

        for lane in [0, 1]:
            net.add_lane("se", "ex",
                            CircularLane(center, radii[lane], np.deg2rad(90 - alpha), np.deg2rad(alpha), clockwise=False, line_types=line[lane],),
            )
            net.add_lane("ex", "ee",
                            CircularLane(center, radii[lane], np.deg2rad(alpha), np.deg2rad(-alpha), clockwise=False, line_types=line[lane],),
            )
            net.add_lane("ee", "nx",
                            CircularLane(center, radii[lane], np.deg2rad(-alpha), np.deg2rad(-90 + alpha), clockwise=False, line_types=line[lane],),
            )
            net.add_lane("nx", "ne",
                            CircularLane(center, radii[lane], np.deg2rad(-90 + alpha), np.deg2rad(-90 - alpha), clockwise=False, line_types=line[lane],),
            )
            net.add_lane("ne", "wx",
                            CircularLane(center, radii[lane], np.deg2rad(-90 - alpha), np.deg2rad(-180 + alpha), clockwise=False, line_types=line[lane],),
            )
            net.add_lane("wx", "we",
                            CircularLane(center, radii[lane], np.deg2rad(-180 + alpha), np.deg2rad(-180 - alpha), clockwise=False, line_types=line[lane],),
            )
            net.add_lane("we", "sx",
                            CircularLane(center, radii[lane], np.deg2rad(180 - alpha), np.deg2rad(90 + alpha), clockwise=False, line_types=line[lane],),
            )
            net.add_lane("sx", "se",
                            CircularLane(center, radii[lane], np.deg2rad(90 + alpha), np.deg2rad(90 - alpha), clockwise=False, line_types=line[lane],),
            )

        
        net.add_lane("ser", "ses", StraightLane([2 + center[0], access + center[1]], [2 + center[0], dev / 2 + center[1]], line_types=(s, c))
        )
        net.add_lane("ses", "se", SineLane([2 + center[0] + a, dev / 2 + center[1]], [2 + a + center[0], dev / 2 - delta_st + center[1]], a, w,
                -np.pi / 2,
                line_types=(c, c),),
        )
        net.add_lane("sx", "sxs", SineLane([-2 - a+ center[0], -dev / 2 + delta_en + center[1]], [-2 - a+ center[0], dev / 2 + center[1]], a, w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),),
        )
        net.add_lane("sxs", "sxr", StraightLane([-2+ center[0], dev / 2 + center[1]], [-2+ center[0], access + center[1]], line_types=(n, c))
        )


        net.add_lane("eer", "ees", StraightLane([access+ center[0], -2 + center[1]], [dev / 2 + center[0], -2 + center[1]], line_types=(s, c))
        )
        net.add_lane("ees", "ee", SineLane([dev / 2 + center[0], -2 - a + center[1]], [dev / 2 - delta_st + center[0], -2 - a + center[1]], a, w,
                -np.pi / 2,
                line_types=(c, c),),
        )
        net.add_lane("ex", "exs", SineLane([-dev / 2 + delta_en + center[0], 2 + a + center[1]], [dev / 2 + center[0], 2 + a + center[1]], a, w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),),
        )
        net.add_lane("exs", "o1", StraightLane([dev / 2 + center[0], 2 + center[1]], [access + center[0], 2 + center[1]], line_types=(n, c))
        )


        net.add_lane("ner", "nes", StraightLane([-2 + center[0], -access + center[1]], [-2 + center[0], -dev / 2 + center[1]], line_types=(s, c))
        )
        net.add_lane("nes", "ne", SineLane([-2 - a + center[0], -dev / 2 + center[1]], [-2 - a + center[0], -dev / 2  + center[1] + delta_st], a, w,
                -np.pi / 2,
                line_types=(c, c),),
        )
        net.add_lane("nx", "nxs", SineLane( [2 + a + center[0], dev / 2 - delta_en + center[1]], [2 + a + center[0], -dev / 2 + center[1]], a, w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),),
        )
        net.add_lane("nxs", "nxr", StraightLane([2 + center[0], -dev / 2 + center[1]], [2 + center[0], -access + center[1]], line_types=(n, c))
        )


        net.add_lane(
            "wer", "wes", StraightLane([-access + center[0], 2 + center[1]], [-dev / 2 + center[0], 2 + center[1]], line_types=(c, c))
        )
        net.add_lane("wes", "we", SineLane([-dev / 2 + center[0], 2 + a + center[1]], [-dev / 2 + delta_st + center[0], 2 + a + center[1]], a, w,
                -np.pi / 2,
                line_types=(c, c),),
        )
        net.add_lane("wx", "wxs", SineLane([dev / 2 - delta_en + center[0], -2 - a + center[1]], [-dev / 2 + center[0], -2 - a + center[1]], a, w,
                -np.pi / 2 + w * delta_en,
                line_types=(c, c),),
        )
        net.add_lane(
            "wxs", "wxr", StraightLane([-dev / 2 + center[0], -2 + center[1]], [-access + center[0], -2 + center[1]], line_types=(n, c))
        )

        #-------------------------------------------------------INTERSECTION------------------------------------------------------
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_int = 50 + 50  # [m]
        
        roundabout_end = access + center[0]
        int_center = roundabout_end + access_int

        pivot = np.array([int_center, center[1]])

        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            # Incoming
            start = pivot + rotation @ np.array(
                [lane_width / 2, access_int + outer_distance]
            )
            end = pivot + rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                "o" + str(corner),
                "ir" + str(corner),
                StraightLane(
                    start, end, line_types=[s, c], priority=priority, speed_limit=10.0
                ),
            )
            # Right turn
            r_center = pivot + rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner - 1) % 4),
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, c],
                    priority=priority,
                    speed_limit=10.0,
                ),
            )
            # Left turn
            l_center = pivot + rotation @ (
                np.array(
                    [
                        -left_turn_radius + lane_width / 2,
                        left_turn_radius - lane_width / 2,
                    ]
                )
            )
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 1) % 4),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[n, n],
                    priority=priority - 1,
                    speed_limit=10.0,
                ),
            )
            # Straight
            start = pivot + rotation @ np.array([lane_width / 2, outer_distance])
            end = pivot + rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 2) % 4),
                StraightLane(
                    start, end, line_types=[s, n], priority=priority, speed_limit=10.0
                ),
            )
            # Exit
            start = pivot + rotation @ np.flip(
                [lane_width / 2, access_int + outer_distance], axis=0
            )
            end = pivot + rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(
                "il" + str((corner - 1) % 4),
                "o" + str((corner - 1) % 4),
                StraightLane(
                    end, start, line_types=[n, c], priority=priority, speed_limit=10.0
                ),
            )


        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        road.objects.append(Obstacle(road, lbc.position(ends[2] + ends[3], -2*StraightLane.DEFAULT_WIDTH)))
        self.exit_lane_index = ("il0", "o0", 0)

        exit_lane = net.get_lane(self.exit_lane_index)

        self.exit_position = exit_lane.end
        self.exit_heading = exit_lane.heading_at(exit_lane.length)
        
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.

        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("h0", "h1", 3)).position(10.0, 0.0), speed=30.0
        )
        ego_vehicle.plan_route_to("o0")      
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        # for position, speed in [(90.0, 29.0), (70.0, 31.0), (5.0, 31.5)]:
        #     lane = road.network.get_lane(("a", "b", self.np_random.integers(2)))
        #     position = lane.position(position + self.np_random.uniform(-5.0, 5.0), 0.0)
        #     speed += self.np_random.uniform(-1.0, 1.0)
            #road.vehicles.append(other_vehicles_type(road, position, speed=speed))

        # #merging_v = other_vehicles_type(
        # #    road, road.network.get_lane(("j", "k", 0)).position(110.0, 0.0), speed=20.0
        # #)
        # merging_v.target_speed = 30.0
        # road.vehicles.append(merging_v)
        
        # ego_vehicle.plan_route_to("exr")

        # for _ in range(4):

        #     lane_id = self.np_random.integers(4)

        #     lane = road.network.get_lane(
        #         ("h0", "h1", lane_id)
        #     )

        #     position = lane.position(
        #         self.np_random.uniform(30, 120),
        #         0,
        #     )

        #     speed = self.np_random.uniform(
        #         26,
        #         32,
        #     )

        #     vehicle = other_vehicles_type(
        #         road,
        #         position,
        #         speed=speed,
        #     )

        #     road.vehicles.append(vehicle)
