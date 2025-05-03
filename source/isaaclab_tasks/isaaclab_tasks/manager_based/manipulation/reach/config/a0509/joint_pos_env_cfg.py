# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import A0509_CFG  # isort: skip


##
# Environment configuration
##

# TODO: Table is not solid, need to fix it
# TODO: Reward is probably not suitable and the parameters are not good


@configclass
class A0509ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to A0509
        self.scene.robot = A0509_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)

        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link_6"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link_6"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link_6"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "link_6"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)


@configclass
class A0509ReachEnvCfg_PLAY(A0509ReachEnvCfg):
    def __post_init__(self):
        # post init of parent./
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
