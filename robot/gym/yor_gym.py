import gym
import mujoco
import mujoco.viewer
import mink
import time
import numpy as np
import threading
from typing import Optional
from pathlib import Path
import atexit

from robot.arm.ik_solver import SingleArmIK



class YOR_Complete_Mujoco:
    def __init__(self, mjcf_path: str, solver_dt: float = 0.01):
        self.mjcf_path = mjcf_path
        self.solver_dt = solver_dt

        # launch mujoco
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        # initialize arm
        self.left_q_desired: Optional[np.ndarray] = None
        self.left_q_desired_lock = threading.Lock()
        self.left_ik_solver = SingleArmIK(
            mjcf_path,
            solver_dt=self.solver_dt,
            joint_names=[
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
            ],
            ee_frame="left_arm_ee",
        )

        self.right_q_desired: Optional[np.ndarray] = None
        self.right_q_desired_lock = threading.Lock()
        self.right_ik_solver = SingleArmIK(
            mjcf_path,
            solver_dt=self.solver_dt,
            joint_names=[
                "right_arm_joint1",
                "right_arm_joint2",
                "right_arm_joint3",
                "right_arm_joint4",
                "right_arm_joint5",
                "right_arm_joint6",
            ],
            ee_frame="right_arm_ee",
        )
