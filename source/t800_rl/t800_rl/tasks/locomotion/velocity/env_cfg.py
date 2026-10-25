"""Starter flat-ground velocity-tracking environment for T800.

Robot-specific link names, actuator limits and rewards must be verified against
the actual T800 model before training.
"""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise

from isaaclab.actuators import ImplicitActuatorCfg

_PROJECT_ROOT = Path(__file__).resolve().parents[6]
_DEFAULT_USD = _PROJECT_ROOT / "assets" / "t800" / "t800.usd"
T800_USD_PATH = Path(os.environ.get("T800_USD_PATH", _DEFAULT_USD)).expanduser()
T800_BASE_LINK = os.environ.get("T800_BASE_LINK", "LINK_BASE")


def _t800_cfg() -> ArticulationCfg:
    if not T800_USD_PATH.is_file():
        raise FileNotFoundError(
            f"T800 USD not found: {T800_USD_PATH}. Set T800_USD_PATH to the robot USD file; "
            "then verify the joint/link names and actuator limits in env_cfg.py."
        )
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(T800_USD_PATH),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.9), joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=0.9,
        # Starter values only. Replace from T800 actuator specifications before training.
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=30.0,
                stiffness=40.0,
                damping=2.0,
            )
        },
    )


@configclass
class T800SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.6,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = _t800_cfg()
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class T800CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.6, 0.6)
        ),
    )


@configclass
class T800ActionsCfg:
    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class T800ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=UniformNoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=UniformNoise(n_min=-0.05, n_max=0.05)
        )
        velocity_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_position = ObsTerm(func=mdp.joint_pos_rel, noise=UniformNoise(n_min=-0.01, n_max=0.01))
        joint_velocity = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=UniformNoise(n_min=-1.5, n_max=1.5))
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class T800RewardsCfg:
    track_linear_velocity = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_yaw_rate = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    angular_velocity_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)


@configclass
class T800EventsCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )


@configclass
class T800TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=T800_BASE_LINK), "threshold": 1.0},
    )


@configclass
class T800FlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: T800SceneCfg = T800SceneCfg(num_envs=2048, env_spacing=3.0)
    observations: T800ObservationsCfg = T800ObservationsCfg()
    actions: T800ActionsCfg = T800ActionsCfg()
    commands: T800CommandsCfg = T800CommandsCfg()
    rewards: T800RewardsCfg = T800RewardsCfg()
    terminations: T800TerminationsCfg = T800TerminationsCfg()
    events: T800EventsCfg = T800EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
