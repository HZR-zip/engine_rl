"""Isaac Lab port of the CMoE G1 12-DoF rough-terrain task.

Uses the pinned G1 asset and explicit PD controls. Terrain/reward parity and
runtime simulation validation remain separate reproduction gates.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.actuators import IdealPDActuatorCfg
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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from t800_rl.cmoe.contract import ACTION_SCALE, JOINT_NAMES, USD_PATH, joint_parameters
from .observations import CMoEObservation, height_pattern

LEG_JOINTS = list(JOINT_NAMES)
JOINT_PARAMETERS = joint_parameters()

G1_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.2), step_width=0.3, platform_width=2.0
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.2), step_width=0.3, platform_width=2.0
        ),
        "gaps": terrain_gen.MeshGapTerrainCfg(proportion=0.2, gap_width_range=(0.1, 0.4), platform_width=2.0),
        "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.02, 0.1), platform_width=2.0
        ),
    },
)


def _g1_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={name: item["default"] for name, item in JOINT_PARAMETERS.items()},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=LEG_JOINTS,
                effort_limit={name: item["effort"] for name, item in JOINT_PARAMETERS.items()},
                effort_limit_sim={name: item["effort"] for name, item in JOINT_PARAMETERS.items()},
                velocity_limit_sim={name: item["velocity"] for name, item in JOINT_PARAMETERS.items()},
                stiffness={name: item["stiffness"] for name, item in JOINT_PARAMETERS.items()},
                damping={name: item["damping"] for name, item in JOINT_PARAMETERS.items()},
            )
        },
    )


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=G1_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.8,
            dynamic_friction=0.6,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = _g1_cfg()
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(func=height_pattern, resolution=0.1, size=(1.0, 0.6), ordering="yx"),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class G1CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.0, 1.0)
        ),
    )


@configclass
class G1ActionsCfg:
    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=LEG_JOINTS, preserve_order=True, scale=ACTION_SCALE, use_default_offset=True
    )


@configclass
class G1ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        cmoe = ObsTerm(func=CMoEObservation, params={"group": "policy"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        cmoe = ObsTerm(func=CMoEObservation, params={"group": "critic"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1RewardsCfg:
    track_linear_velocity = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_yaw_rate = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    vertical_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    angular_velocity_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.3)
    joint_acceleration = RewTerm(
        func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS)}
    )


@configclass
class G1EventsCfg:
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
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS),
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class G1TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="pelvis"), "threshold": 1.0},
    )


@configclass
class G1CMoEEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1SceneCfg = G1SceneCfg(num_envs=2048, env_spacing=3.0)
    observations: G1ObservationsCfg = G1ObservationsCfg()
    actions: G1ActionsCfg = G1ActionsCfg()
    commands: G1CommandsCfg = G1CommandsCfg()
    rewards: G1RewardsCfg = G1RewardsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()
    events: G1EventsCfg = G1EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
