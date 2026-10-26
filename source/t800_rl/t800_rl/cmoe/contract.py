# SPDX-License-Identifier: BSD-3-Clause
"""G1 conventions from the pinned CMoE source; independent of simulator imports."""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(os.environ.get('ENGINE_RL_ROOT', Path(__file__).resolve().parents[4]))
URDF_PATH = PROJECT_ROOT / 'references/cmoe_isaacgym/legged_gym/resources/robots/g1/29dof_urdf/g1_29dof_with_hand_fixed_modify_collision.urdf'
USD_PATH = Path(os.environ.get('G1_USD_PATH', PROJECT_ROOT / 'assets/g1/g1_cmoe.usd')).expanduser()
JOINT_NAMES = tuple(f'{side}_{joint}_joint' for side in ('left', 'right') for joint in (
    'hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll'))
FRAME_DIM, HISTORY_STEPS, HEIGHT_DIM, ACTION_DIM = 45, 10, 77, 12
POLICY_DIM, CRITIC_DIM = 527, 128
ACTION_SCALE, ACTION_CLIP, OBS_CLIP = 0.25, 100.0, 100.0
COMMAND_SCALE = (2.0, 2.0, 0.25)


def joint_parameters():
    """Read limits from the actual reference URDF, rejecting unexpected articulations."""
    root = ET.parse(URDF_PATH).getroot()
    joints = [joint for joint in root.findall('joint') if joint.get('type') != 'fixed']
    if tuple(joint.get('name') for joint in joints) != JOINT_NAMES:
        raise ValueError('Reference URDF joint order differs from the CMoE contract.')
    result = {}
    for joint in joints:
        name = joint.get('name')
        limits = {key: float(value) for key, value in joint.find('limit').attrib.items()}
        knee, ankle = 'knee' in name, 'ankle' in name
        result[name] = dict(limits, stiffness=150.0 if knee else 40.0 if ankle else 100.0,
                            damping=4.0 if knee else 2.0,
                            default=0.3 if knee else -0.2 if 'ankle_pitch' in name else
                            -0.1 if 'hip_pitch' in name else 0.0)
    return result
