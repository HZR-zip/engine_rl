"""Convert the downloaded T800 URDF into the USD consumed by the RL task."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "assets" / "t800"

parser = argparse.ArgumentParser(description="Convert the EngineAI T800 URDF to USD.")
parser.add_argument("--force", action="store_true", help="Regenerate the USD even if it already exists.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main():
    urdf_path = ASSET_DIR / "t800.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"T800 URDF not found: {urdf_path}")

    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(ASSET_DIR),
        usd_file_name="t800.usd",
        force_usd_conversion=args_cli.force,
        make_instanceable=True,
        fix_base=False,
        root_link_name="LINK_BASE",
        merge_fixed_joints=True,
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=2.0),
        ),
    )
    converter = UrdfConverter(cfg)
    print(f"Created T800 USD: {converter.usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
