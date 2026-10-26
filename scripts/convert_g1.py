"""Convert the vendored CMoE G1 fixed-upper-body URDF to an Isaac Lab USD."""

import argparse

from isaaclab.app import AppLauncher

from t800_rl.cmoe.contract import URDF_PATH, USD_PATH

OUTPUT_DIR = USD_PATH.parent

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force", action="store_true", help="Regenerate the USD if it already exists.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main():
    if not URDF_PATH.is_file():
        raise FileNotFoundError(f"Vendored G1 URDF not found: {URDF_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(URDF_PATH),
            usd_dir=str(OUTPUT_DIR),
            usd_file_name=USD_PATH.name,
            force_usd_conversion=args_cli.force,
            make_instanceable=True,
            fix_base=False,
            root_link_name="pelvis",
            merge_fixed_joints=True,
            self_collision=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="none",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
            ),
        )
    )
    print(f"Created G1 USD: {converter.usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
