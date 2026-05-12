"""Build a standalone Windows executable for the CSIRO Tide Prediction app.

Usage:
  python build_exe.py
  python build_exe.py --onedir
  python build_exe.py --name TidePredictionTool
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import PyInstaller.__main__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TidePredictionTool executable using PyInstaller")
    parser.add_argument("--name", default="TidePredictionTool", help="Output executable name")
    parser.add_argument("--onefile", action="store_true", help="Build as one-file executable")
    parser.add_argument("--onedir", action="store_true", help="Build as one-folder distribution")
    parser.add_argument("--clean", action="store_true", help="Remove old build artifacts first")
    return parser.parse_args()


def clean_old_artifacts(script_dir: Path, exe_name: str) -> None:
    artifacts_dir = script_dir / "artifacts"
    pyi_dir = artifacts_dir / "pyinstaller"
    legacy_build_dir = script_dir / "build"
    legacy_dist_dir = script_dir / "dist"
    legacy_spec_file = script_dir / f"{exe_name}.spec"

    print("Cleaning up old build files...")
    if pyi_dir.exists():
        print(f"  Removing: {pyi_dir}")
        shutil.rmtree(pyi_dir, ignore_errors=True)

    # Remove legacy locations from older script versions.
    if legacy_build_dir.exists():
        print(f"  Removing legacy: {legacy_build_dir}")
        shutil.rmtree(legacy_build_dir, ignore_errors=True)

    if legacy_dist_dir.exists():
        print(f"  Removing legacy: {legacy_dist_dir}")
        shutil.rmtree(legacy_dist_dir, ignore_errors=True)

    if legacy_spec_file.exists():
        print(f"  Removing legacy: {legacy_spec_file}")
        legacy_spec_file.unlink()

    print("Clean complete!\n")


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    artifacts_dir = script_dir / "artifacts"
    pyi_dir = artifacts_dir / "pyinstaller"
    build_dir = pyi_dir / "build"
    dist_dir = pyi_dir / "dist"
    spec_dir = pyi_dir / "spec"
    app_path = script_dir / "TidePredictionApp.py"
    model_path = script_dir / "CSIRO_tidal_const_v12.nc"

    build_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    if not app_path.exists():
        raise FileNotFoundError(f"Cannot find app script: {app_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot find model file: {model_path}\n"
            "Download CSIRO_tidal_const_v12.nc from: https://data.csiro.au/collection/csiro:45584\n"
            "Place it in the project root before building."
        )

    if args.clean:
        clean_old_artifacts(script_dir, args.name)

    # Default behavior mirrors the previous distribution style.
    bundle_mode = "--onedir" if args.onedir else "--onefile"
    if args.onefile:
        bundle_mode = "--onefile"

    print("Starting PyInstaller build...")
    pyinstaller_args = [
        str(app_path),
        f"--name={args.name}",
        bundle_mode,
        "--windowed",
        "--noconfirm",
        "--clean",
        f"--workpath={build_dir}",
        f"--distpath={dist_dir}",
        f"--specpath={spec_dir}",
        "--hidden-import=utide",
        "--hidden-import=utide._reconstruct",
        "--hidden-import=utide.harmonics",
        "--hidden-import=xarray",
        "--hidden-import=scipy",
        "--hidden-import=scipy.spatial",
        "--hidden-import=pytz",
        "--hidden-import=shapefile",
        "--hidden-import=pyproj",
        "--hidden-import=PySide6",
        "--hidden-import=PySide6.QtCore",
        "--hidden-import=PySide6.QtGui",
        "--hidden-import=PySide6.QtWidgets",
        "--hidden-import=PySide6.QtWebChannel",
        "--hidden-import=PySide6.QtWebEngineCore",
        "--hidden-import=PySide6.QtWebEngineWidgets",
        "--hidden-import=matplotlib.backends.backend_qtagg",
        "--collect-all=utide",
        "--collect-all=xarray",
        "--collect-all=pandas",
        "--collect-all=matplotlib",
        "--collect-all=pyproj",
        "--collect-all=PySide6",
    ]

    # Keep the large NetCDF file external and in the same folder as the .exe,
    # matching the existing distribution workflow.
    PyInstaller.__main__.run(pyinstaller_args)

    out_file_onefile = dist_dir / f"{args.name}.exe"
    out_file_onedir = dist_dir / args.name / f"{args.name}.exe"
    print("\n" + "=" * 60)
    print("Build complete")
    print("=" * 60)
    if out_file_onefile.exists():
        print(f"\nExecutable target: {out_file_onefile}")
    else:
        print(f"\nExecutable target: {out_file_onedir}")
    print(f"Build artifacts directory: {artifacts_dir}")
    print("Make sure CSIRO_tidal_const_v12.nc stays beside the executable.")
    print("The input and prediction_outputs folders are created automatically at runtime.")


if __name__ == "__main__":
    # Allow running this script from any working directory.
    os.chdir(Path(__file__).resolve().parent)
    main()
