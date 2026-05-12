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
from zipfile import ZIP_DEFLATED, ZipFile

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


def _zip_directory(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()

    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(src_dir)))


def create_release_distribution(
    script_dir: Path,
    artifacts_dir: Path,
    exe_path: Path,
    model_path: Path,
) -> tuple[Path, Path]:
    releases_dir = artifacts_dir / "releases"
    dist_folder = releases_dir / "TidePredictionTool_Distribution"
    zip_path = releases_dir / "TidePredictionTool_Distribution.zip"

    dist_folder.mkdir(parents=True, exist_ok=True)

    # Always sync latest executable from canonical build output.
    shutil.copy2(exe_path, dist_folder / "TidePredictionTool.exe")

    # Include required runtime model file in distribution package.
    shutil.copy2(model_path, dist_folder / model_path.name)

    # Include docs.
    for doc_name in ["README.md", "QUICK_START_INSTRUCTIONS.txt"]:
        doc_path = script_dir / doc_name
        if doc_path.exists():
            shutil.copy2(doc_path, dist_folder / doc_name)

    # Include input/output folders.
    input_src = script_dir / "input"
    input_dst = dist_folder / "input"
    if input_src.exists():
        if input_dst.exists():
            shutil.rmtree(input_dst, ignore_errors=True)
        shutil.copytree(input_src, input_dst)
    else:
        input_dst.mkdir(parents=True, exist_ok=True)

    output_dst = dist_folder / "prediction_outputs"
    output_dst.mkdir(parents=True, exist_ok=True)

    _zip_directory(dist_folder, zip_path)
    return dist_folder, zip_path


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
    exe_path = out_file_onefile if out_file_onefile.exists() else out_file_onedir

    release_folder, release_zip = create_release_distribution(
        script_dir=script_dir,
        artifacts_dir=artifacts_dir,
        exe_path=exe_path,
        model_path=model_path,
    )
    print("\n" + "=" * 60)
    print("Build complete")
    print("=" * 60)
    print(f"\nExecutable target: {exe_path}")
    print(f"Build artifacts directory: {artifacts_dir}")
    print(f"Release folder: {release_folder}")
    print(f"Release zip: {release_zip}")
    print("Distribution zip includes CSIRO_tidal_const_v12.nc and is ready for GitHub Releases.")


if __name__ == "__main__":
    # Allow running this script from any working directory.
    os.chdir(Path(__file__).resolve().parent)
    main()
