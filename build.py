"""Build one backend's compiled extensions as wheels for one torch pin, test them, and describe them.

    python build.py <backend> <torch> <build> [--out dist] [--work work] [--skip-torch]

builds.json says what exists to build; it is depth-pro-gui's `python python/objects/cli.py
wheel-builds`, so the pins here are the pins the app installs. For the entry named:

1. Upstream's source at the pinned commit, from codeload (nothing else of upstream's is used).
2. torch at the pinned version from its own index, plus what a --no-build-isolation build needs.
3. Each package built with `pip wheel`, upstream's setup.py unchanged. The environment decides the
   rest: USE_NATIVE_ARCH=0, so nothing is tuned to the build machine's processor; USE_CUDA and
   TORCH_CUDA_ARCH_LIST from the build's compute-capability range, since the runner has no card to
   ask; NVCC_APPEND_FLAGS=-allow-unsupported-compiler, because CUDA 12.4's nvcc refuses the MSVC the
   runners carry. The Stability AI Community License and its notice go into every wheel.
4. The wheels installed and run on the processor (smoke.py), and for a CUDA build the check that
   its kernels were actually compiled in.
5. dist/<asset> for each wheel -- named <backend>-torch<torch>-<build>-<wheel file>, so one release
   holds every build -- and dist/manifest-<backend>-<torch>-<build>.json describing them.

Only the Python standard library; run with the Python the wheels are for.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_URL = "https://codeload.github.com/{repo}/tar.gz/{commit}"
# Compute capabilities NVIDIA has shipped, as the numbers nvcc takes. TORCH_CUDA_ARCH_LIST is this
# list cut to the build's range; the newest gets +PTX so a later card can still JIT the kernels.
KNOWN_CAPABILITIES = ["5.0", "5.2", "6.0", "6.1", "7.0", "7.5", "8.0", "8.6", "8.9", "9.0", "10.0", "12.0"]
# What upstream's setup.py leaves open; setuptools is SF3D's README pin for exactly this build.
BUILD_REQUIREMENTS = ["setuptools==69.5.1", "wheel", "ninja"]
NOTICE = ("This Stability AI Model is licensed under the Stability AI Community License, "
          "Copyright © Stability AI Ltd. All Rights Reserved\n")


def capability_key(value: str) -> tuple:
    major, minor = value.split(".")
    return int(major), int(minor)


def arch_list(low: str, high: str) -> str:
    inside = [c for c in KNOWN_CAPABILITIES if capability_key(low) <= capability_key(c) <= capability_key(high)]
    if not inside:
        raise SystemExit(f"no known compute capability between {low} and {high}")
    return ";".join(inside[:-1] + [inside[-1] + "+PTX"])


def entry(backend: str, torch: str, build: str) -> dict:
    builds = json.loads((HERE / "builds.json").read_text(encoding="utf-8"))
    for row in builds:
        if (row["backend"], row["torch"], row["build"]) == (backend, torch, build):
            return row
    raise SystemExit(f"builds.json has no {backend} torch {torch} {build}")


def fetch_source(repo: str, commit: str, work: Path) -> Path:
    into = work / "src"
    if into.exists():
        return into
    work.mkdir(parents=True, exist_ok=True)
    archive = work / "source.tar.gz"
    with urllib.request.urlopen(SOURCE_URL.format(repo=repo, commit=commit)) as response, open(archive, "wb") as out:
        shutil.copyfileobj(response, out)
    root = (work / "x").resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not str((root / member.name).resolve()).startswith(str(root)):
                raise SystemExit(f"refusing an archive entry outside its folder: {member.name}")
        tar.extractall(root)
    tops = [p for p in root.iterdir() if p.is_dir()]
    if len(tops) != 1:
        raise SystemExit(f"expected one folder in the source archive, found {len(tops)}")
    tops[0].rename(into)
    return into


def run(argv: list, **kwargs) -> None:
    print("+", " ".join(str(a) for a in argv), flush=True)
    subprocess.run([str(a) for a in argv], check=True, **kwargs)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("backend")
    parser.add_argument("torch")
    parser.add_argument("build")
    parser.add_argument("--out", type=Path, default=HERE / "dist")
    parser.add_argument("--work", type=Path, default=HERE / "work")
    parser.add_argument("--skip-torch", action="store_true", help="torch is already installed at the pin")
    args = parser.parse_args()
    row = entry(args.backend, args.torch, args.build)
    tag = f"{row['backend']}-torch{row['torch']}-{row['build']}"
    work = args.work / tag
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    python_now = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_now != row["python"]:
        raise SystemExit(f"{tag} is for Python {row['python']}; this is {python_now}")
    src = fetch_source(row["repo"], row["commit"], work)
    licence = src / "LICENSE.md"
    if not licence.is_file():
        raise SystemExit("upstream's LICENSE.md is missing; its wheels cannot be shared without it")

    pip = [sys.executable, "-m", "pip"]
    if not args.skip_torch:
        run([*pip, "install", f"torch=={row['torch']}", "--index-url", row["index"]])
    run([*pip, "install", *BUILD_REQUIREMENTS])

    env = dict(os.environ, USE_NATIVE_ARCH="0", USE_CUDA="0" if row["build"] == "cpu" else "1")
    cuda_arch = None
    if row["build"] != "cpu":
        cuda_arch = arch_list(row["min_capability"], row["max_capability"])
        env.update(TORCH_CUDA_ARCH_LIST=cuda_arch, DISTUTILS_USE_SDK="1",
                   NVCC_APPEND_FLAGS=(env.get("NVCC_APPEND_FLAGS", "") + " -allow-unsupported-compiler").strip())
        if not (env.get("CUDA_PATH") or env.get("CUDA_HOME")):
            raise SystemExit(f"{tag} builds CUDA kernels, and no CUDA toolkit is set up (CUDA_PATH)")

    built = []
    staging = work / "wheels"
    shutil.rmtree(staging, ignore_errors=True)
    for package in row["packages"]:
        project = src / package["dir"]
        # setuptools puts LICEN[CS]E* and NOTICE* from the project folder into the wheel's
        # .dist-info, which is what the licence's "Notice text file distributed as part of such
        # copies" asks for.
        shutil.copyfile(licence, project / "LICENSE.md")
        (project / "NOTICE").write_text(NOTICE, encoding="utf-8")
        run([*pip, "wheel", "--no-build-isolation", "--no-deps", "-w", staging / package["package"], project], env=env)
        wheels = list((staging / package["package"]).glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"expected one wheel for {package['package']}, found {len(wheels)}")
        built.append((package["package"], wheels[0]))

    run([*pip, "install", "--no-deps", "--force-reinstall", *[w for _, w in built]])
    # Away from the source tree, so the imports are the installed wheels and nothing else.
    probe = work / "probe"
    probe.mkdir(exist_ok=True)
    run([sys.executable, HERE / "smoke.py", "--expect-cuda" if row["build"] != "cpu" else "--expect-cpu"], cwd=probe)

    entries = []
    for package, wheel in built:
        asset = f"{tag}-{wheel.name}"
        shutil.copyfile(wheel, out / asset)
        entries.append({"backend": row["backend"], "package": package, "python": row["python"],
                        "torch": row["torch"], "build": row["build"], "platform": row["platform"],
                        "commit": row["commit"], "file": wheel.name, "asset": asset,
                        "sha256": sha256_of(wheel), "bytes": wheel.stat().st_size,
                        "cuda_arch_list": cuda_arch})
    (out / f"manifest-{tag}.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    shutil.copyfile(licence, out / f"{row['backend']}-LICENSE.md")
    for e in entries:
        print(f"{e['asset']}  {e['bytes']} bytes  sha256 {e['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
