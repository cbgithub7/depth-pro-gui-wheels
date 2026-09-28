# depth-pro-gui-wheels

Prebuilt Windows wheels of **texture_baker** and **uv_unwrapper**, the two compiled extensions that
Stability AI's [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) and
[SPAR3D](https://github.com/Stability-AI/stable-point-aware-3d) need. Upstream publishes no wheels,
so both models have needed Visual Studio Build Tools, and the CUDA Toolkit, to install. With these
wheels, depth-pro-gui installs them with no compiler.

Powered by Stability AI.

## What is built

`builds.json` lists every build. It is the output of depth-pro-gui's
`python python/objects/cli.py wheel-builds`, so these wheels match the pins the app installs:

- One build per backend and torch pin: SF3D and SPAR3D, each with torch 2.5.1 cu124, torch 2.7.1
  cu128 and torch 2.5.1 cpu, on Python 3.10.
- An extension links against libtorch, so a wheel works only with the torch it was built against.
- SF3D and SPAR3D get separate wheels even though their C++ is identical. A wheel carries its
  commit's Python files too, and SPAR3D's `uv_unwrapper/unwrap.py` differs from SF3D's.

The source is upstream's own, fetched at the pinned commit, and upstream's `setup.py` runs
unchanged. Only the environment is set:

- `USE_NATIVE_ARCH=0`: nothing is tuned to the build machine's processor.
- `TORCH_CUDA_ARCH_LIST`: every compute capability the torch build supports, the newest with PTX.
  For cu124 that is 5.0 to 9.0; for cu128 it is 7.5 to 12.0.
- `NVCC_APPEND_FLAGS=-allow-unsupported-compiler`: CUDA 12.4's nvcc refuses the MSVC the runners
  carry.
- No OpenMP, the same as upstream's own Windows builds: MSVC ignores the GCC `-fopenmp` flag its
  `setup.py` passes. So the processor path runs on one thread.

Every wheel is installed and run before it is kept. `smoke.py` unwraps a sphere, bakes its
positions, and checks that they land on the sphere. For a CUDA build it also checks that the CUDA
kernels were compiled in. The runners have no GPU, so the kernels themselves are run on a real card
separately.

## Releases

Run the `build` workflow by hand; with a release tag, it publishes every wheel under one release:

- Each wheel is named `<backend>-torch<version>-<build>-<wheel file>`.
- `manifest.json` gives each wheel's file name, sha256, upstream commit and torch pin. depth-pro-gui
  copies these into its contract, and installs a wheel only if its bytes match that sha256.

## Licence

The wheels are compiled from Stability AI's code and fall under the
[Stability AI Community License](https://stability.ai/community-license-agreement). Each wheel
carries the licence and the `NOTICE` below in its `.dist-info`, and every release attaches both.

> This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved

