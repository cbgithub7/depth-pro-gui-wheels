"""Run the installed texture_baker and uv_unwrapper the way SF3D does, on a sphere.

    python smoke.py --expect-cpu | --expect-cuda [--device cuda]

Unwraps a unit sphere (uv_unwrapper), rasterizes the atlas and interpolates positions back
(texture_baker), and checks the answer: part of the texture covered, and every covered texel's
position on the sphere. On the processor by default, which is what a runner without a card can do.

--expect-cuda also asks torch's dispatcher whether texture_baker registered CUDA kernels, which is
true only if they were compiled in; no card is needed for that. --device cuda runs the bake on the
card and compares it with the processor's.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import torch


def sphere(rings: int = 24, segments: int = 48):
    verts, faces = [], []
    for i in range(rings + 1):
        theta = math.pi * i / rings
        for j in range(segments):
            phi = 2 * math.pi * j / segments
            verts.append((math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi)))
    for i in range(rings):
        for j in range(segments):
            a, b = i * segments + j, i * segments + (j + 1) % segments
            c, d = a + segments, b + segments
            if i > 0:
                faces.append((a, b, c))
            if i < rings - 1:
                faces.append((b, d, c))
    v = torch.tensor(verts, dtype=torch.float32)
    return v, v.clone(), torch.tensor(faces, dtype=torch.int64)


def bake(device: str, resolution: int = 256):
    import texture_baker
    import uv_unwrapper

    v, n, f = sphere()
    uv, indices = uv_unwrapper.Unwrapper()(v, n, f, 0.02)
    uv_flat = uv[indices].reshape(-1, 2)
    points = v[f].reshape(-1, 3)
    faces = torch.arange(points.shape[0], dtype=f.dtype).reshape(-1, 3)
    baker = texture_baker.TextureBaker()
    rast = baker.rasterize(uv_flat.to(device), faces.to(device), resolution)
    mask = baker.get_mask(rast)
    pos = baker.interpolate(points.to(device), rast, faces.to(device))
    return rast.cpu(), mask.cpu(), pos.cpu()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-cpu", action="store_true")
    parser.add_argument("--expect-cuda", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import texture_baker
    import uv_unwrapper

    report = {"torch": torch.__version__, "texture_baker": texture_baker.__file__, "uv_unwrapper": uv_unwrapper.__file__}
    has_cuda = torch._C._dispatch_has_kernel_for_dispatch_key("texture_baker_cpp::rasterize", "CUDA")
    report["cuda_kernels"] = has_cuda
    if args.expect_cuda and not has_cuda:
        raise SystemExit("texture_baker has no CUDA kernels registered: the CUDA half was not compiled in")
    if args.expect_cpu and has_cuda:
        raise SystemExit("a processor-only build registered CUDA kernels")

    rast, mask, pos = bake("cpu")
    coverage = mask.float().mean().item()
    radius = pos[mask].norm(dim=-1)
    report.update(coverage=round(coverage, 4), radius_min=round(radius.min().item(), 4),
                  radius_max=round(radius.max().item(), 4))
    if not 0.2 < coverage < 0.98:
        raise SystemExit(f"the atlas covers {coverage:.1%} of the texture; an unwrapped sphere covers a good part of it")
    # Texels interpolate flat triangles, which sit inside the unit sphere by at most the sagitta.
    if radius.min().item() < 0.97 or radius.max().item() > 1.001:
        raise SystemExit(f"baked positions leave the sphere: radius {radius.min().item():.4f}..{radius.max().item():.4f}")

    if args.device == "cuda":
        rast_gpu, mask_gpu, pos_gpu = bake("cuda")
        agree = (mask_gpu == mask).float().mean().item()
        both = mask & mask_gpu
        report.update(cuda_mask_agreement=round(agree, 5),
                      cuda_position_error=round((pos_gpu[both] - pos[both]).abs().max().item(), 6))
        if agree < 0.995:
            raise SystemExit(f"the card's bake covers different texels from the processor's ({agree:.2%} agree)")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
