"""Highlight shadows in images and videos.

usage: shadow-highlight [--threshold adaptive|fixed] <image|video>

Writes ``<stem>_shadow.<ext>`` and ``<stem>_mask.<ext>`` next to the input
(videos are always written as ``.mp4``).

As a library, ``process(frames, threshold)`` takes an iterable of BGR uint8 arrays and
yields ``(highlighted, mask)`` pairs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Literal, get_args

import cv2
import numpy as np
from numpy.typing import NDArray

U8 = NDArray[np.uint8]
F32 = NDArray[np.float32]
Threshold = Literal["adaptive", "fixed"]

# local-background estimate: max-filter footprint and blur sigma (pixels)
BG_DILATE = 61
BG_SIGMA = 25.0
# pre-smoothing of luminance and feathering of the mask edge
L_SIGMA = 1.5
MASK_FEATHER = 1.2
# fixed threshold: soft = (RATIO_CENTER - L/bg) / RATIO_RAMP, cut at 0.5 (i.e. L/bg < 0.475)
RATIO_CENTER = 0.55
RATIO_RAMP = 0.15


def background_ratio(L: F32) -> F32:
    """Luminance relative to local background (tracks vignetting)."""
    kernel = np.ones((BG_DILATE, BG_DILATE), np.uint8)
    bg = cv2.GaussianBlur(cv2.dilate(L, kernel), (0, 0), BG_SIGMA)
    return L / np.maximum(bg, 1)


def adaptive_mask(L: F32) -> U8:
    """Otsu on L/bg: picks the valley between the shadow bins and everything else."""
    r = np.clip(background_ratio(L) * 255, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ((r < t) * 255).astype(np.uint8)


def fixed_mask(L: F32) -> U8:
    """Fixed cut on L/bg, dropping blobs that touch the top edge (vignette band)."""
    r = background_ratio(L)
    soft = np.clip((RATIO_CENTER - r) / RATIO_RAMP, 0, 1)
    n, lbl, st, _ = cv2.connectedComponentsWithStats((soft > 0.05).astype(np.uint8))
    for i in range(1, n):
        if st[i, cv2.CC_STAT_TOP] == 0:
            soft[lbl == i] = 0
    return ((soft > 0.5) * 255).astype(np.uint8)


MASKS: dict[str, Callable[[F32], U8]] = {"adaptive": adaptive_mask, "fixed": fixed_mask}


def shadow(frame: U8, threshold: Threshold = "adaptive") -> tuple[U8, U8]:
    """Return ``(highlighted frame, binary shadow mask 0/255)`` for a BGR frame."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = cv2.GaussianBlur(lab[..., 0], (0, 0), L_SIGMA)
    mask = MASKS[threshold](L)
    m = cv2.GaussianBlur(mask, (0, 0), MASK_FEATHER).astype(np.float32)[..., None] / 255

    # in shadow: invert + lift L, flip and boost chroma
    inv = np.stack(
        [
            255 - lab[..., 0] * 0.6 + 40,
            128 - (lab[..., 1] - 128) * 1.5,
            128 - (lab[..., 2] - 128) * 2.0,
        ],
        axis=-1,
    )
    blended = lab * (1 - m) + inv * m
    out = cv2.cvtColor(blended.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out, mask


def process(
    frames: Iterable[U8], threshold: Threshold = "adaptive"
) -> Iterator[tuple[U8, U8]]:
    """Lazily apply :func:`shadow` to every frame."""
    for f in frames:
        yield shadow(f, threshold)


def read_video(cap: cv2.VideoCapture) -> Iterator[U8]:
    """Yield frames from an opened capture until it runs dry."""
    while True:
        ok, f = cap.read()
        if not ok:
            return
        yield f


def _sibling(p: Path, tag: str, suffix: str | None = None) -> Path:
    return p.with_name(f"{p.stem}_{tag}{suffix or p.suffix}")


def run_image(p: Path, img: U8, threshold: Threshold) -> list[Path]:
    out, mask = shadow(img, threshold)
    paths = [_sibling(p, "shadow"), _sibling(p, "mask")]
    cv2.imwrite(str(paths[0]), out)
    cv2.imwrite(str(paths[1]), mask)
    return paths


def run_video(p: Path, threshold: Threshold) -> list[Path]:
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        sys.exit(f"can't read {p}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    paths = [_sibling(p, "shadow", ".mp4"), _sibling(p, "mask", ".mp4")]
    wo = cv2.VideoWriter(str(paths[0]), fourcc, fps, size)
    wm = cv2.VideoWriter(str(paths[1]), fourcc, fps, size)
    try:
        for out, mask in process(read_video(cap), threshold):
            wo.write(out)
            wm.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    finally:
        cap.release()
        wo.release()
        wm.release()
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="shadow-highlight",
        description="Highlight shadows in an image or video.",
    )
    ap.add_argument("input", type=Path, help="image or video file")
    ap.add_argument(
        "--threshold",
        choices=get_args(Threshold),
        default="adaptive",
        help="adaptive: Otsu per frame (default); fixed: constant L/bg cut, drops top-edge blobs",
    )
    args = ap.parse_args(argv)
    p: Path = args.input
    if not p.is_file():
        sys.exit(f"no such file: {p}")

    img = cv2.imread(str(p))
    written = (
        run_image(p, img, args.threshold) if img is not None else run_video(p, args.threshold)
    )
    for w in written:
        print(w)


if __name__ == "__main__":
    main()
