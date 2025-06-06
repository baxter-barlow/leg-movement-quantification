from __future__ import annotations
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

from sklearn.cluster import KMeans
import cv2
import numpy as np
from scipy.signal import medfilt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PRE_FRAMES  = 15
POST_FRAMES = 15

def iter_frames(cap: cv2.VideoCapture) -> Iterator[np.ndarray]:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

@dataclass(frozen=True)
class VideoMeta:
    fps: float
    width: int
    height: int
    n_frames: int

def open_video(path: Path | str) -> Tuple[cv2.VideoCapture, VideoMeta]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
    meta = VideoMeta(
        fps=float(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return cap, meta

def process_video(
    input_path: str | Path,
    stim_output_dir: str | Path = "stim_videos",
    *,
    lower_cyan: Tuple[int, int, int] = (80, 60, 240),
    upper_cyan: Tuple[int, int, int] = (100, 255, 255),
    median_k: int = 15,
    stim_threshold: float = 1e-6,
) -> None:

    input_path = Path(input_path)
    cap, meta = open_video(input_path)
    logger.info(
        "Opened %s – %d frames @ %.2f fps (%dx%d)",
        input_path.name,
        meta.n_frames,
        meta.fps,
        meta.width,
        meta.height,
    )

    lower = np.array(lower_cyan, dtype=np.uint8)
    upper = np.array(upper_cyan, dtype=np.uint8)
    total_pixels = meta.width * meta.height

    cyan_pct: List[float] = []

    for frame in tqdm(iter_frames(cap), total=meta.n_frames, desc="Scanning"):
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        pct  = (mask.sum() / (255 * total_pixels)) * 100.0
        cyan_pct.append(pct)

    cap.release()

    percentages = np.array(cyan_pct, dtype=np.float32)
    smooth = medfilt(percentages, kernel_size=median_k)
    arr_reshaped = smooth.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(arr_reshaped)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    bigger_cluster_label = np.argmax(centers)
    stim_mask = labels == bigger_cluster_label

    blocks = find_stim_blocks(stim_mask)
    logger.info("Detected %d stimulation blocks", len(blocks))

    if blocks:
        save_blocks_parallel(input_path, blocks, meta, stim_output_dir)

def find_stim_blocks(mask: np.ndarray) -> List[Tuple[int, int]]:
    padded = np.pad(mask.astype(int), 1)
    diff   = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1
    return [(max(s - PRE_FRAMES, 0), e + POST_FRAMES) for s, e in zip(starts, ends)]


def _write_block(video_path: Path, start: int, end: int, meta: VideoMeta, out_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, meta.fps, (meta.width, meta.height))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for _ in range(start, end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalised = clahe.apply(gray)
        out_frame = cv2.cvtColor(equalised, cv2.COLOR_GRAY2BGR)
        writer.write(out_frame)

    writer.release()
    cap.release()


def save_blocks_parallel(
    video_path: Path | str,
    blocks: List[Tuple[int, int]],
    meta: VideoMeta,
    output_dir: Path | str,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as exe:
        futures = []
        for i, (s, e) in enumerate(blocks):
            out_path = out_dir / f"stim_block_{i:03d}.mp4"
            futures.append(exe.submit(_write_block, Path(video_path), s, e, meta, out_path))

        for f in tqdm(futures, desc="Writing clips"):
            f.result()

    logger.info("%d clips saved to %s", len(blocks), out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract light-triggered stim clips.")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--out-dir", default="stim_videos", help="Directory for stim clips")
    args = parser.parse_args()

    process_video(args.video, args.out_dir)
