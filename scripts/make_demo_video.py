"""Create a short synthetic snooker-like video for local demos."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def main() -> Path:
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "uploads" / "demo_synthetic_snooker.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    w, h, fps, secs = 960, 540, 25, 20
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    bursts = [(4.0, 6.5), (11.0, 13.0), (16.0, 18.0)]
    base_balls = [(200, 270), (480, 300), (700, 250), (400, 200)]

    for i in range(fps * secs):
        t = i / fps
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, 35, 30)
        cv2.rectangle(frame, (80, 80), (w - 80, h - 80), (40, 140, 40), -1)
        cv2.rectangle(frame, (80, 80), (w - 80, h - 80), (20, 70, 20), 6)
        for p in [
            (90, 90),
            (w // 2, 85),
            (w - 90, 90),
            (90, h - 90),
            (w // 2, h - 85),
            (w - 90, h - 90),
        ]:
            cv2.circle(frame, p, 14, (10, 10, 10), -1)

        moving = False
        phase = 0.0
        for a, b in bursts:
            if a <= t <= b:
                moving = True
                phase = (t - a) / max(b - a, 1e-6)
                break

        for bi, (bx, by) in enumerate(base_balls):
            x, y = bx, by
            if moving:
                dx = int(80 * np.sin(phase * 6 + bi) * (1 - phase * 0.3))
                dy = int(40 * np.cos(phase * 5 + bi * 0.7))
                x, y = bx + dx, by + dy
            if bi == 0:
                color = (240, 240, 240)
            elif bi == 1:
                color = (30, 30, 200)
            elif bi == 2:
                color = (0, 200, 255)
            else:
                color = (0, 0, 0)
            cv2.circle(frame, (int(x), int(y)), 12, color, -1)
            cv2.circle(frame, (int(x), int(y)), 12, (20, 20, 20), 1)

        if not moving and int(t) % 5 == 0:
            cv2.rectangle(frame, (20, 200), (70, 400), (80, 90, 160), -1)

        writer.write(frame)

    writer.release()
    print(out.resolve())
    print(f"bytes={out.stat().st_size}")
    return out


if __name__ == "__main__":
    main()
