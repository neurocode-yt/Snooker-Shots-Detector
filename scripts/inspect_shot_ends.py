"""Debug helper: print motion series vs detected shot ends for recent jobs."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "jobs"
    jobs = sorted(root.glob("*/analysis.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for jp in jobs[:1]:
        d = json.loads(jp.read_text(encoding="utf-8"))
        print("===", jp.parent.name, "mode", d.get("mode"), "shots", len(d.get("shots", [])))
        after_lens = []
        for s in d.get("shots", []):
            after = s["clip_end"] - s["cue_strike"]
            after_lens.append(after)
            print(
                f"  shot{s['shot_id']:02d}: strike={s['cue_strike']:7.2f} "
                f"end={s['ball_motion_end']:7.2f} "
                f"after={after:5.2f}s conf={s['end_confidence']:.2f} "
                f"{'CAP' if after >= 9.9 else 'ok '}"
            )
        if after_lens:
            import statistics as st

            print(
                f"  after-strike: median={st.median(after_lens):.2f} "
                f"mean={st.mean(after_lens):.2f} "
                f"capped={sum(1 for x in after_lens if x >= 9.9)}/{len(after_lens)}"
            )
        feats = d.get("features", [])
        if not feats or not d.get("shots"):
            continue
        # Inspect first 3 shots decay profiles
        for s in d["shots"][:3]:
            st_t = s["cue_strike"]
            print(f"\n  --- profile shot {s['shot_id']} strike={st_t:.2f} ---")
            win = [f for f in feats if st_t - 0.5 <= f["t"] <= st_t + 11]
            for f in win[::3]:
                raw = f.get("motion_raw", 0)
                print(
                    f"  t={f['t']:6.2f} raw={raw:.3f} ema={f['motion_score']:.3f} "
                    f"mean={f['residual_motion_mean']:.2f} p95={f['residual_motion_max']:.2f} "
                    f"area={f['motion_area_ratio']:.4f} cam={f.get('camera_motion_magnitude',0):.1f}"
                )


if __name__ == "__main__":
    main()
