"""Explainable hard NMS for small sets of labeled xyxy detections."""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


def number(value, name):
    if type(value) not in (int, float) or abs(value) > 1e9 or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number with magnitude <= 1e9")
    return float(value)


def probability(value, name):
    value = number(value, name)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


@dataclass(frozen=True)
class Detection:
    id: str
    label: str
    score: float
    box: tuple[float, float, float, float]

    def __post_init__(self):
        for name in ("id", "label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.box, (list, tuple)) or len(self.box) != 4:
            raise ValueError("box must contain four xyxy coordinates")
        box = tuple(number(x, "coordinate") for x in self.box)
        if not (box[0] < box[2] and box[1] < box[3]):
            raise ValueError("box must have positive width and height in xyxy order")
        object.__setattr__(self, "box", box)
        object.__setattr__(self, "score", probability(self.score, "score"))


def iou(a, b):
    """Return intersection over union for two validated detections."""
    x1, y1, x2, y2 = a.box
    u1, v1, u2, v2 = b.box
    w = max(0.0, min(x2, u2) - max(x1, u1))
    h = max(0.0, min(y2, v2) - max(y1, v1))
    if w == 0 or h == 0:
        return 0.0
    # Normalize both boxes by the same per-axis factors before multiplying.
    sx, sy = max(x2 - x1, u2 - u1), max(y2 - y1, v2 - v1)
    overlap = (w / sx) * (h / sy)
    area_a = ((x2 - x1) / sx) * ((y2 - y1) / sy)
    area_b = ((u2 - u1) / sx) * ((v2 - v1) / sy)
    return overlap / (area_a + (area_b - overlap))


def suppress(detections, threshold=0.5, min_score=0.0, *, agnostic=False):
    """Return kept IDs and a decision trace; ties use input order."""
    threshold = probability(threshold, "IoU threshold")
    min_score = probability(min_score, "minimum score")
    detections = list(detections)
    if len({d.id for d in detections}) != len(detections):
        raise ValueError("detection IDs must be unique")
    kept, trace = [], []
    for d in sorted(detections, key=lambda d: -d.score):
        if d.score < min_score:
            trace.append({"id": d.id, "decision": "below-score"})
            continue
        for winner in kept:
            if agnostic or winner.label == d.label:
                overlap = iou(winner, d)
                if overlap > threshold:
                    trace.append({"id": d.id, "decision": "suppressed",
                                  "by": winner.id, "iou": overlap})
                    break
        else:
            kept.append(d)
            trace.append({"id": d.id, "decision": "kept"})
    return {"kept": [d.id for d in kept], "trace": trace}


def load(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("JSON must be a list of detection objects")
    fields = {"id", "label", "score", "box"}
    if any(not isinstance(row, dict) or row.keys() != fields for row in rows):
        raise ValueError("each row needs exactly id, label, score and box")
    return [Detection(**row) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON file of detections")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--agnostic", action="store_true", help="compare all labels")
    args = parser.parse_args()
    try:
        print(json.dumps(suppress(load(args.input), args.iou, args.min_score,
                                  agnostic=args.agnostic), indent=2))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
