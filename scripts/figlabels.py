"""Label placement for the small scatter panels.

With five protocols "above, else below" was enough. With ten, several points
sit within a label's width of each other, so each label is placed by trying
positions on rings around its point and taking the first that overlaps no
point and no label already placed.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

__all__ = ["place_labels"]

Box = Tuple[float, float, float, float]


def _overlap(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def place_labels(ax, xs: Sequence[float], ys: Sequence[float],
                 names: Sequence[str], bold: Iterable[str] = (),
                 fontsize: float = 8, color: str = "#0b0b0b",
                 leader: str = "#52514e") -> None:
    """Annotate each point. Call after the layout is final."""
    fig = ax.figure
    fig.canvas.draw()
    px = fig.dpi / 72.0                       # pixels per point
    pts = ax.transData.transform(list(zip(xs, ys)))
    frame = ax.get_window_extent()
    dots: List[Box] = [(x - 6 * px, y - 6 * px, x + 6 * px, y + 6 * px)
                       for x, y in pts]
    placed: List[Box] = []
    offsets = [(r * math.cos(math.radians(a)), r * math.sin(math.radians(a)))
               for r in (13, 22, 32, 44)
               for a in (90, 270, 45, 135, 315, 225, 0, 180)]
    # most crowded first, so the points with fewest options choose first
    crowd = [sum(math.hypot(x - u, y - v) < 60 * px for u, v in pts)
             for x, y in pts]
    for i in sorted(range(len(names)), key=lambda i: -crowd[i]):
        w = (len(names[i]) * fontsize * 0.58 + 4) * px
        h = (fontsize * 1.35) * px
        best, best_cost = offsets[0], None
        for dx, dy in offsets:
            cx, cy = pts[i][0] + dx * px, pts[i][1] + dy * px
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            cost = sum(_overlap(box, d) for k, d in enumerate(dots))
            cost += sum(_overlap(box, b) for b in placed)
            if (box[0] < frame.x0 or box[2] > frame.x1
                    or box[1] < frame.y0 or box[3] > frame.y1):
                cost += w * h
            if best_cost is None or cost < best_cost - 1e-9:
                best, best_cost = (dx, dy), cost
            if cost == 0:
                break
        dx, dy = best
        cx, cy = pts[i][0] + dx * px, pts[i][1] + dy * px
        placed.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        far = math.hypot(dx, dy) > 18
        ax.annotate(
            names[i], (xs[i], ys[i]), textcoords="offset points",
            xytext=(dx, dy), ha="center", va="center", fontsize=fontsize,
            color=color, fontweight="bold" if names[i] in bold else "normal",
            arrowprops=(dict(arrowstyle="-", color=leader, lw=0.6,
                             shrinkA=1, shrinkB=5) if far else None),
            zorder=4)
