"""
Data-driven sheet layout locator.

Finds the actual printed bubbles on an aligned OMR sheet and fits the grid of
every field (name, roll number, class, section, school code, answers) to them.

Why: the printed layout is not constant. Different print runs / printers /
scanners throughout the year change the scale, margins and even the row
spacing of the form relative to the corner markers (e.g. the 2026-27 sheets
have ~58px name rows vs ~52px on 2025-26 in the canonical frame). Hardcoded
coordinates break on every such change; locating bubbles from the image does
not.

Pipeline:
1. Ink mask (coloured print + dark pencil/pen) on the aligned image
2. Remove long ruled lines, keep them separately to find the field boxes
3. Fill the inside of every printed ring so bubbles become solid disks
4. Morphological opening keeps only disk-sized blobs -> bubble centres
5. Per field box, fit a lattice (rows x cols) to those centres
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SheetLayout:
    """Bubble centres (x, y) for every field, in aligned-image coordinates."""
    name: np.ndarray          # (26 rows, N cols, 2); N = 25 or 27 depending on print
    roll_number: np.ndarray   # (10 rows, 3 cols, 2)
    class_: np.ndarray        # (8 rows, 2)
    section: np.ndarray       # (8 rows, 2)
    school_code: np.ndarray   # (10 rows, 3 cols, 2)
    answers: np.ndarray       # (60 questions, 4 options, 2)
    bubble_radius: float


class LayoutLocator:
    """Locates bubble grids on an aligned sheet (see module docstring)."""

    NAME_ROWS, NAME_COLS = 26, 25
    MIN_NAME_COLS, MAX_NAME_COLS = 20, 32
    INFO_ROWS = 10            # roll / school code digits 0-9
    INFO_COLS = 8             # roll x3, class, section, school code x3
    CLASS_ROWS = 8            # classes 3-10, sections A-H
    ANSWER_ROWS, ANSWER_BLOCKS, ANSWER_OPTIONS = 10, 6, 4

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.last_error = ""

    # ------------------------------------------------------------------ public

    def locate(self, bgr: np.ndarray) -> Optional[SheetLayout]:
        """Return the fitted layout, or None if the sheet could not be located."""
        self.last_error = ""
        try:
            if bgr is None or len(bgr.shape) != 3:
                self.last_error = "colour image required"
                return None

            ink = self._ink_mask(bgr)
            bubbles_only, lines = self._split_lines(ink)
            centres, radius = self._bubble_centres(bubbles_only)
            if len(centres) < 200:
                self.last_error = f"only {len(centres)} bubbles found"
                return None

            boxes = self._find_boxes(lines, bgr.shape[1], bgr.shape[0])
            if boxes is None:
                self.last_error = "field boxes not found"
                return None
            name_box, info_box, answer_box = boxes

            rings = cv2.dilate(ink, np.ones((5, 5), np.uint8))
            ring_r = radius * 0.85

            name = self._fit_name(self._inside(centres, name_box), radius)
            info = self._fit_info(self._inside(centres, info_box), radius,
                                  bubbles_only, info_box, rings)
            answers = self._fit_answers(self._inside(centres, answer_box), radius)
            if name is None or info is None or answers is None:
                self.last_error = "grid fit failed: " + ", ".join(
                    n for n, v in (("name", name), ("info", info), ("answers", answers)) if v is None)
                return None

            # A row of faint, undetected bubbles plus handwriting next to the
            # grid (written digits, "School Name") can slide a fit by one row.
            # Verify each field against ring evidence in the ink and correct.
            name = self._best_row_shift(rings, [name], ring_r)[0]
            info = self._best_row_shift(rings, list(info), ring_r)
            # answers is (block*10 + row, option); put rows on axis 0 for the check
            rows_first = answers.reshape(self.ANSWER_BLOCKS, self.ANSWER_ROWS, self.ANSWER_OPTIONS, 2)
            rows_first = self._best_row_shift(rings, [rows_first.transpose(1, 0, 2, 3)], ring_r)[0]
            answers = rows_first.transpose(1, 0, 2, 3).reshape(-1, self.ANSWER_OPTIONS, 2)

            roll, class_, section, school = info
            return SheetLayout(
                name=name, roll_number=roll, class_=class_, section=section,
                school_code=school, answers=answers, bubble_radius=radius,
            )
        except Exception as e:  # never break processing; caller falls back
            self.last_error = f"layout error: {e}"
            return None

    # ------------------------------------------------------------- detection

    @staticmethod
    def _ink_mask(bgr: np.ndarray) -> np.ndarray:
        """Printed colour (red/pink/blue) plus dark marks."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        mask = (((sat > 60) & (val > 60)) | (val < 120)).astype(np.uint8) * 255
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    @staticmethod
    def _split_lines(mask: np.ndarray, length: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Separate long horizontal/vertical ruled lines from everything else."""
        h = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1)))
        v = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, length)))
        lines = cv2.bitwise_or(h, v)
        return cv2.subtract(mask, lines), lines

    @classmethod
    def _bubble_centres(cls, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fill printed rings into disks and return centres of disk-sized blobs.

        Runs twice: on the raw mask, and on a slightly dilated mask that closes
        small breaks in thin/light printed rings (common on blue Chayan sheets
        and faint prints). Results are merged.
        """
        pts, sizes = cls._disk_blobs(mask)
        grown = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        pts2, sizes2 = cls._disk_blobs(grown)

        a = np.array(pts, dtype=np.float64).reshape(-1, 2)
        b = np.array(pts2, dtype=np.float64).reshape(-1, 2)
        if len(a) and len(b):
            dist = np.hypot(b[:, None, 0] - a[None, :, 0], b[:, None, 1] - a[None, :, 1])
            new = dist.min(axis=1) > 10
        else:
            new = np.ones(len(b), dtype=bool)
        merged = np.vstack([a, b[new]])
        merged_sizes = list(sizes) + [s - 2 for s, n in zip(sizes2, new) if n]  # undo dilation growth
        if not len(merged):
            return np.zeros((0, 2)), 0.0
        return merged, float(np.median(merged_sizes))

    @staticmethod
    def _disk_blobs(mask: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        flood = mask.copy()
        h, w = mask.shape
        cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
        filled = mask | cv2.bitwise_not(flood)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        disks = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)

        n, _, stats, cents = cv2.connectedComponentsWithStats(disks)
        pts, sizes = [], []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if 24 <= bw <= 60 and 24 <= bh <= 60 and 0.7 < bw / bh < 1.4 and area > 0.5 * bw * bh:
                pts.append(cents[i])
                sizes.append((bw + bh) / 4.0)
        return pts, sizes

    @staticmethod
    def _find_boxes(lines: np.ndarray, width: int, height: int):
        """Find name box, info (roll/class/...) box and answer box from ruled lines."""
        thick = cv2.dilate(lines, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(thick, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(c) for c in contours]
        rects = [r for r in rects if r[2] > 0.12 * width and r[3] > 0.06 * height]

        answers = [r for r in rects if r[2] > 0.7 * width and r[1] > 0.4 * height]
        if not answers:
            return None
        answer_box = min(answers, key=lambda r: r[2] * r[3])  # inner border

        above = [r for r in rects if r[1] + r[3] <= answer_box[1] + 10]
        name = [r for r in above if r[0] < 0.3 * width and r[2] > 0.35 * width and r[3] > 0.25 * height]
        right = [r for r in above if r[0] > 0.5 * width]
        if not name or not right:
            return None
        name_box = min(name, key=lambda r: r[2] * r[3])
        # The info box is the lowest box on the right (instructions box is above it)
        lowest_top = max(r[1] for r in right)
        info_box = min((r for r in right if r[1] > lowest_top - 30), key=lambda r: r[2] * r[3])
        return name_box, info_box, answer_box

    @staticmethod
    def _ring_score(rings: np.ndarray, centres: np.ndarray, r: float) -> float:
        """Mean fraction of a circle of radius r (around each centre) on ink."""
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        pts = centres.reshape(-1, 2)
        xs = np.round(pts[:, 0:1] + r * np.cos(angles)[None, :]).astype(int)
        ys = np.round(pts[:, 1:2] + r * np.sin(angles)[None, :]).astype(int)
        h, w = rings.shape
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        hits = np.zeros(xs.shape, dtype=bool)
        hits[inside] = rings[ys[inside], xs[inside]] > 0
        return float(hits.mean())

    @classmethod
    def _best_row_shift(cls, rings: np.ndarray, grids: List[np.ndarray], r: float) -> List[np.ndarray]:
        """
        Shift a field's grids (which share one row lattice) by -1/0/+1 rows,
        keeping the shift whose bubble positions sit best on printed rings.
        """
        def row_axis_step(g):
            # Rows are axis 0; step = mean spacing between consecutive rows
            return (g[-1] - g[0]) / (len(g) - 1)

        best, best_score = grids, -1.0
        for shift in (0, -1, 1):
            moved = [g + shift * row_axis_step(g) for g in grids]
            score = np.mean([cls._ring_score(rings, g, r) for g in moved])
            if score > best_score + (0.02 if shift else 0):  # prefer no shift on ties
                best, best_score = moved, score
        return best

    @staticmethod
    def _inside(pts: np.ndarray, rect) -> np.ndarray:
        x, y, w, h = rect
        keep = (pts[:, 0] > x) & (pts[:, 0] < x + w) & (pts[:, 1] > y) & (pts[:, 1] < y + h)
        return pts[keep]

    # ---------------------------------------------------------- 1-D helpers

    @staticmethod
    def _cluster_1d(values: np.ndarray, tol: float) -> List[Tuple[float, int]]:
        """Cluster sorted 1-D values; returns (mean, count) per cluster."""
        if len(values) == 0:
            return []
        values = np.sort(values)
        clusters, current = [], [values[0]]
        for v in values[1:]:
            if v - current[-1] <= tol:
                current.append(v)
            else:
                clusters.append(current)
                current = [v]
        clusters.append(current)
        return [(float(np.mean(c)), len(c)) for c in clusters]

    @staticmethod
    def _fit_lattice(clusters: List[Tuple[float, int]], n: int, min_step: float,
                     max_step: float, tol: float) -> Optional[Tuple[float, float]]:
        """
        Fit start + k*step (k = 0..n-1) to weighted 1-D cluster centres.

        Returns (start, step) maximising the matched weight, refined by
        weighted least squares, or None.
        """
        if len(clusters) < 3:
            return None
        centres = np.array([c for c, _ in clusters])
        weights = np.array([w for _, w in clusters], dtype=np.float64)

        # Candidate steps from pairwise differences divided by small integers
        diffs = np.abs(centres[:, None] - centres[None, :])[np.triu_indices(len(centres), 1)]
        cand = np.concatenate([diffs / k for k in range(1, 4)])
        cand = cand[(cand >= min_step) & (cand <= max_step)]
        if len(cand) == 0:
            return None
        cand = np.unique(np.round(cand, 0))

        best, best_score = None, -1.0
        for step in cand:
            # Try every assignment of every centre to every lattice index
            starts = (centres[:, None] - np.arange(n)[None, :] * step).ravel()
            k = np.round((centres[None, :] - starts[:, None]) / step)          # (S, C)
            ok = ((k >= 0) & (k < n)
                  & (np.abs(centres[None, :] - (starts[:, None] + k * step)) < tol))
            scores = (ok * weights[None, :]).sum(axis=1)
            i = int(np.argmax(scores))
            if scores[i] > best_score:
                best_score, best = scores[i], (starts[i], step, ok[i], k[i])
        if best is None:
            return None

        start, step, ok, k = best
        if ok.sum() >= 2:
            A = np.stack([np.ones(ok.sum()), k[ok]], axis=1)
            W = np.sqrt(weights[ok])
            sol, *_ = np.linalg.lstsq(A * W[:, None], centres[ok] * W, rcond=None)
            start, step = float(sol[0]), float(sol[1])
        return start, step

    @staticmethod
    def _affine_grid(pts: np.ndarray, x0: float, dx: float, y0: float, dy: float,
                     rows: int, cols: int, tol: float) -> np.ndarray:
        """
        Assign points to (row, col) of an initial axis-aligned lattice, then fit
        x = a0 + a1*col + a2*row, y = b0 + b1*col + b2*row to absorb residual
        rotation/shear. Returns (rows, cols, 2) centres.
        """
        c = np.round((pts[:, 0] - x0) / dx)
        r = np.round((pts[:, 1] - y0) / dy)
        ok = ((c >= 0) & (c < cols) & (r >= 0) & (r < rows)
              & (np.abs(pts[:, 0] - (x0 + c * dx)) < tol)
              & (np.abs(pts[:, 1] - (y0 + r * dy)) < tol))

        ax = np.array([x0, dx, 0.0])
        ay = np.array([y0, 0.0, dy])
        if ok.sum() >= 6:
            A = np.stack([np.ones(ok.sum()), c[ok], r[ok]], axis=1)
            ax, *_ = np.linalg.lstsq(A, pts[ok, 0], rcond=None)
            ay, *_ = np.linalg.lstsq(A, pts[ok, 1], rcond=None)

        cc, rr = np.meshgrid(np.arange(cols), np.arange(rows))
        gx = ax[0] + ax[1] * cc + ax[2] * rr
        gy = ay[0] + ay[1] * cc + ay[2] * rr
        return np.stack([gx, gy], axis=-1)

    # ------------------------------------------------------------ field fits

    def _fit_name(self, pts: np.ndarray, radius: float) -> Optional[np.ndarray]:
        if len(pts) < 60:
            return None
        tol = radius * 0.6
        lo, hi = radius * 2.0, radius * 4.0
        yf = self._fit_lattice(self._cluster_1d(pts[:, 1], tol), self.NAME_ROWS, lo, hi, tol)
        if yf is None:
            return None

        # The number of name columns differs between print runs (25 on
        # 2025-26 sheets, 27 on 2026-27), so derive it from the data: every
        # well-supported column cluster on the step lattice counts.
        x_clusters = self._cluster_1d(pts[:, 0], tol)
        xf = self._fit_lattice(x_clusters, self.NAME_COLS, lo, hi, tol)
        if xf is None:
            return None
        step = xf[1]
        strong = [c for c, n in x_clusters if n >= max(3, 0.15 * max(w for _, w in x_clusters))]
        cols = int(round((max(strong) - min(strong)) / step)) + 1 if strong else self.NAME_COLS
        if not self.MIN_NAME_COLS <= cols <= self.MAX_NAME_COLS:
            cols = self.NAME_COLS
        xf = self._fit_lattice(x_clusters, cols, lo, hi, tol)
        if xf is None:
            return None
        return self._affine_grid(pts, xf[0], xf[1], yf[0], yf[1],
                                 self.NAME_ROWS, cols, tol)

    def _fit_info(self, pts: np.ndarray, radius: float, ink: np.ndarray, box,
                  rings: np.ndarray):
        if len(pts) < 10:
            return None
        tol = radius * 0.6

        # Columns first: a bubble column stacks up to 10 detections at one x,
        # whereas handwriting (written digits, school name) scatters in x.
        cols = [c for c in self._cluster_1d(pts[:, 0], tol) if c[1] >= 2]
        xs = sorted(c for c, _ in sorted(cols, key=lambda c: -c[1])[:self.INFO_COLS])
        if len(xs) == self.INFO_COLS:
            near = np.min(np.abs(pts[:, 0:1] - np.array(xs)[None, :]), axis=1) < tol
            yf = self._fit_lattice(self._cluster_1d(pts[near, 1], tol), self.INFO_ROWS,
                                   radius * 2.0, radius * 4.0, tol)
            if yf is None:
                return None
            y0, dy = yf
            r = np.round((pts[:, 1] - y0) / dy)
            on_rows = (near & (r >= 0) & (r < self.INFO_ROWS)
                       & (np.abs(pts[:, 1] - (y0 + r * dy)) < tol))
        else:
            # Faint print: too few filled rings to find 8 columns. Fit rows on
            # all points; columns are chosen from ring evidence below.
            yf = self._fit_lattice(self._cluster_1d(pts[:, 1], tol), self.INFO_ROWS,
                                   radius * 2.0, radius * 4.0, tol)
            if yf is None:
                return None
            y0, dy = yf
            r = np.round((pts[:, 1] - y0) / dy)
            on_rows = (r >= 0) & (r < self.INFO_ROWS) & (np.abs(pts[:, 1] - (y0 + r * dy)) < tol)

        # Final column choice by ring evidence: candidates are the point
        # clusters plus ink-profile peaks; a faint real column with few
        # detections still has printed rings, a handwriting cluster doesn't.
        cands = [c for c, _ in self._cluster_1d(pts[:, 0], tol)]
        cands += self._profile_columns(ink, box, y0 - radius,
                                       y0 + (self.INFO_ROWS - 1) * dy + radius,
                                       2 * self.INFO_COLS, radius) or []
        ys = y0 + np.arange(self.CLASS_ROWS) * dy
        scored = sorted(((self._ring_score(rings, np.stack([np.full_like(ys, x), ys], -1),
                                           radius * 0.85), x) for x in cands), reverse=True)
        xs = []
        for score, x in scored:
            if all(abs(x - k) > 1.5 * radius for k in xs):
                xs.append(x)
            if len(xs) == self.INFO_COLS:
                break
        if len(xs) < self.INFO_COLS:
            return None
        # Snap each chosen column to its point cluster centre when available
        clusters = [c for c, n in self._cluster_1d(pts[:, 0], tol) if n >= 2]
        xs = sorted(min(clusters, key=lambda c: abs(c - x)) if clusters and
                    min(abs(c - x) for c in clusters) < tol else x for x in xs)

        # Per-column refinement: y = y0 + r*dy + slope*x (page tilt)
        sel = pts[on_rows]
        rsel = r[on_rows]
        A = np.stack([np.ones(len(sel)), rsel, sel[:, 0]], axis=1)
        coef, *_ = np.linalg.lstsq(A, sel[:, 1], rcond=None)

        def column(x, rows):
            ys = coef[0] + coef[1] * np.arange(rows) + coef[2] * x
            return np.stack([np.full(rows, x), ys], axis=-1)

        roll = np.stack([column(x, self.INFO_ROWS) for x in xs[0:3]], axis=1)
        class_ = column(xs[3], self.CLASS_ROWS)
        section = column(xs[4], self.CLASS_ROWS)
        school = np.stack([column(x, self.INFO_ROWS) for x in xs[5:8]], axis=1)
        return roll, class_, section, school

    @staticmethod
    def _profile_columns(ink: np.ndarray, box, y_top: float, y_bottom: float,
                         count: int, radius: float) -> Optional[List[float]]:
        """Find the `count` strongest bubble columns from the ink x-profile."""
        x, y, w, h = box
        y1, y2 = max(int(y_top), y), min(int(y_bottom), y + h)
        if y2 <= y1:
            return None
        band = ink[y1:y2, x + 10:x + w - 10] > 0
        profile = band.sum(axis=0).astype(np.float64)
        win = max(3, int(radius))
        profile = np.convolve(profile, np.ones(win) / win, mode="same")

        # Greedy peak picking with minimum separation of ~1.5 radii
        order = np.argsort(profile)[::-1]
        peaks: List[int] = []
        for i in order:
            if profile[i] <= 0:
                break
            if all(abs(i - j) > 1.5 * radius for j in peaks):
                peaks.append(int(i))
                if len(peaks) == count:
                    break
        if len(peaks) < count:
            return None
        return sorted(float(p + x + 10) for p in peaks)

    def _fit_answers(self, pts: np.ndarray, radius: float) -> Optional[np.ndarray]:
        if len(pts) < 60:
            return None
        tol = radius * 0.6
        yf = self._fit_lattice(self._cluster_1d(pts[:, 1], tol), self.ANSWER_ROWS,
                               radius * 2.5, radius * 7.0, tol)
        if yf is None:
            return None
        y0, dy = yf
        r = np.round((pts[:, 1] - y0) / dy)
        on_rows = (r >= 0) & (r < self.ANSWER_ROWS) & (np.abs(pts[:, 1] - (y0 + r * dy)) < tol)
        sel, rsel = pts[on_rows], r[on_rows]

        cols = self._cluster_1d(sel[:, 0], tol)
        block_xs = self._fit_answer_columns(cols, radius, tol)
        if block_xs is None:
            return None

        A = np.stack([np.ones(len(sel)), rsel, sel[:, 0]], axis=1)
        coef, *_ = np.linalg.lstsq(A, sel[:, 1], rcond=None)

        out = np.zeros((self.ANSWER_BLOCKS * self.ANSWER_ROWS, self.ANSWER_OPTIONS, 2))
        for b, bx in enumerate(block_xs):
            for row in range(self.ANSWER_ROWS):
                q = b * self.ANSWER_ROWS + row
                for o, x in enumerate(bx):
                    out[q, o] = (x, coef[0] + coef[1] * row + coef[2] * x)
        return out

    def _fit_answer_columns(self, clusters: List[Tuple[float, int]], radius: float,
                            tol: float) -> Optional[List[List[float]]]:
        """
        Fit x = origin + block*pitch + option*step (6 blocks x 4 options) to
        weighted x-clusters. Stray clusters (question numbers, stamps) simply
        don't match the model, so they can't derail the fit.
        """
        if len(clusters) < 8:
            return None
        xs = np.array([c for c, _ in clusters])
        ws = np.array([w for _, w in clusters], dtype=np.float64)
        nb, no = self.ANSWER_BLOCKS, self.ANSWER_OPTIONS

        def peak(lo, hi, rank_by):
            d = xs[None, :] - xs[:, None]
            wgt = ws[:, None] * ws[None, :]
            m = (d >= lo) & (d <= hi)
            if not m.any():
                return None
            bins = np.arange(int(lo), int(hi) + 2)
            hist, _ = np.histogram(d[m], bins=bins, weights=wgt[m])
            hist = np.convolve(hist, np.ones(5) / 5, mode="same")
            return float(bins[int(np.argmax(hist))]) if rank_by else None

        step = peak(radius * 2.0, radius * 4.0, True)
        if step is None:
            return None
        pitch = peak(step * 4.4, step * 9.0, True)
        if pitch is None:
            return None

        # Joint search over small perturbations and every anchoring
        anchors = xs[np.argsort(-ws)[:8]]
        idx_b, idx_o = np.meshgrid(np.arange(nb), np.arange(no), indexing="ij")
        idx_b, idx_o = idx_b.ravel(), idx_o.ravel()
        best, best_score = None, -1.0
        for s in np.arange(step - 3, step + 3.1, 0.5):
            for p in np.arange(pitch - 6, pitch + 6.1, 1.0):
                offs = idx_b * p + idx_o * s                       # (24,)
                origins = (anchors[:, None] - offs[None, :]).ravel()  # (8*24,)
                pos = origins[:, None] + offs[None, :]             # (A, 24)
                dist = np.abs(pos[:, :, None] - xs[None, None, :])  # (A, 24, C)
                hit = dist.min(axis=2) < tol
                near = dist.argmin(axis=2)
                score = (ws[near] * hit).sum(axis=1)
                k = int(np.argmax(score))
                if score[k] > best_score:
                    best_score, best = score[k], (origins[k], s, p)
        if best is None:
            return None
        origin, s, p = best

        # Require most of the 24 columns to be supported by detected bubbles
        pos = origin + idx_b * p + idx_o * s
        support = np.abs(pos[:, None] - xs[None, :]).min(axis=1) < tol
        if support.sum() < nb * no * 0.75:
            return None

        # Snap supported columns to their detected cluster centre
        out = []
        for b in range(nb):
            row = []
            for o in range(no):
                x = origin + b * p + o * s
                j = int(np.argmin(np.abs(xs - x)))
                row.append(float(xs[j]) if abs(xs[j] - x) < tol else float(x))
            out.append(row)
        return out


def locate_layout(bgr: np.ndarray) -> Optional[SheetLayout]:
    """Convenience wrapper."""
    return LayoutLocator().locate(bgr)
