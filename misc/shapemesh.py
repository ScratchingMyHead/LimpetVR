#!/usr/bin/env python3
"""
shapemesh.py - 9x9 mesh shape editor for LimpetVR.

Uses and updates: shapes.json in the same directory as this script.

Main screen:
  - loads all currently defined shapes
  - add new / remove / rename / edit shapes
  - Save button writes shapes.json
  - each shape shows a thumbnail: distorted 9x9 grid + coloured dots
    for points that have moved (colour intensity grows with displacement)

Mesh deformer window (opened via Edit):
  - 9x9 line grid (8x8 cells), all 81 points draggable
  - multi-select: click selects, Shift/Ctrl-click toggles, drag on empty
    area draws a marquee box; dragging / arrow keys move the whole selection
    together by the same delta (snap adjusts the shared delta via the anchor)
  - single-level undo (button or Ctrl+Z)
  - snap-to-grid option with definable divisions + snap-grid overlay
    (8 and 16 line up exactly with the 9x9 mesh points)
  - background image selector; retained across mesh edits within one run,
    defaults to white when nothing loaded
  - left side is primary: buttons to mirror L->R
  - top is primary: button to mirror T->B
  - smooth row / column helpers

Grid convention:
  - grid is 9x9x2, values (u, v) in normalised coords, u left->right,
    v top->bottom. Identity is the regular grid u=j/8, v=i/8.
  - warp ("pull") convention: each mesh point shows the background texel
    from its identity position, so dragging a point drags image content
    with it. Identity = no-op.
  - stored row-major in shapes.json as 81 [u, v] pairs (row 0 = top).

shapes.json format:
  {"shapes": [{"id": "a1b2c3d4", "name": "flat", "grid": [[u, v], ... 81 pairs]}]}
  ids are stable per mesh (kept across renames, assigned on save if missing).
"""

import json
import math
import os
import sys
import tkinter as tk
import uuid
from tkinter import filedialog, messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from PIL import ImageTk
except Exception:  # headless / missing tkinter bits: GUI will fail later, logic still works
    ImageTk = None

N = 9  # 9x9 points -> 8x8 cells
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHAPES_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "shapes.json"))

# Session-retained background (kept across mesh edits, not across runs).
SESSION_BG_PATH = {"path": None}
SESSION_BG_IMAGE = {"img": None}  # cached PIL RGB image


# ----------------------------------------------------------------------------
# grid / shapes.json helpers (pure logic, headless-testable)
# ----------------------------------------------------------------------------

def identity_grid():
    """Return a fresh (9,9,2) float64 identity grid."""
    g = np.zeros((N, N, 2), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            g[i, j, 0] = j / (N - 1)
            g[i, j, 1] = i / (N - 1)
    return g


def grid_to_list(grid):
    """(9,9,2) -> 81 row-major [u, v] pairs."""
    g = np.asarray(grid, dtype=np.float64).reshape(N, N, 2)
    return [[float(g[i, j, 0]), float(g[i, j, 1])] for i in range(N) for j in range(N)]


def list_to_grid(lst):
    """81 row-major pairs (or 9x9 nested) -> (9,9,2) array. Raises ValueError."""
    a = np.asarray(lst, dtype=np.float64)
    if a.shape == (N, N, 2):
        return a.copy()
    if a.shape == (N * N, 2):
        return a.reshape(N, N, 2).copy()
    raise ValueError(f"grid must be 9x9x2 or 81x2, got shape {a.shape}")


def empty_shapes():
    return []


def new_shape_id(existing=()):
    """Generate a unique 8-hex-char mesh id not present in existing."""
    taken = {s.get("id") for s in existing if isinstance(s, dict)}
    while True:
        cand = uuid.uuid4().hex[:8]
        if cand not in taken:
            return cand


def ensure_shape_ids(shapes):
    """Assign ids to any shape missing one (in place). Returns the list."""
    for s in shapes:
        if not s.get("id"):
            s["id"] = new_shape_id(shapes)
    return shapes


def load_shapes(path=SHAPES_PATH):
    """Load shape list from path.

    Returns list of {'id': str|None, 'name': str, 'grid': (9,9,2)}.
    Older files without ids load fine (id None, assigned on save).
    """
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "shapes" in data:
        raw = data["shapes"]
    elif isinstance(data, list):
        raw = data
    else:
        raise ValueError("shapes.json must contain {'shapes': [...]} or a plain list")
    shapes = []
    for entry in raw:
        name = str(entry.get("name", "shape"))
        sid = entry.get("id", None)
        sid = str(sid) if sid is not None else None
        grid = list_to_grid(entry["grid"])
        shapes.append({"id": sid, "name": name, "grid": grid})
    return shapes


def save_shapes(shapes, path=SHAPES_PATH):
    """Write shape list to path (creates parent dirs).

    Every mesh is saved with a stable id; shapes missing one are assigned
    ids in place so the caller sees them too.
    """
    ensure_shape_ids(shapes)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    data = {"shapes": [{"id": s.get("id"), "name": s["name"],
                        "grid": grid_to_list(s["grid"])} for s in shapes]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def displacement(grid):
    """(9,9,2) magnitudes of offset from identity, in normalised units."""
    return np.linalg.norm(np.asarray(grid) - identity_grid(), axis=-1)


def max_displacement(grid):
    return float(displacement(grid).max())


def is_inward_move(grid, i, j, eps=1e-9):
    """True if point (i,j) has moved toward the centre (0.5,0.5).

    Dot product of displacement vs vector to centre > eps.
    Centre point itself never counts as inward.
    """
    g = np.asarray(grid, dtype=np.float64)
    ix, iy = j / (N - 1), i / (N - 1)
    dx, dy = float(g[i, j, 0] - ix), float(g[i, j, 1] - iy)
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return False
    cx, cy = 0.5 - ix, 0.5 - iy
    if abs(cx) < 1e-9 and abs(cy) < 1e-9:
        return False
    return (dx * cx + dy * cy) > eps


# --- mirror helpers: left side / top side is primary ------------------------

def mirror_lr_flip(grid):
    """Mirror left half to right half across vertical centre line.

    Right side becomes a true mirror image: u mirrored, v copied.
    Centre column is straightened to u=0.5.
    Operates in place on the (9,9,2) array, also returns it.
    """
    g = np.asarray(grid, dtype=np.float64)
    for i in range(N):
        for j in range(4):  # cols 0..3 -> cols 8..5
            g[i, N - 1 - j, 0] = 1.0 - g[i, j, 0]
            g[i, N - 1 - j, 1] = g[i, j, 1]
        g[i, 4, 0] = 0.5
    return g


def mirror_lr_copy(grid):
    """Copy left-side *offsets* to the right side (same dx, dy, not mirrored).

    Useful when the mesh is intentionally asymmetric but you want the same
    nudge applied on both sides. Centre column untouched.
    Operates in place, also returns it.
    """
    g = np.asarray(grid, dtype=np.float64)
    ident = identity_grid()
    for i in range(N):
        for j in range(4):
            dx = g[i, j, 0] - ident[i, j, 0]
            dy = g[i, j, 1] - ident[i, j, 1]
            g[i, N - 1 - j, 0] = ident[i, N - 1 - j, 0] + dx
            g[i, N - 1 - j, 1] = ident[i, N - 1 - j, 1] + dy
    return g


def mirror_tb_flip(grid):
    """Mirror top half to bottom half across horizontal centre line.

    Bottom becomes mirror image: v mirrored, u copied. Centre row v=0.5.
    Operates in place, also returns it.
    """
    g = np.asarray(grid, dtype=np.float64)
    for i in range(4):  # rows 0..3 -> rows 8..5
        for j in range(N):
            g[N - 1 - i, j, 0] = g[i, j, 0]
            g[N - 1 - i, j, 1] = 1.0 - g[i, j, 1]
    for j in range(N):
        g[4, j, 1] = 0.5
    return g


# --- smoothing ----------------------------------------------------------------

def smooth_row(grid, i, alpha=0.5):
    """One gentle Laplacian smooth pass over row i (in place).

    Interior: p = (1-alpha)*p + alpha*(left+right)/2, alpha=0.5 default.
    Endpoints are pulled slightly toward their single neighbour.
    """
    g = np.asarray(grid, dtype=np.float64)
    row = g[i].copy()
    new = row.copy()
    for j in range(1, N - 1):
        avg = (row[j - 1] + row[j + 1]) * 0.5
        new[j] = (1 - alpha) * row[j] + alpha * avg
    new[0] = (1 - 0.5 * alpha) * row[0] + (0.5 * alpha) * row[1]
    new[N - 1] = (1 - 0.5 * alpha) * row[N - 1] + (0.5 * alpha) * row[N - 2]
    g[i] = new
    return g


def smooth_col(grid, j, alpha=0.5):
    """One gentle Laplacian smooth pass over column j (in place)."""
    g = np.asarray(grid, dtype=np.float64)
    col = g[:, j].copy()
    new = col.copy()
    for i in range(1, N - 1):
        avg = (col[i - 1] + col[i + 1]) * 0.5
        new[i] = (1 - alpha) * col[i] + alpha * avg
    new[0] = (1 - 0.5 * alpha) * col[0] + (0.5 * alpha) * col[1]
    new[N - 1] = (1 - 0.5 * alpha) * col[N - 1] + (0.5 * alpha) * col[N - 2]
    g[:, j] = new
    return g


def smooth_all(grid, alpha=0.5):
    """One smooth pass over every row then every column (in place)."""
    for i in range(N):
        smooth_row(grid, i, alpha)
    for j in range(N):
        smooth_col(grid, j, alpha)
    return grid


# --- multi-select move + snap (pure logic, headless-testable) -------------------

U_MIN, U_MAX = 0.0, 1.0  # points stay within the visible bounds


def clamp_uv(u, v):
    """Clamp (u, v) to the visible [0, 1] range."""
    return min(U_MAX, max(U_MIN, u)), min(U_MAX, max(U_MIN, v))


def snap_value(x, divisions):
    """Snap a single coordinate to multiples of 1/divisions."""
    if divisions is None or divisions < 2:
        return x
    return round(float(x) * divisions) / divisions


def snap_points(grid, divisions, selection=None):
    """Snap points to a 1/divisions lattice (in place).

    selection: iterable of (i, j) or None for all 81 points.
    """
    g = np.asarray(grid, dtype=np.float64)
    pts = selection if selection is not None else [(i, j) for i in range(N) for j in range(N)]
    for (i, j) in pts:
        g[i, j, 0] = snap_value(g[i, j, 0], divisions)
        g[i, j, 1] = snap_value(g[i, j, 1], divisions)
    return g


def snapped_group_delta(grid, anchor, du, dv, snap_div):
    """Adjust (du, dv) so the anchor point lands on the snap lattice.

    Returns the adjusted delta; apply it uniformly to the whole selection
    so every point moves by exactly the same amount.
    """
    g = np.asarray(grid, dtype=np.float64)
    ou, ov = float(g[anchor[0], anchor[1], 0]), float(g[anchor[0], anchor[1], 1])
    tu, tv = clamp_uv(ou + du, ov + dv)
    tu, tv = snap_value(tu, snap_div), snap_value(tv, snap_div)
    return tu - ou, tv - ov


def move_points_by_delta(grid, selection, du, dv, snap_div=None, anchor=None):
    """Move every point in selection by (du, dv) (in place).

    Each point keeps its relative offset; results are clamped to the
    visible [0, 1] range. If snap_div and anchor are given, the shared delta is
    first adjusted so the anchor lands on the lattice, then applied
    uniformly (so the group stays rigid). Without an anchor, each point
    is snapped individually (legacy behaviour, can shear a group).
    selection: iterable of (i, j). Empty selection = no-op.
    """
    g = np.asarray(grid, dtype=np.float64)
    sel = [tuple(pt) for pt in selection]
    if not sel:
        return g
    if snap_div is not None and anchor is not None and tuple(anchor) in sel:
        du, dv = snapped_group_delta(g, tuple(anchor), du, dv, snap_div)
        snap_div = None
    # capture originals first so overlapping updates can't cascade
    orig = {pt: (float(g[pt[0], pt[1], 0]), float(g[pt[0], pt[1], 1])) for pt in sel}
    for pt in sel:
        ou, ov = orig[pt]
        nu, nv = clamp_uv(ou + du, ov + dv)
        if snap_div is not None:
            nu, nv = snap_value(nu, snap_div), snap_value(nv, snap_div)
        g[pt[0], pt[1], 0], g[pt[0], pt[1], 1] = nu, nv
    return g


# --- warp preview ---------------------------------------------------------------

def build_remap(grid, w, h):
    """Build cv2.remap maps (map_x, map_y) for image size w x h.

    "Pull" convention: each mesh point marks where the source texel from
    the corresponding identity position is displayed, so dragging a point
    pulls image content along with it. Implemented as the exact inverse
    of the bilinear mesh: every display pixel is located inside a mesh
    cell (split into two triangles, as in dome_transform.py) and mapped
    back to source coords with that triangle's affine transform.
    Pixels not covered by any triangle (outside the mesh / degenerate
    cells) fall back to identity. Identity mesh = no-op.
    """
    g = np.asarray(grid, dtype=np.float64).reshape(N, N, 2)
    frac = np.arange(N, dtype=np.float64) / (N - 1)
    src_x = np.tile(frac[None, :], (N, 1)) * (w - 1)
    src_y = np.tile(frac[:, None], (1, N)) * (h - 1)
    dst_x = g[:, :, 0] * (w - 1)
    dst_y = g[:, :, 1] * (h - 1)

    yy, xx = np.mgrid[0:h, 0:w]
    map_x = xx.astype(np.float32)
    map_y = yy.astype(np.float32)

    for cj in range(N - 1):
        for ci in range(N - 1):
            dx = np.array([[dst_x[cj, ci], dst_y[cj, ci]],
                           [dst_x[cj, ci + 1], dst_y[cj, ci + 1]],
                           [dst_x[cj + 1, ci + 1], dst_y[cj + 1, ci + 1]],
                           [dst_x[cj + 1, ci], dst_y[cj + 1, ci]]])
            sx = np.array([[src_x[cj, ci], src_y[cj, ci]],
                           [src_x[cj, ci + 1], src_y[cj, ci + 1]],
                           [src_x[cj + 1, ci + 1], src_y[cj + 1, ci + 1]],
                           [src_x[cj + 1, ci], src_y[cj + 1, ci]]])
            for a, b, c in ((0, 1, 2), (0, 2, 3)):
                tri_dst = dx[[a, b, c]].astype(np.float32)
                tri_src = sx[[a, b, c]].astype(np.float32)
                area = 0.5 * abs(tri_dst[0, 0] * (tri_dst[1, 1] - tri_dst[2, 1])
                                 + tri_dst[1, 0] * (tri_dst[2, 1] - tri_dst[0, 1])
                                 + tri_dst[2, 0] * (tri_dst[0, 1] - tri_dst[1, 1]))
                if area < 0.5:
                    continue
                M = cv2.getAffineTransform(tri_dst, tri_src)
                xs, ys = tri_dst[:, 0], tri_dst[:, 1]
                x_min, x_max = max(0, int(np.floor(xs.min()))), min(w - 1, int(np.ceil(xs.max())))
                y_min, y_max = max(0, int(np.floor(ys.min()))), min(h - 1, int(np.ceil(ys.max())))
                if x_min > x_max or y_min > y_max:
                    continue
                px = np.arange(x_min, x_max + 1, dtype=np.float32)
                py = np.arange(y_min, y_max + 1, dtype=np.float32)
                PX, PY = np.meshgrid(px, py)
                v0 = tri_dst[2] - tri_dst[0]
                v1 = tri_dst[1] - tri_dst[0]
                v2x, v2y = PX - tri_dst[0, 0], PY - tri_dst[0, 1]
                dot00, dot01 = float(np.dot(v0, v0)), float(np.dot(v0, v1))
                dot11 = float(np.dot(v1, v1))
                dot02, dot12 = v2x * v0[0] + v2y * v0[1], v2x * v1[0] + v2y * v1[1]
                denom = dot00 * dot11 - dot01 * dot01
                if abs(denom) < 1e-10:
                    continue
                inv = 1.0 / denom
                u = (dot11 * dot02 - dot01 * dot12) * inv
                v = (dot00 * dot12 - dot01 * dot02) * inv
                mask = (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6)
                if not mask.any():
                    continue
                region_x = map_x[y_min:y_max + 1, x_min:x_max + 1]
                region_y = map_y[y_min:y_max + 1, x_min:x_max + 1]
                region_x[mask] = (M[0, 0] * PX + M[0, 1] * PY + M[0, 2])[mask]
                region_y[mask] = (M[1, 0] * PX + M[1, 1] * PY + M[1, 2])[mask]
    return map_x, map_y


def warp_image(pil_img, grid):
    """Warp a PIL RGB image through the mesh. Returns PIL RGB image."""
    arr = np.asarray(pil_img.convert("RGB"))
    h, w = arr.shape[:2]
    map_x, map_y = build_remap(grid, w, h)
    warped = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


# --- thumbnails ------------------------------------------------------------------

def render_thumbnail_pil(grid, size=144):
    """Render a shape thumbnail as a PIL image.

    White background, distorted 9x9 grid lines, plus one dot per control
    point: unmoved points are small grey dots; moved points are coloured
    (amber->red) with radius/intensity growing with displacement so shapes
    are distinguishable at a glance.
    """
    g = np.asarray(grid, dtype=np.float64)
    mag = displacement(g)
    mmax = float(mag.max()) if mag.size else 0.0

    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    pad = 10
    span = size - 2 * pad

    def to_px(u, v):
        return pad + float(u) * span, pad + float(v) * span

    # distorted grid lines (8x8 cells)
    for i in range(N):
        pts = [to_px(g[i, j, 0], g[i, j, 1]) for j in range(N)]
        d.line(pts, fill=(150, 150, 150), width=1)
    for j in range(N):
        pts = [to_px(g[i, j, 0], g[i, j, 1]) for i in range(N)]
        d.line(pts, fill=(150, 150, 150), width=1)

    # identity reference cross (faint)
    for j in range(N):
        x = pad + j / (N - 1) * span
        d.line([(x, pad), (x, pad + span)], fill=(235, 235, 235), width=1)
    # redraw distorted lines already done; now dots on top
    for i in range(N):
        for j in range(N):
            m = float(mag[i, j])
            x, y = to_px(g[i, j, 0], g[i, j, 1])
            if m < 1e-6:
                r = 2
                d.ellipse([x - r, y - r, x + r, y + r], fill=(170, 170, 170))
            else:
                t = m / mmax if mmax > 1e-9 else 1.0  # 0..1 intensity
                t = max(0.15, min(1.0, t))
                inward = is_inward_move(g, i, j)
                if inward:
                    # light green -> strong green
                    rcol = int(120 - 90 * t)
                    gcol = int(200 - 20 * t)
                    bcol = int(120 - 90 * t)
                    outline = (40, 120, 40)
                else:
                    # light red -> strong red
                    rcol = 255
                    gcol = int(200 - 170 * t)  # 200 -> 30
                    bcol = int(60 - 40 * t)
                    outline = (120, 0, 0)
                r = 2 + 4 * t
                d.ellipse([x - r, y - r, x + r, y + r],
                          fill=(rcol, gcol, bcol), outline=outline)
    # border
    d.rectangle([0, 0, size - 1, size - 1], outline=(90, 90, 90))
    return img


# ----------------------------------------------------------------------------
# session background handling
# ----------------------------------------------------------------------------

def get_session_bg(display_size=(640, 640)):
    """Return cached background as PIL RGB of display_size, or plain white."""
    if SESSION_BG_IMAGE["img"] is not None:
        try:
            return SESSION_BG_IMAGE["img"].convert("RGB").resize(display_size, Image.BILINEAR)
        except Exception:
            pass
    return Image.new("RGB", display_size, (255, 255, 255))


def load_session_bg(path):
    """Load image file into the session cache. Returns PIL image or raises."""
    img = Image.open(path).convert("RGB")
    SESSION_BG_PATH["path"] = path
    SESSION_BG_IMAGE["img"] = img
    return img


def clear_session_bg():
    SESSION_BG_PATH["path"] = None
    SESSION_BG_IMAGE["img"] = None


# ----------------------------------------------------------------------------
# GUI: main screen
# ----------------------------------------------------------------------------

class ShapeManagerApp(tk.Tk):
    def __init__(self, shapes_path=SHAPES_PATH):
        super().__init__()
        self.title("shapemesh - LimpetVR shape editor")
        self.geometry("640x560")
        self.shapes_path = shapes_path
        try:
            self.shapes = load_shapes(shapes_path)
        except Exception as e:
            messagebox.showerror("Load error", f"Failed to load {shapes_path}:\n{e}\nStarting empty.")
            self.shapes = []
        self.dirty = False
        self.thumbs = []  # keep PhotoImage refs alive

        # top bar
        bar = tk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        tk.Button(bar, text="Add shape", command=self.add_shape).pack(side=tk.LEFT)
        tk.Button(bar, text="Duplicate", command=self.duplicate_shape).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="Rename", command=self.rename_shape).pack(side=tk.LEFT)
        tk.Button(bar, text="Remove", command=self.remove_shape).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="Edit mesh…", command=self.edit_shape).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="Save", command=self.save, bg="#cfe8cf").pack(side=tk.RIGHT)
        tk.Button(bar, text="Background…", command=self.choose_bg).pack(side=tk.RIGHT, padx=4)

        self.path_label = tk.Label(self, text=f"file: {self.shapes_path}", anchor="w", fg="#555")
        self.path_label.pack(side=tk.TOP, fill=tk.X, padx=8)
        self.bg_label = tk.Label(self, text=self._bg_text(), anchor="w", fg="#555")
        self.bg_label.pack(side=tk.TOP, fill=tk.X, padx=8)

        # scrollable card list
        container = tk.Frame(self)
        container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.list_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        self.list_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.selected = None
        self.refresh_list()
        self._update_title()

    def _bg_text(self):
        p = SESSION_BG_PATH["path"]
        return f"background: {p}" if p else "background: <white default>"

    def _update_title(self):
        star = "*" if self.dirty else ""
        self.title(f"shapemesh{star} - LimpetVR shape editor")

    def refresh_list(self):
        for child in self.list_frame.winfo_children():
            child.destroy()
        self.thumbs = []
        if ImageTk is None:
            tk.Label(self.list_frame, text="PIL ImageTk unavailable - thumbnails disabled").pack()
            return
        if not self.shapes:
            tk.Label(self.list_frame, text="No shapes yet. Click 'Add shape'.",
                     fg="#666", pady=20).pack()
            return
        for idx, s in enumerate(self.shapes):
            card = tk.Frame(self.list_frame, relief=tk.RIDGE, borderwidth=1)
            card.pack(fill=tk.X, padx=2, pady=3)
            thumb_pil = render_thumbnail_pil(s["grid"], size=120)
            photo = ImageTk.PhotoImage(thumb_pil)
            self.thumbs.append(photo)  # keep ref
            lbl_img = tk.Label(card, image=photo)
            lbl_img.pack(side=tk.LEFT, padx=6, pady=6)
            lbl_img.bind("<Button-1>", lambda e, i=idx: self.select(i))
            lbl_img.bind("<Double-Button-1>", lambda e, i=idx: (self.select(i), self.edit_shape()))

            right = tk.Frame(card)
            right.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=6)
            md = max_displacement(s["grid"])
            nmoved = int((displacement(s["grid"]) > 1e-6).sum())
            sel = ">> " if idx == self.selected else ""
            tk.Label(right, text=f"{sel}{s['name']}", font=("TkDefaultFont", 11, "bold"),
                     anchor="w").pack(anchor="w")
            tk.Label(right, text=f"id: {s.get('id') or '<assigned on save>'}",
                     fg="#888", anchor="w").pack(anchor="w")
            tk.Label(right, text=f"max move: {md:.3f}   moved pts: {nmoved}/81",
                     fg="#555", anchor="w").pack(anchor="w")
            btns = tk.Frame(right)
            btns.pack(anchor="w", pady=4)
            tk.Button(btns, text="Select", command=lambda i=idx: self.select(i)).pack(side=tk.LEFT)
            tk.Button(btns, text="Edit…", command=lambda i=idx: (self.select(i), self.edit_shape())).pack(side=tk.LEFT, padx=4)
            tk.Button(btns, text="Remove", command=lambda i=idx: (self.select(i), self.remove_shape())).pack(side=tk.LEFT, padx=4)
            # order arrows
            tk.Button(btns, text="↑", width=2,
                      state=tk.NORMAL if idx > 0 else tk.DISABLED,
                      command=lambda i=idx: self.move_shape(i, -1)).pack(side=tk.LEFT, padx=(8, 1))
            tk.Button(btns, text="↓", width=2,
                      state=tk.NORMAL if idx < len(self.shapes) - 1 else tk.DISABLED,
                      command=lambda i=idx: self.move_shape(i, 1)).pack(side=tk.LEFT)

    def select(self, idx):
        self.selected = idx
        self.refresh_list()

    def move_shape(self, idx, delta):
        """Move shape at idx by delta (-1 up, +1 down)."""
        n = len(self.shapes)
        j = idx + delta
        if not (0 <= idx < n and 0 <= j < n):
            return
        self.shapes[idx], self.shapes[j] = self.shapes[j], self.shapes[idx]
        # keep selection on the moved shape
        if self.selected == idx:
            self.selected = j
        elif self.selected == j:
            self.selected = idx
        self.dirty = True
        self._update_title()
        self.refresh_list()

    def _current(self):
        if self.selected is None or not (0 <= self.selected < len(self.shapes)):
            messagebox.showinfo("No selection", "Select a shape first.")
            return None
        return self.shapes[self.selected]

    # -- actions ---------------------------------------------------------
    def add_shape(self):
        name = simpledialog.askstring("Add shape", "Name for new shape:", parent=self)
        if not name:
            return
        if any(s["name"] == name for s in self.shapes):
            messagebox.showerror("Add shape", f"A shape named '{name}' already exists.")
            return
        self.shapes.append({"id": new_shape_id(self.shapes), "name": name, "grid": identity_grid()})
        self.selected = len(self.shapes) - 1
        self.dirty = True
        self._update_title()
        self.refresh_list()

    def duplicate_shape(self):
        s = self._current()
        if s is None:
            return
        base = s["name"] + "_copy"
        name = base
        k = 2
        while any(x["name"] == name for x in self.shapes):
            name = f"{base}{k}"
            k += 1
        # duplicate gets a fresh id even though the mesh starts as a copy
        self.shapes.append({"id": new_shape_id(self.shapes), "name": name, "grid": s["grid"].copy()})
        self.selected = len(self.shapes) - 1
        self.dirty = True
        self._update_title()
        self.refresh_list()

    def rename_shape(self):
        s = self._current()
        if s is None:
            return
        name = simpledialog.askstring("Rename shape", "New name:", initialvalue=s["name"], parent=self)
        if not name or name == s["name"]:
            return
        if any(x["name"] == name for x in self.shapes):
            messagebox.showerror("Rename", f"A shape named '{name}' already exists.")
            return
        s["name"] = name
        self.dirty = True
        self._update_title()
        self.refresh_list()

    def remove_shape(self):
        s = self._current()
        if s is None:
            return
        if not messagebox.askyesno("Remove", f"Remove shape '{s['name']}'?"):
            return
        self.shapes.pop(self.selected)
        self.selected = None
        self.dirty = True
        self._update_title()
        self.refresh_list()

    def choose_bg(self):
        path = filedialog.askopenfilename(
            title="Select background image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")])
        if path:
            try:
                load_session_bg(path)
            except Exception as e:
                messagebox.showerror("Background", f"Failed to load image:\n{e}")
                return
        self.bg_label.config(text=self._bg_text())

    def edit_shape(self):
        s = self._current()
        if s is None:
            return
        editor = MeshEditor(self, s["name"], s["grid"])
        self.wait_window(editor)
        if editor.applied:
            s["grid"] = editor.result_grid
            self.dirty = True
            self._update_title()
            self.refresh_list()

    def save(self):
        try:
            save_shapes(self.shapes, self.shapes_path)
        except Exception as e:
            messagebox.showerror("Save", f"Failed to save:\n{e}")
            return
        self.dirty = False
        self._update_title()
        messagebox.showinfo("Save", f"Saved {len(self.shapes)} shape(s) to\n{self.shapes_path}")


# ----------------------------------------------------------------------------
# GUI: mesh deformer window
# ----------------------------------------------------------------------------

class MeshEditor(tk.Toplevel):
    CANVAS_SIZE = 620
    HIT_RADIUS = 12

    def __init__(self, parent, name, grid):
        super().__init__(parent)
        self.title(f"Edit mesh - {name}")
        self.geometry("940x780")
        self.transient(parent)
        self.grab_set()
        self.grid = np.asarray(grid, dtype=np.float64).reshape(N, N, 2).copy()
        self.result_grid = self.grid.copy()
        self.applied = False
        # multi-selection: set of (i, j); anchor is the point being dragged
        self.selection = set()
        self.anchor = None  # (i, j) drag anchor within selection
        self.dragging = False       # dragging selected point(s)
        self.marquee = False        # rubber-band box selection in progress
        self._drag_start_uv = None
        self._drag_orig = {}
        self._marquee_start = None  # (x, y) canvas px
        self._marquee_rect = None   # canvas item id
        self._marquee_additive = False
        self._undo_grid = None      # single-level undo snapshot
        self.show_grid = tk.BooleanVar(value=True)
        self.show_points = tk.BooleanVar(value=True)
        self.preview_warp = tk.BooleanVar(value=True)
        self.snap_enabled = tk.BooleanVar(value=False)
        self.snap_div = tk.IntVar(value=16)
        self.show_snap = tk.BooleanVar(value=True)
        self._warp_after_id = None
        self._bg_photo = None
        self._warped_cache = None

        # top toolbar (undo / background / snap) - keeps the side panel short
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 0))
        tk.Label(top, text=name, font=("TkDefaultFont", 11, "bold")).pack(side=tk.LEFT)
        self.undo_btn = tk.Button(top, text="Undo", command=self.do_undo, state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT, padx=8)
        tk.Label(top, text="BG:").pack(side=tk.LEFT)
        tk.Button(top, text="Load…", command=self.load_bg).pack(side=tk.LEFT)
        tk.Button(top, text="White", command=self.white_bg).pack(side=tk.LEFT, padx=2)
        tk.Checkbutton(top, text="Warp", variable=self.preview_warp,
                       command=self.redraw).pack(side=tk.LEFT)
        self.bg_path_label = tk.Label(top, text=self._bg_short(), fg="#555")
        self.bg_path_label.pack(side=tk.LEFT, padx=4)
        tk.Label(top, text="Snap:").pack(side=tk.LEFT, padx=(10, 0))
        tk.Checkbutton(top, text="On", variable=self.snap_enabled).pack(side=tk.LEFT)
        tk.Spinbox(top, from_=2, to=90, width=4, textvariable=self.snap_div,
                   command=self.redraw).pack(side=tk.LEFT)
        tk.Label(top, text="div").pack(side=tk.LEFT)
        tk.Checkbutton(top, text="Show", variable=self.show_snap,
                       command=self.redraw).pack(side=tk.LEFT)
        tk.Button(top, text="Snap sel", command=self.do_snap_selection).pack(side=tk.LEFT, padx=2)

        # bottom bar (status + OK/Cancel)
        bottom = tk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)
        self.status = tk.Label(bottom, text="", anchor="w", fg="#333")
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(bottom, text="OK (apply)", command=self.on_ok, bg="#cfe8cf", width=10).pack(side=tk.RIGHT)
        tk.Button(bottom, text="Cancel", command=self.on_cancel, width=10).pack(side=tk.RIGHT, padx=6)

        main = tk.Frame(self)
        main.pack(fill=tk.BOTH, expand=True)

        # canvas (left) + hint underneath
        leftcol = tk.Frame(main)
        leftcol.pack(side=tk.LEFT, padx=8, pady=4)
        self.canvas = tk.Canvas(leftcol, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
                                bg="white", highlightthickness=1, highlightbackground="#888")
        self.canvas.pack(side=tk.TOP)
        tk.Label(leftcol, wraplength=600, justify="left", fg="#555",
                 text="click: select · Shift/Ctrl-click: add · drag empty: box-select · "
                      "drag point: move selection · right-click: reset · Ctrl+Z: undo").pack(
            side=tk.TOP, anchor="w", pady=(2, 0))
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<Shift-Button-1>", self.on_press)
        self.canvas.bind("<Control-Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.bind("<Left>", lambda e: self.nudge(-1, 0))
        self.bind("<Right>", lambda e: self.nudge(1, 0))
        self.bind("<Up>", lambda e: self.nudge(0, -1))
        self.bind("<Down>", lambda e: self.nudge(0, 1))
        self.bind("<Shift-Left>", lambda e: self.nudge(-5, 0))
        self.bind("<Shift-Right>", lambda e: self.nudge(5, 0))
        self.bind("<Shift-Up>", lambda e: self.nudge(0, -5))
        self.bind("<Shift-Down>", lambda e: self.nudge(0, 5))
        self.bind("<Escape>", lambda e: self.clear_selection())
        self.bind("<Control-a>", lambda e: self.select_all())
        self.bind("<Control-A>", lambda e: self.select_all())
        self.bind("<Control-z>", lambda e: self.do_undo())
        self.bind("<Control-Z>", lambda e: self.do_undo())

        # side panel (right) - only mesh ops; toolbar above holds the rest
        side = tk.Frame(main)
        side.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)

        self.sel_label = tk.Label(side, text="0 selected", fg="#555")
        self.sel_label.pack(anchor="w")
        srow = tk.Frame(side)
        srow.pack(anchor="w", pady=(0, 4))
        tk.Button(srow, text="Select all", command=self.select_all).pack(side=tk.LEFT)
        tk.Button(srow, text="Clear", command=self.clear_selection).pack(side=tk.LEFT, padx=4)

        # mirror helpers
        mir = tk.LabelFrame(side, text="Mirror (left / top primary)")
        mir.pack(fill=tk.X, pady=4)
        tk.Button(mir, text="Mirror L→R  (mirror X)", command=self.do_mirror_lr_flip).pack(
            fill=tk.X, padx=4, pady=2)
        tk.Button(mir, text="Mirror L→R  (copy offsets)", command=self.do_mirror_lr_copy).pack(
            fill=tk.X, padx=4, pady=2)
        tk.Button(mir, text="Mirror T→B  (mirror Y)", command=self.do_mirror_tb_flip).pack(
            fill=tk.X, padx=4, pady=2)

        # smoothing (+ row/col select sharing the same spinboxes)
        smo = tk.LabelFrame(side, text="Smooth")
        smo.pack(fill=tk.X, pady=4)
        rowf = tk.Frame(smo)
        rowf.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(rowf, text="Row:").pack(side=tk.LEFT)
        self.row_var = tk.IntVar(value=4)
        tk.Spinbox(rowf, from_=0, to=8, width=3, textvariable=self.row_var).pack(side=tk.LEFT, padx=2)
        tk.Button(rowf, text="Sel", command=self.do_select_row).pack(side=tk.LEFT)
        tk.Button(rowf, text="Smooth", command=self.do_smooth_row).pack(side=tk.LEFT, padx=2)
        colf = tk.Frame(smo)
        colf.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(colf, text="Col:").pack(side=tk.LEFT)
        self.col_var = tk.IntVar(value=4)
        tk.Spinbox(colf, from_=0, to=8, width=3, textvariable=self.col_var).pack(side=tk.LEFT, padx=2)
        tk.Button(colf, text="Sel", command=self.do_select_col).pack(side=tk.LEFT)
        tk.Button(colf, text="Smooth", command=self.do_smooth_col).pack(side=tk.LEFT, padx=2)
        tk.Button(smo, text="Smooth all", command=self.do_smooth_all).pack(padx=4, pady=2)

        # view / reset
        view = tk.LabelFrame(side, text="View / reset")
        view.pack(fill=tk.X, pady=4)
        tk.Checkbutton(view, text="Show grid lines", variable=self.show_grid,
                       command=self.redraw).pack(anchor="w", padx=4)
        tk.Checkbutton(view, text="Show points", variable=self.show_points,
                       command=self.redraw).pack(anchor="w", padx=4)
        rrow = tk.Frame(view)
        rrow.pack(padx=4, pady=4)
        tk.Button(rrow, text="Reset all", command=self.do_reset).pack(side=tk.LEFT)
        tk.Button(rrow, text="Reset sel", command=self.do_reset_sel).pack(side=tk.LEFT, padx=4)

        self.snap_div.trace_add("write", lambda *a: self.redraw())
        self._refresh_undo()
        self.redraw()

    # -- coordinate helpers --------------------------------------------------
    def uv_to_xy(self, u, v):
        s = self.CANVAS_SIZE
        return float(u) * s, float(v) * s

    def xy_to_uv(self, x, y):
        s = self.CANVAS_SIZE
        return x / s, y / s

    def find_point(self, x, y):
        best, bestd = None, self.HIT_RADIUS
        for i in range(N):
            for j in range(N):
                px, py = self.uv_to_xy(self.grid[i, j, 0], self.grid[i, j, 1])
                d = math.hypot(px - x, py - y)
                if d <= bestd:
                    bestd, best = d, (i, j)
        return best

    # -- background -----------------------------------------------------------
    def _bg_short(self):
        p = SESSION_BG_PATH["path"]
        if not p:
            return "<white default>"
        return p if len(p) < 40 else "…" + p[-39:]

    def load_bg(self):
        path = filedialog.askopenfilename(
            title="Select background image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")])
        if path:
            try:
                load_session_bg(path)
            except Exception as e:
                messagebox.showerror("Background", f"Failed to load image:\n{e}", parent=self)
                return
            self.bg_path_label.config(text=self._bg_short())
            self._warped_cache = None
            self.redraw()

    def white_bg(self):
        clear_session_bg()
        self.bg_path_label.config(text=self._bg_short())
        self._warped_cache = None
        self.redraw()

    def get_warped_bg_photo(self):
        """Warped session background as ImageTk photo (cached per grid state)."""
        if ImageTk is None:
            return None
        s = self.CANVAS_SIZE
        base = get_session_bg((s, s))
        if self.preview_warp.get():
            try:
                warped = warp_image(base, self.grid)
            except Exception:
                warped = base
        else:
            warped = base
        self._bg_photo = ImageTk.PhotoImage(warped)
        return self._bg_photo

    @property
    def selected_pt(self):
        """Backwards-compatible alias: the drag anchor (or None)."""
        return self.anchor

    @selected_pt.setter
    def selected_pt(self, value):
        self.anchor = value
        if value is None:
            self.selection = set()
        else:
            self.selection = {tuple(value)}

    # -- drawing ----------------------------------------------------------------
    def redraw(self):
        try:
            self.canvas.delete("all")
        except Exception:
            return
        s = self.CANVAS_SIZE
        # background
        photo = self.get_warped_bg_photo()
        if photo is not None:
            self.canvas.create_image(0, 0, anchor="nw", image=photo)
        # snap lattice overlay
        if self.show_snap.get() and self.snap_enabled.get():
            try:
                div = int(self.snap_div.get())
            except Exception:
                div = 0
            if div >= 2:
                step = s / div
                k = 1
                while k < div:
                    p = k * step
                    self.canvas.create_line(p, 0, p, s, fill="#9ecfff", width=1)
                    self.canvas.create_line(0, p, s, p, fill="#9ecfff", width=1)
                    k += 1
        # grid lines
        if self.show_grid.get():
            for i in range(N):
                pts = [self.uv_to_xy(*self.grid[i, j]) for j in range(N)]
                flat = [c for p in pts for c in p]
                self.canvas.create_line(*flat, fill="#2a2a2a", width=1)
            for j in range(N):
                pts = [self.uv_to_xy(*self.grid[i, j]) for i in range(N)]
                flat = [c for p in pts for c in p]
                self.canvas.create_line(*flat, fill="#2a2a2a", width=1)
        # points
        if self.show_points.get():
            mag = displacement(self.grid)
            mmax = float(mag.max()) if mag.size else 0.0
            for i in range(N):
                for j in range(N):
                    x, y = self.uv_to_xy(*self.grid[i, j])
                    m = float(mag[i, j])
                    sel = (i, j) in self.selection
                    is_anchor = self.anchor == (i, j)
                    if m < 1e-6 and not sel:
                        r, fill, outline = 3, "#bbbbbb", "#666666"
                    elif sel:
                        # selected: blue (keeps selection distinct; intensity hints at displacement)
                        t = (m / mmax) if mmax > 1e-9 else 1.0
                        t = max(0.2, min(1.0, t)) if m > 1e-6 else 0.2
                        fill = f"#{int(90 + 30 * (1-t)):02x}{int(150 + 40 * (1-t)):02x}{255:02x}"
                        r = 3 + 4 * t
                        outline = "#0033aa" if is_anchor else "#0055dd"
                    else:
                        t = (m / mmax) if mmax > 1e-9 else 1.0
                        t = max(0.2, min(1.0, t))
                        inward = is_inward_move(self.grid, i, j)
                        if inward:
                            # light green -> strong green
                            fill = f"#{int(160 - 130 * t):02x}{int(220 - 40 * t):02x}{int(160 - 130 * t):02x}"
                            outline = "#1a6b1a"
                        else:
                            # light red/pink -> strong red
                            fill = f"#{255:02x}{int(200 - 170 * t):02x}{int(60 - 40 * t):02x}"
                            outline = "#780000"
                        r = 3 + 4 * t
                    self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                            fill=fill, outline=outline,
                                            width=3 if is_anchor else (2 if sel else 1),
                                            tags=(f"pt_{i}_{j}",))
        self.update_status()

    def update_status(self):
        try:
            self.sel_label.config(text=f"{len(self.selection)} selected")
        except Exception:
            pass
        if not self.selection:
            self.status.config(text="Nothing selected. Click a point, Shift/Ctrl-click to add, drag empty area for box-select.")
            return
        if self.anchor is not None:
            i, j = self.anchor
            u, v = self.grid[i, j]
            iu, iv = j / (N - 1), i / (N - 1)
            extra = f"  +{len(self.selection) - 1} more" if len(self.selection) > 1 else ""
            self.status.config(
                text=f"anchor row={i} col={j}  uv=({u:.4f},{v:.4f})  offset=({u - iu:+.4f},{v - iv:+.4f}){extra}")
        else:
            self.status.config(text=f"{len(self.selection)} points selected.")

    def schedule_redraw(self):
        if self._warp_after_id is not None:
            try:
                self.after_cancel(self._warp_after_id)
            except Exception:
                pass
        self._warp_after_id = self.after(30, self._do_scheduled_redraw)

    def _do_scheduled_redraw(self):
        self._warp_after_id = None
        self.redraw()

    # -- selection helpers ----------------------------------------------------------
    def select_all(self):
        self.selection = {(i, j) for i in range(N) for j in range(N)}
        self.anchor = (4, 4)
        self.redraw()

    def clear_selection(self):
        self.selection = set()
        self.anchor = None
        self.redraw()

    def do_select_row(self):
        try:
            i = int(self.row_var.get())
        except Exception:
            return
        if 0 <= i < N:
            self.selection = {(i, j) for j in range(N)}
            self.anchor = (i, 4)
            self.redraw()

    def do_select_col(self):
        try:
            j = int(self.col_var.get())
        except Exception:
            return
        if 0 <= j < N:
            self.selection = {(i, j) for i in range(N)}
            self.anchor = (4, j)
            self.redraw()

    def _active_snap_div(self):
        if not self.snap_enabled.get():
            return None
        try:
            div = int(self.snap_div.get())
        except Exception:
            return None
        return div if div >= 2 else None

    def do_snap_selection(self):
        try:
            div = int(self.snap_div.get())
        except Exception:
            return
        if div < 2:
            return
        self._push_undo()
        pts = self.selection if self.selection else [(i, j) for i in range(N) for j in range(N)]
        snap_points(self.grid, div, pts)
        self.redraw()

    # -- undo (single level) ----------------------------------------------------------
    def _push_undo(self):
        self._undo_grid = self.grid.copy()
        self._refresh_undo()

    def _refresh_undo(self):
        try:
            self.undo_btn.config(state=tk.NORMAL if self._undo_grid is not None else tk.DISABLED)
        except Exception:
            pass

    def do_undo(self):
        if self._undo_grid is None:
            return
        self.grid = self._undo_grid
        self._undo_grid = None
        self._refresh_undo()
        self.redraw()

    # -- mouse --------------------------------------------------------------------
    @staticmethod
    def _is_additive(event):
        # Shift (0x0001) or Control (0x0004) held -> toggle/add
        try:
            return bool(event.state & 0x0001) or bool(event.state & 0x0004)
        except Exception:
            return False

    def on_press(self, event):
        hit = self.find_point(event.x, event.y)
        additive = self._is_additive(event)
        if hit is not None:
            if additive:
                if hit in self.selection:
                    self.selection.discard(hit)
                    if self.anchor == hit:
                        self.anchor = next(iter(self.selection), None)
                else:
                    self.selection.add(hit)
                    self.anchor = hit
                self.dragging = False
                self.redraw()
            else:
                if hit not in self.selection:
                    self.selection = {hit}
                self.anchor = hit
                # begin group drag: snapshot for undo, remember start + originals
                self._push_undo()
                self.dragging = True
                self._drag_start_uv = self.xy_to_uv(event.x, event.y)
                self._drag_orig = {pt: (float(self.grid[pt[0], pt[1], 0]),
                                        float(self.grid[pt[0], pt[1], 1]))
                                   for pt in self.selection}
                self.redraw()
        else:
            # empty area -> start marquee box-select
            self.marquee = True
            self._marquee_start = (event.x, event.y)
            if self._marquee_rect is not None:
                try:
                    self.canvas.delete(self._marquee_rect)
                except Exception:
                    pass
                self._marquee_rect = None
            if not additive:
                self.selection = set()
                self.anchor = None
                self.redraw()
            self._marquee_additive = additive

    def on_drag(self, event):
        if self.marquee and self._marquee_start is not None:
            x0, y0 = self._marquee_start
            if self._marquee_rect is not None:
                try:
                    self.canvas.delete(self._marquee_rect)
                except Exception:
                    pass
            self._marquee_rect = self.canvas.create_rectangle(
                x0, y0, event.x, event.y, outline="#0066ff", dash=(4, 3))
            return
        if not self.dragging or self.anchor is None or not self.selection:
            return
        u, v = self.xy_to_uv(event.x, event.y)
        su, sv = self._drag_start_uv
        du, dv = u - su, v - sv
        snap_div = self._active_snap_div()
        if snap_div is not None and self.anchor in self._drag_orig:
            # snap the shared delta via the anchor so the group stays rigid
            oau, oav = self._drag_orig[self.anchor]
            tu, tv = clamp_uv(oau + du, oav + dv)
            tu, tv = snap_value(tu, snap_div), snap_value(tv, snap_div)
            du, dv = tu - oau, tv - oav
            snap_div = None
        for pt in self.selection:
            ou, ov = self._drag_orig[pt]
            nu, nv = clamp_uv(ou + du, ov + dv)
            if snap_div is not None:
                nu, nv = snap_value(nu, snap_div), snap_value(nv, snap_div)
            self.grid[pt[0], pt[1], 0] = nu
            self.grid[pt[0], pt[1], 1] = nv
        self.schedule_redraw()
        self.update_status()

    def on_release(self, event):
        if self.marquee and self._marquee_start is not None:
            x0, y0 = self._marquee_start
            x1, y1 = event.x, event.y
            xa, xb = sorted((x0, x1))
            ya, yb = sorted((y0, y1))
            descent = 4
            boxed = set()
            if (xb - xa) > descent or (yb - ya) > descent:
                for i in range(N):
                    for j in range(N):
                        px, py = self.uv_to_xy(self.grid[i, j, 0], self.grid[i, j, 1])
                        if xa <= px <= xb and ya <= py <= yb:
                            boxed.add((i, j))
                if getattr(self, "_marquee_additive", False):
                    self.selection |= boxed
                else:
                    self.selection = boxed
                self.anchor = next(iter(boxed), self.anchor) if boxed else self.anchor
            else:
                # plain click on empty space clears (already cleared on press)
                pass
            if self._marquee_rect is not None:
                try:
                    self.canvas.delete(self._marquee_rect)
                except Exception:
                    pass
                self._marquee_rect = None
            self.marquee = False
            self._marquee_start = None
            self.redraw()
            return
        self.dragging = False
        self._drag_orig = {}
        # a bare click (no movement) pushed an undo snapshot of identical
        # state - drop it so it doesn't swallow the real undo slot
        try:
            if self._undo_grid is not None and np.array_equal(self.grid, self._undo_grid):
                self._undo_grid = None
                self._refresh_undo()
        except Exception:
            pass

    def on_right_click(self, event):
        hit = self.find_point(event.x, event.y)
        if hit is None:
            return
        self._push_undo()
        pts = self.selection if hit in self.selection and self.selection else {hit}
        for (i, j) in pts:
            self.grid[i, j, 0] = j / (N - 1)
            self.grid[i, j, 1] = i / (N - 1)
        self.selection = {hit}
        self.anchor = hit
        self.redraw()

    def nudge(self, dx_px, dy_px):
        if not self.selection:
            return
        s = self.CANVAS_SIZE
        self._push_undo()
        move_points_by_delta(self.grid, self.selection, dx_px / s, dy_px / s,
                             snap_div=self._active_snap_div(), anchor=self.anchor)
        self.redraw()

    # -- panel actions ---------------------------------------------------------------
    def do_mirror_lr_flip(self):
        self._push_undo()
        mirror_lr_flip(self.grid)
        self.redraw()

    def do_mirror_lr_copy(self):
        self._push_undo()
        mirror_lr_copy(self.grid)
        self.redraw()

    def do_mirror_tb_flip(self):
        self._push_undo()
        mirror_tb_flip(self.grid)
        self.redraw()

    def do_smooth_row(self):
        try:
            i = int(self.row_var.get())
        except Exception:
            return
        if 0 <= i < N:
            self._push_undo()
            smooth_row(self.grid, i)
            self.redraw()

    def do_smooth_col(self):
        try:
            j = int(self.col_var.get())
        except Exception:
            return
        if 0 <= j < N:
            self._push_undo()
            smooth_col(self.grid, j)
            self.redraw()

    def do_smooth_all(self):
        self._push_undo()
        smooth_all(self.grid)
        self.redraw()

    def do_reset(self):
        if messagebox.askyesno("Reset", "Reset all 81 points to identity?", parent=self):
            self._push_undo()
            self.grid = identity_grid()
            self.redraw()

    def do_reset_sel(self):
        if not self.selection:
            return
        self._push_undo()
        for (i, j) in self.selection:
            self.grid[i, j, 0] = j / (N - 1)
            self.grid[i, j, 1] = i / (N - 1)
        self.redraw()

    def on_ok(self):
        self.result_grid = self.grid.copy()
        self.applied = True
        self.destroy()

    def on_cancel(self):
        self.applied = False
        self.destroy()


def main(argv=None):
    path = SHAPES_PATH
    if argv and len(argv) > 1:
        path = argv[1]
    app = ShapeManagerApp(shapes_path=path)
    app.mainloop()


if __name__ == "__main__":
    main(sys.argv)
