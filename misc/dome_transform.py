#!/usr/bin/env python3
"""
dome_transform.py [input_file] [output_file]
Applies the fitted dome/eqarectangular warp using the 65x65 subdivided mesh.
"""
import sys
import os
import numpy as np
import cv2

# Fitted symmetry-constrained polynomial coefficients (from gridpoints3.npz)
FIT_COEFFS = {
    'dx': np.array([3.55137128, -58.89225498, 6.27069289, -6.71262652, 5.90288003]),
    'dy': np.array([2.09336732, 22.91471597, 2.31771394, -3.7321752, -5.61181181]),
}

def build_warp_maps(w, h, n=65):
    cx, cy = FIT_COEFFS['dx'], FIT_COEFFS['dy']
    us = np.linspace(-1, 1, n)
    vs = np.linspace(-1, 1, n)
    U, V = np.meshgrid(us, vs)
    U2, V2 = U*U, V*V
    U4, V4 = U2*U2, V2*V2
    dx = U * (cx[0] + cx[1]*V2 + cx[2]*V4 + cx[3]*U2 + cx[4]*U2*V2)
    dy = V * (1 - V2) * (cy[0] + cy[1]*U2 + cy[2]*U4 + cy[3]*V2 + cy[4]*U2*V2)
    dst_x = (50 + U*50 + dx) * w / 100.0
    dst_y = (50 + V*50 + dy) * h / 100.0
    src_x = (50 + U*50) * w / 100.0
    src_y = (50 + V*50) * h / 100.0
    dst_grid = np.stack([dst_x, dst_y], axis=-1).astype(np.float32)
    src_grid = np.stack([src_x, src_y], axis=-1).astype(np.float32)

    map_x = np.full((h, w), -1.0, dtype=np.float32)
    map_y = np.full((h, w), -1.0, dtype=np.float32)

    for cj in range(n-1):
        for ci in range(n-1):
            tl, tr = dst_grid[cj, ci], dst_grid[cj, ci+1]
            br, bl = dst_grid[cj+1, ci+1], dst_grid[cj+1, ci]
            stl, str_ = src_grid[cj, ci], src_grid[cj, ci+1]
            sbr, sbl = src_grid[cj+1, ci+1], src_grid[cj+1, ci]
            if np.linalg.norm(tr-tl) < 0.5 or np.linalg.norm(br-bl) < 0.5 or \
               np.linalg.norm(bl-tl) < 0.5 or np.linalg.norm(br-tr) < 0.5:
                continue
            for tri_dst, tri_src in [([tl,tr,br],[stl,str_,sbr]), ([tl,br,bl],[stl,sbr,sbl])]:
                tri_dst = np.array(tri_dst, dtype=np.float32)
                tri_src = np.array(tri_src, dtype=np.float32)
                area = 0.5*abs(tri_dst[0,0]*(tri_dst[1,1]-tri_dst[2,1])+tri_dst[1,0]*(tri_dst[2,1]-tri_dst[0,1])+tri_dst[2,0]*(tri_dst[0,1]-tri_dst[1,1]))
                if area < 0.5:
                    continue
                M = cv2.getAffineTransform(tri_dst, tri_src)
                xs, ys = tri_dst[:,0], tri_dst[:,1]
                x_min, x_max = max(0,int(np.floor(xs.min()))), min(w-1,int(np.ceil(xs.max())))
                y_min, y_max = max(0,int(np.floor(ys.min()))), min(h-1,int(np.ceil(ys.max())))
                if x_min >= x_max or y_min >= y_max:
                    continue
                px = np.arange(x_min, x_max+1, dtype=np.float32)
                py = np.arange(y_min, y_max+1, dtype=np.float32)
                PX, PY = np.meshgrid(px, py)
                v0=tri_dst[2]-tri_dst[0]; v1=tri_dst[1]-tri_dst[0]
                v2=np.stack([PX-tri_dst[0,0], PY-tri_dst[0,1]], axis=-1)
                dot00=np.dot(v0,v0); dot01=np.dot(v0,v1)
                dot02=v2[:,:,0]*v0[0]+v2[:,:,1]*v0[1]
                dot11=np.dot(v1,v1); dot12=v2[:,:,0]*v1[0]+v2[:,:,1]*v1[1]
                denom=dot00*dot11-dot01*dot01
                inv=np.zeros_like(denom); inv[np.abs(denom)>1e-10]=1/denom[np.abs(denom)>1e-10]
                u=(dot11*dot02-dot01*dot12)*inv; v=(dot00*dot12-dot01*dot02)*inv
                mask=(u>=-1e-6)&(v>=-1e-6)&(u+v<=1+1e-6)
                if not mask.any():
                    continue
                sx=M[0,0]*PX+M[0,1]*PY+M[0,2]; sy=M[1,0]*PX+M[1,1]*PY+M[1,2]
                map_x[y_min:y_max+1, x_min:x_max+1][mask]=sx[mask]
                map_y[y_min:y_max+1, x_min:x_max+1][mask]=sy[mask]
    mask=(map_x<0)|(map_y<0)
    if mask.any():
        map_x[mask]=0; map_y[mask]=0
    return map_x, map_y

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {os.path.basename(sys.argv[0])} [input_file] [output_file]", file=sys.stderr)
        sys.exit(1)
    inp, out = sys.argv[1], sys.argv[2]
    img = cv2.imread(inp)
    if img is None:
        print(f"Failed to read {inp}", file=sys.stderr)
        sys.exit(1)
    h, w = img.shape[:2]
    map_x, map_y = build_warp_maps(w, h, n=65)
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    ok = cv2.imwrite(out, warped)
    if not ok:
        print(f"Failed to write {out}", file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {out} ({w}x{h})")

if __name__ == "__main__":
    main()
