"""Reannotate 3D OBB (axis-aligned in mesh-local frame) for all assets.

For each asset under assets/objects/:
  - Rigid (flat) layout (has collision/baseN.glb): compute the AABB of the
    collision mesh vertices (in unscaled GLB-native frame) and write the
    `center` and `extents` fields into model_dataN.json. The original values
    are backed up into `_orig_center` and `_orig_extents` on the first run.
  - Articulated (URDF) layout (has subdirs with mobility.urdf and
    bounding_box.json): copy `min`/`max` from bounding_box.json into
    model_data.json as `center`/`extents` for static-time use; runtime
    SAPIEN loading already reads bounding_box.json so this just keeps the
    JSON self-contained.

This matches the convention used by Base_Task.save_obb2d() (envs/_base_task.py), which
multiplies the stored values by `scale` and projects the 8 corners into each camera
while recording data.

Usage:
  python script/reannotate_obb.py           # process all
  python script/reannotate_obb.py 001_bottle 020_hammer  # subset
  python script/reannotate_obb.py --dry-run # report changes only
  python script/reannotate_obb.py --report-only  # CSV of stored vs actual
"""
import argparse
import json
import sys
import csv
from pathlib import Path
import numpy as np
import trimesh

ASSETS_ROOT = Path(__file__).resolve().parents[1] / "assets" / "objects"


def load_mesh_aabb(glb_path: Path):
    """Return (center, extents) of the AABB of GLB vertices in the scene frame."""
    scene = trimesh.load(str(glb_path), force="scene")
    try:
        mesh = trimesh.util.concatenate(scene.dump())
        verts = np.asarray(mesh.vertices)
    except Exception:
        verts = np.concatenate([np.asarray(g.vertices) for g in scene.geometry.values()])
    if verts.size == 0:
        return None, None
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return ((lo + hi) / 2.0).tolist(), (hi - lo).tolist()


def find_glb_or_obj(folder: Path, idx):
    """Resolve baseN.glb or textured.obj inside `folder`."""
    suffix = "" if idx is None else str(idx)
    for fn in [f"base{suffix}.glb", f"textured{suffix}.obj"]:
        p = folder / fn
        if p.exists():
            return p
    return None


def is_articulated(asset_dir: Path) -> bool:
    return any(
        (sub / "mobility.urdf").exists()
        for sub in asset_dir.iterdir()
        if sub.is_dir() and sub.name != "visual"
    )


def list_rigid_variants(asset_dir: Path):
    """Yield (json_path, mesh_path) for each rigid variant."""
    for jf in sorted(asset_dir.glob("model_data*.json")):
        # extract index from "model_data{idx}.json"
        stem = jf.stem.replace("model_data", "")
        idx = int(stem) if stem.isdigit() else None
        mesh = None
        for folder in [asset_dir / "collision", asset_dir]:
            if folder.exists():
                mesh = find_glb_or_obj(folder, idx)
                if mesh is not None and mesh.exists():
                    break
        # collision missing → fallback to visual
        if mesh is None:
            for folder in [asset_dir / "visual"]:
                if folder.exists():
                    mesh = find_glb_or_obj(folder, idx)
                    if mesh is not None and mesh.exists():
                        break
        yield jf, mesh


def list_articulated_variants(asset_dir: Path):
    """Yield (json_path, bbox_json_path) for each articulated variant."""
    for sub in sorted([d for d in asset_dir.iterdir() if d.is_dir() and d.name != "visual"]):
        jf = sub / "model_data.json"
        bb = sub / "bounding_box.json"
        if jf.exists():
            yield jf, bb if bb.exists() else None


def maybe_backup(md: dict):
    """Save originals once for traceability."""
    if "_orig_extents" not in md and "extents" in md:
        md["_orig_extents"] = md["extents"]
    if "_orig_center" not in md and "center" in md:
        md["_orig_center"] = md["center"]


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=4))


def process_asset(asset_dir: Path, dry_run: bool, report_rows: list, verbose: bool):
    name = asset_dir.name
    articulated = is_articulated(asset_dir)
    updated, skipped, errored = 0, 0, 0
    if articulated:
        for jf, bb in list_articulated_variants(asset_dir):
            try:
                md = json.loads(jf.read_text())
            except Exception as e:
                print(f"  [ERR] {jf}: {e}")
                errored += 1
                continue
            if bb is None:
                if verbose:
                    print(f"  [skip] {jf.relative_to(ASSETS_ROOT)}: no bounding_box.json")
                skipped += 1
                continue
            try:
                bb_data = json.loads(bb.read_text())
                bmin, bmax = np.array(bb_data["min"]), np.array(bb_data["max"])
            except Exception as e:
                print(f"  [ERR] {bb}: {e}")
                errored += 1
                continue
            new_center = ((bmin + bmax) / 2.0).tolist()
            new_extents = (bmax - bmin).tolist()
            old_center = md.get("center")
            old_extents = md.get("extents")
            report_rows.append({
                "asset": name,
                "variant": jf.parent.name,
                "type": "articulated",
                "old_center": old_center,
                "old_extents": old_extents,
                "new_center": new_center,
                "new_extents": new_extents,
            })
            if not dry_run:
                maybe_backup(md)
                md["center"] = new_center
                md["extents"] = new_extents
                write_json(jf, md)
            updated += 1
    else:
        for jf, mesh in list_rigid_variants(asset_dir):
            try:
                md = json.loads(jf.read_text())
            except Exception as e:
                print(f"  [ERR] {jf}: {e}")
                errored += 1
                continue
            if mesh is None:
                if verbose:
                    print(f"  [skip] {jf.relative_to(ASSETS_ROOT)}: no mesh")
                skipped += 1
                continue
            try:
                new_center, new_extents = load_mesh_aabb(mesh)
            except Exception as e:
                print(f"  [ERR] {mesh}: {e}")
                errored += 1
                continue
            if new_center is None:
                skipped += 1
                continue
            old_center = md.get("center")
            old_extents = md.get("extents")
            report_rows.append({
                "asset": name,
                "variant": jf.stem.replace("model_data", ""),
                "type": "rigid",
                "old_center": old_center,
                "old_extents": old_extents,
                "new_center": new_center,
                "new_extents": new_extents,
            })
            if not dry_run:
                maybe_backup(md)
                md["center"] = new_center
                md["extents"] = new_extents
                write_json(jf, md)
            updated += 1
    return updated, skipped, errored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", nargs="*", help="Optional list of asset dir names to process (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Compute new values but do not write JSON")
    parser.add_argument("--report-only", action="store_true", help="Implies --dry-run, just save report CSV")
    parser.add_argument("--report", default="obb_reannotation_report.csv", help="CSV report path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.report_only:
        args.dry_run = True

    if args.assets:
        targets = [ASSETS_ROOT / a for a in args.assets]
    else:
        targets = sorted([d for d in ASSETS_ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")])

    rows = []
    totals = {"updated": 0, "skipped": 0, "errored": 0}
    for asset_dir in targets:
        if not asset_dir.exists():
            print(f"[warn] {asset_dir} does not exist")
            continue
        u, s, e = process_asset(asset_dir, args.dry_run, rows, args.verbose)
        totals["updated"] += u
        totals["skipped"] += s
        totals["errored"] += e
        print(f"{asset_dir.name}: updated={u} skipped={s} errored={e}")

    # CSV report
    if rows:
        with open(args.report, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["asset", "variant", "type",
                        "old_cx", "old_cy", "old_cz",
                        "old_ex", "old_ey", "old_ez",
                        "new_cx", "new_cy", "new_cz",
                        "new_ex", "new_ey", "new_ez",
                        "max_extent_diff"])
            for r in rows:
                oc = list(r["old_center"] or [None]*3) + [None]*3
                oe = list(r["old_extents"] or [None]*3) + [None]*3
                nc, ne = r["new_center"], r["new_extents"]
                try:
                    diff = max(abs((oe[i] or 0) - ne[i]) for i in range(3))
                except Exception:
                    diff = ""
                w.writerow([r["asset"], r["variant"], r["type"],
                            *oc[:3], *oe[:3], *nc, *ne, diff])
        print(f"\nReport: {args.report}  ({len(rows)} rows)")

    print(f"\nTOTAL: updated={totals['updated']} skipped={totals['skipped']} errored={totals['errored']}")


if __name__ == "__main__":
    main()
