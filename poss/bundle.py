"""Asset inventory, completeness check and adoption of browser downloads for the full data bundle.

Expected sizes are the sizes of the shared Drive files (from the Drive listing: names/sizes only), so a
truncated download is detected by size. Nothing here reads match content.
"""
import os
import shutil

import poss  # noqa: F401
from games import GAME, SOURCES, asset_paths

# (source, asset) -> expected byte size on Drive
EXPECTED = {
    ("tactical", "video"): 5222307334,
    ("tactical", "atd_raw"): 195714285,
    ("tactical", "atd_meta"): 59057,
    ("tactical", "smart_tagging"): 261055,
    ("tactical", "homography"): 23187043,
    ("broadcast", "video"): 13962360320,
    ("broadcast", "atd_raw"): 141966465,
    ("broadcast", "atd_meta"): 64956,
    ("broadcast", "homography"): 15401635,
}

# what the model uses each asset for (reported per asset)
USE = {
    ("tactical", "video"): "stoppage cues: pitch motion energy (v3), audio whistle events if an audio track exists; frame-count/fps alignment check",
    ("tactical", "atd_raw"): "core: players + ball tracking -> control evidence, stoppage cues",
    ("tactical", "atd_meta"): "team of each track slot (home/away)",
    ("tactical", "smart_tagging"): "soft evidence: set-piece windows, learned team sign per tag kind, restart priors",
    ("tactical", "homography"): "camera reliability per frame (valid/reprojection error); pitch mask for the video cue; tracking-vs-image consistency check",
    ("broadcast", "video"): "targeted ball-detection windows around tactical ball gaps / disputed turnovers / uncertain restarts (mapped by the tracking alignment); used only if verified on held-out ball positions",
    ("broadcast", "atd_raw"): "second ball/player track: aligned to the tactical timeline, fills tactical ball gaps on confidently matched intervals",
    ("broadcast", "atd_meta"): "team of each broadcast track slot",
    ("broadcast", "homography"): "broadcast validity per frame; cut/replay evidence for the alignment report",
}


def rows(data_root, downloads=None):
    """One dict per asset: path, present, size, expected, size_ok, where (data_root | downloads | missing)."""
    out = []
    for source in SOURCES:
        paths = asset_paths(GAME, data_root, source)
        for asset, path in paths.items():
            exp = EXPECTED.get((source, asset))
            where, size = "missing", None
            if os.path.isfile(path):
                where, size = "data_root", os.path.getsize(path)
            elif downloads and os.path.isfile(os.path.join(downloads, os.path.basename(path))):
                where, size = "downloads", os.path.getsize(os.path.join(downloads, os.path.basename(path)))
            out.append({"source": source, "asset": asset, "path": path, "where": where, "size": size,
                        "expected": exp, "size_ok": (size == exp) if (size is not None and exp) else False,
                        "use": USE.get((source, asset), "")})
    return out


def adopt_downloads(data_root, downloads, log=print):
    """Move complete browser downloads (exact name + exact size) into <data_root>/<source>/. Idempotent."""
    moved = 0
    for r in rows(data_root, downloads):
        if r["where"] != "downloads" or not r["size_ok"]:
            continue
        dst = r["path"]
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(os.path.join(downloads, os.path.basename(dst)), dst)
        log("adopted {} ({:.1f} MB)".format(os.path.basename(dst), r["size"] / 1e6))
        moved += 1
    return moved


def in_progress(downloads):
    """Chrome partial downloads still running (names only)."""
    if not downloads or not os.path.isdir(downloads):
        return []
    return [f for f in os.listdir(downloads) if f.endswith(".crdownload")]
