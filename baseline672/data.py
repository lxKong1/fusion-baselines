"""Portable, episode-aligned offline672-v1 (Python 3.8, NumPy only)."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np


VERSION = "offline672-v1"
MANIFEST_SHA256 = "3ca8c899158b39b85d2db128d5dd3aab16bd764966c03f20fe3c2fe82f74ce3f"
RELEASE_SHA256 = "d7a9096d3b418628ef3a154159b4166c78de21458212a3f3a061681967d4dbf4"
N, T = 672, 141345
OUTCOMES = {"success": 556, "collision": 77, "off_road": 38, "timeout": 1}
SOURCE_COUNTS = {"human": 472, "policy": 200}
# A frame field has one extra observation per episode, not per dataset.
FIELDS = {
    "states": ("<f4", (35,), True), "lidars": ("<f4", (240,), True),
    "bevs": ("u1", (5, 84, 84), True), "actions": ("<f4", (2,), False),
    "rewards": ("<f4", (), False), "costs": ("<f4", (), False),
    "terminated": ("?", (), False), "truncated": ("?", (), False),
}
MEMBERS = {"manifest.json"} | {k + ".npy" for k in FIELDS}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _read(data_dir, full=False):
    root = Path(data_dir)
    _check(sha256(root / "manifest.json") == MANIFEST_SHA256, "not the sealed offline672-v1 manifest")
    meta = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    _check(meta["schema"] == VERSION and set(meta["files_sha256"]) == MEMBERS - {"manifest.json"},
           "dataset schema / file list mismatch")
    episodes = meta["episodes"]
    _check(len(episodes) == N and sum(e["length"] for e in episodes) == T and
           len({(e["scene"], e["seed"]) for e in episodes}) == N and
           Counter(e["outcome"] for e in episodes) == OUTCOMES and
           Counter(e["source"] for e in episodes) == SOURCE_COUNTS,
           "unexpected episode identities / outcome counts")
    arrays = {}
    for k, (dtype, dims, obs) in FIELDS.items():
        p = root / (k + ".npy")
        if full:
            _check(sha256(p) == meta["files_sha256"][p.name], "SHA256 mismatch: " + p.name)
        a = np.load(str(p), mmap_mode="r", allow_pickle=False)
        _check(a.dtype == np.dtype(dtype) and a.shape == ((T + (N if obs else 0)),) + dims,
               "shape/dtype mismatch: " + k)
        arrays[k] = a
    result = []
    pos = 0
    for i, ep in enumerate(episodes):
        n = ep["length"]
        _check(isinstance(n, int) and n > 0, "empty episode")
        row = dict(ep)
        for k, (_, _, obs) in FIELDS.items():
            start = pos + (i if obs else 0)
            row[k] = arrays[k][start:start + n + int(obs)]
        c = row["costs"]
        bad = ep["outcome"] in ("collision", "off_road")
        trunc = ep["outcome"] == "timeout"
        _check(not np.any(c[:-1]) and c[-1] == float(bad) and
               not np.any(row["terminated"][:-1] | row["truncated"][:-1]) and
               bool(row["terminated"][-1]) != trunc and bool(row["truncated"][-1]) == trunc,
               "invalid costs / terminal flags: %s %s" % (ep["scene"], ep["seed"]))
        result.append(row)
        pos += n
    return result


def load_episodes(data_dir="dataset/offline672-v1", safe_only=False):
    """List of dicts; arrays are read-only mmap views. Call verify once after download."""
    rows = _read(data_dir)
    return [r for r in rows if r["outcome"] == "success"] if safe_only else rows


def verify(data_dir):
    rows = _read(data_dir, full=True)
    summary = dict(episodes=len(rows), transitions=sum(r["length"] for r in rows),
                   outcomes=dict(Counter(r["outcome"] for r in rows)),
                   sources=dict(Counter(r["source"] for r in rows)),
                   positive_cost_steps=int(sum(r["costs"][-1] for r in rows)))
    print(json.dumps(summary, sort_keys=True))
    return summary


def _max_size(name):
    if name == "manifest.json":
        return 2 << 20
    dtype, dims, obs = FIELDS[name[:-4]]
    return (T + (N if obs else 0)) * int(np.prod(dims)) * np.dtype(dtype).itemsize + 65536


def unpack(archive, dest, expected):
    """No extractall: only create the fixed, flat list of numeric NPY files."""
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    _check(sha256(archive) == expected, "archive SHA256 mismatch")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        stage = Path(tmp) / VERSION
        stage.mkdir()
        with zipfile.ZipFile(archive) as z:
            _check(len(z.infolist()) == len(MEMBERS) and set(z.namelist()) == MEMBERS,
                   "unexpected/duplicate archive member")
            for info in z.infolist():
                _check(info.file_size <= _max_size(info.filename), "oversize archive member")
                with z.open(info) as src, (stage / info.filename).open("xb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
        verify(stage)
        stage.rename(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ver = sub.add_parser("verify")
    ver.add_argument("--data", default="dataset/" + VERSION)
    imp = sub.add_parser("unpack")
    imp.add_argument("--archive", required=True)
    imp.add_argument("--data", default="dataset/" + VERSION)
    imp.add_argument("--sha256", default=RELEASE_SHA256, help="expected archive digest (pinned release by default)")
    args = parser.parse_args()
    if args.command == "verify":
        verify(args.data)
    else:
        unpack(args.archive, args.data, args.sha256)


if __name__ == "__main__":
    main()
