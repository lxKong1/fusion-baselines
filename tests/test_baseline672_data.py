"""Small synthetic checks: do not require the full dataset, simulator or network."""
import json
import zipfile

import numpy as np
import pytest

from baseline672 import data


def archive(root, destination):
    """Build only the tiny synthetic fixture used in these tests."""
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(data.MEMBERS):
            bundle.write(root / name, arcname=name)


@pytest.fixture
def fixture_data(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    monkeypatch.setattr(data, "T", 4)
    monkeypatch.setattr(data, "N", 3)
    monkeypatch.setattr(data, "OUTCOMES", {"success": 1, "collision": 1, "timeout": 1})
    monkeypatch.setattr(data, "SOURCE_COUNTS", {"human": 2, "policy": 1})
    rows = [dict(scene="O", seed=i, source=src, outcome=outcome, length=n)
            for i, (src, outcome, n) in enumerate([
                ("human", "success", 2), ("human", "collision", 1), ("policy", "timeout", 1)])]
    digests = {}
    for k, (dtype, dims, obs) in data.FIELDS.items():
        a = np.zeros((4 + (3 if obs else 0),) + dims, dtype=dtype)
        if k in ("states", "actions"):
            a[:, 0] = np.arange(len(a))
        if k == "terminated":
            a[[1, 2]] = True
        if k == "truncated":
            a[3] = True
        if k == "costs":
            a[2] = 1
        p = root / (k + ".npy")
        np.save(str(p), a, allow_pickle=False)
        digests[p.name] = data.sha256(p)
    m = root / "manifest.json"
    m.write_text(json.dumps(dict(schema=data.VERSION, episodes=rows, files_sha256=digests)))
    monkeypatch.setattr(data, "MANIFEST_SHA256", data.sha256(m))
    return root


def test_no_cross_episode_and_safe_filter(fixture_data):
    rows = data.load_episodes(fixture_data)
    assert [r["states"][:, 0].tolist() for r in rows] == [[0, 1, 2], [3, 4], [5, 6]]
    assert [r["actions"][:, 0].tolist() for r in rows] == [[0, 1], [2], [3]]
    assert isinstance(rows[1]["bevs"], np.memmap)
    assert not rows[1]["bevs"].flags.writeable
    assert [r["seed"] for r in data.load_episodes(fixture_data, safe_only=True)] == [0]
    assert data.verify(fixture_data)["positive_cost_steps"] == 1


def test_checksum_and_identity_tamper(fixture_data):
    path = fixture_data / "bevs.npy"
    with path.open("r+b") as f:
        f.seek(-1, 2)
        f.write(b"X")
    with pytest.raises(ValueError, match="SHA256"):
        data.verify(fixture_data)
    with (fixture_data / "manifest.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="sealed"):
        data.load_episodes(fixture_data)


def test_unpack_roundtrip_no_overwrite(fixture_data, tmp_path):
    dst, bundle = tmp_path / "fresh", tmp_path / "test.zip"
    archive(fixture_data, bundle)
    with pytest.raises(ValueError, match="checksum|SHA256"):
        data.unpack(bundle, dst, "invalid")
    assert not dst.exists()
    data.unpack(bundle, dst, data.sha256(bundle))
    for f in fixture_data.iterdir():
        assert data.sha256(dst / f.name) == data.sha256(f)
    with pytest.raises(FileExistsError):
        data.unpack(bundle, dst, data.sha256(bundle))
    with pytest.raises(FileExistsError):
        archive(fixture_data, bundle)


def test_unpack_unknown_member(tmp_path):
    bundle = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(bundle, "x") as z:
        z.writestr("../unexpected", "oops")
    with pytest.raises(ValueError, match="archive member"):
        data.unpack(bundle, tmp_path / "dst", data.sha256(bundle))
    assert not (tmp_path / "unexpected").exists()
    assert not (tmp_path / "dst").exists()
