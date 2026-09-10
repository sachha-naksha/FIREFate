"""``DatasetPaths.find``: locate datasets.yaml from the kernel's cwd or an env var."""
import pytest

from focalfire.io import DatasetPaths

YAML = """
roots:
  base: /data/{name}
  name: bcell
scalars:
  DYNAMIC_H5: "{base}/dynamic.h5"
groups:
  PB:
    template: "{base}/enrichment_ep{i}_pb.csv"
    n_episodes: 2
"""


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "temporal").mkdir()
    (tmp_path / "temporal" / "datasets.yaml").write_text(YAML)
    (tmp_path / "temporal" / "trajectory" / "deep").mkdir(parents=True)
    return tmp_path


def test_from_yaml_expands_roots_scalars_and_groups(tree):
    cfg = DatasetPaths.from_yaml(tree / "temporal" / "datasets.yaml")
    assert cfg.DYNAMIC_H5 == "/data/bcell/dynamic.h5"
    assert cfg.PB == {"ep1": "/data/bcell/enrichment_ep1_pb.csv", "ep2": "/data/bcell/enrichment_ep2_pb.csv"}


def test_find_walks_up_from_the_cwd(tree, monkeypatch):
    monkeypatch.delenv("FOCALFIRE_DATASETS", raising=False)
    monkeypatch.chdir(tree / "temporal" / "trajectory" / "deep")
    cfg = DatasetPaths.find()
    assert cfg.source == str(tree / "temporal" / "datasets.yaml")
    assert cfg.DYNAMIC_H5 == "/data/bcell/dynamic.h5"


def test_find_accepts_an_explicit_start(tree, monkeypatch):
    monkeypatch.delenv("FOCALFIRE_DATASETS", raising=False)
    monkeypatch.chdir(tree)                      # above the yaml: cwd alone would not find it
    cfg = DatasetPaths.find(start=tree / "temporal" / "trajectory")
    assert cfg.source == str(tree / "temporal" / "datasets.yaml")


def test_env_var_wins_over_the_walk(tree, tmp_path, monkeypatch):
    other = tmp_path / "elsewhere.yaml"
    other.write_text(YAML.replace("bcell", "tcell"))
    monkeypatch.setenv("FOCALFIRE_DATASETS", str(other))
    monkeypatch.chdir(tree / "temporal" / "trajectory")
    cfg = DatasetPaths.find()
    assert cfg.source == str(other)
    assert cfg.DYNAMIC_H5 == "/data/tcell/dynamic.h5"


def test_find_reports_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.delenv("FOCALFIRE_DATASETS", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="FOCALFIRE_DATASETS"):
        DatasetPaths.find(filename="definitely_missing.yaml")
