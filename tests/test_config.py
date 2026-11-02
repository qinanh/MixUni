from pathlib import Path

import pytest

from mixuni.config import CONFIG_DIR, load_config, run_name

CONFIGS = sorted(p for p in CONFIG_DIR.rglob("*.yaml") if not p.name.startswith("_"))


@pytest.mark.parametrize("path", CONFIGS, ids=[run_name(p) for p in CONFIGS])
def test_every_config_loads(path):
    cfg = load_config(path)
    assert cfg["name"] == run_name(path)
    split = cfg["data"]["split"]
    assert split == "random" or (CONFIG_DIR.parent / split).exists()


def test_base_merge_overrides_only_given_keys():
    main, ablation = load_config(CONFIG_DIR / "mixuni.yaml"), load_config(CONFIG_DIR / "ablations/no_moe.yaml")
    assert ablation["model"]["shared_head"] == "per_property"
    assert ablation["model"]["physics_head"] == main["model"]["physics_head"]
    assert ablation["data"]["batch_size"] == 512 and ablation["data"]["split"] == main["data"]["split"]


def test_unknown_key_is_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"base: {CONFIG_DIR / 'mixuni.yaml'}\nmodel:\n  hiden_dim: 32\n")
    with pytest.raises(ValueError, match="unknown model keys"):
        load_config(bad)
