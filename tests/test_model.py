import pytest

torch = pytest.importorskip("torch")

from mixuni.config import CONFIG_DIR, load_config  # noqa: E402
from mixuni.data import collate  # noqa: E402
from mixuni.model import build_model  # noqa: E402
from mixuni.train.engine import compute_loss  # noqa: E402
from mixuni.train.losses import build_weighting  # noqa: E402

from conftest import PROPS  # noqa: E402

VARIANTS = {
    "mixuni": {},
    "vtf": {"physics_head": "vtf"},
    "arrhenius": {"physics_head": "arrhenius"},
    "free_mlp": {"physics_head": "free_mlp"},
    "no_physics": {"physics_head": "none"},
    "per_property": {"shared_head": "per_property"},
    "geomix": {"shared_head": "per_property", "physics_head": "none", "use_gil": True, "gil_experts": 3},
}
STATS = {"prop_mean": [2.0, 3.0, 4.0], "prop_std": [1.5, 2.0, 1.0], "cond_log_mean": [1.0, 0.0, 0.0],
         "cond_log_std": [0.5, 1.0, 1.0]}


def _config(**model):
    cfg = load_config(CONFIG_DIR / "mixuni.yaml")
    cfg["properties"] = PROPS
    cfg["model"].update(hidden_dim=32, egnn_layers=2, mixture_layers=1, **model)
    return cfg


@pytest.mark.parametrize("variant", VARIANTS)
def test_forward_backward_and_reload(variant, toy_samples):
    torch.manual_seed(0)
    cfg = _config(**VARIANTS[variant])
    model = build_model(cfg, STATS)
    batch = collate(toy_samples(9))

    model.train()
    info = compute_loss(model, batch, weighting=build_weighting("uncertainty", 3), num_properties=3, aux_weight=1.0)
    info["loss"].backward()
    assert torch.isfinite(info["loss"])
    assert all(p.grad is not None for n, p in model.named_parameters() if "encoder.solvent_encoder" in n)

    model.eval()
    reloaded = build_model(cfg)
    reloaded.load_state_dict(model.state_dict())  # statistics travel with the weights
    reloaded.eval()
    with torch.no_grad():
        torch.manual_seed(1)  # GIL jitters its frames even in eval mode
        out = model(batch)
        torch.manual_seed(1)
        out2 = reloaded(batch)
    assert out["pred"].shape == (9,)
    assert torch.equal(out["pred_for_metrics"], out2["pred_for_metrics"])


def test_conductivity_head_round_trip():
    model = build_model(_config(), STATS).eval()
    head = model.head
    ids = torch.zeros(4, dtype=torch.long)
    sigma = torch.tensor([0.5, 2.0, 5.0, 10.0])
    z = head._sigma_to_z(sigma, ids)
    expected = (sigma - STATS["prop_mean"][0]) / STATS["prop_std"][0]
    assert torch.allclose(head._z_to_norm(z, ids), expected, atol=1e-5)
