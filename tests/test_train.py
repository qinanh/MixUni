import math

import pytest

torch = pytest.importorskip("torch")

from mixuni.data import collate  # noqa: E402
from mixuni.model import build_model  # noqa: E402
from mixuni.train import evaluate  # noqa: E402
from mixuni.train.engine import train_epoch  # noqa: E402
from mixuni.train.metrics import regression_metrics  # noqa: E402

from test_model import STATS, _config  # noqa: E402


def test_metrics_match_numpy():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50)
    m = regression_metrics(torch.tensor(a), torch.tensor(b))
    assert m["pearson"] == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-6)
    assert m["mae"] == pytest.approx(np.abs(a - b).mean(), abs=1e-6)
    assert math.isnan(regression_metrics(torch.tensor([float("nan")]), torch.tensor([1.0]))["mae"])


@pytest.mark.parametrize("weighting,pcgrad", [("uncertainty", False), ("gradnorm", False), ("plain", True)])
def test_train_epoch_and_evaluate(weighting, pcgrad, toy_samples):
    from mixuni.train import build_weighting
    torch.manual_seed(0)
    cfg = _config()
    cfg["train"].update(loss_weighting=weighting, pcgrad=pcgrad)
    model = build_model(cfg, STATS)
    w = build_weighting(weighting, 3)
    params = list(model.parameters()) + (list(w.parameters()) if w is not None else [])
    opt = torch.optim.AdamW(params, lr=1e-3)
    loader = [collate(toy_samples(9)), collate(toy_samples(6))]
    before = [p.detach().clone() for p in model.parameters()]
    tr = train_epoch(model, loader, opt, device="cpu", cfg=cfg["train"], weighting=w, scaler=None,
                     property_names=cfg["properties"])
    va = evaluate(model, loader, device="cpu", cfg=cfg["train"], weighting=w, property_names=cfg["properties"])
    assert math.isfinite(tr["overall"]["loss"]) and set(va["per_property"]) == set(cfg["properties"])
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
