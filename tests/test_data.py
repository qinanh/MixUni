import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import Subset  # noqa: E402

from mixuni.data import balanced_sampler, collate, make_splits, normalize_targets, target_statistics  # noqa: E402

from conftest import PROPS  # noqa: E402


class _DS(torch.utils.data.Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def test_random_split_sizes_add_up(toy_samples):
    ds = _DS(toy_samples(23))
    splits = make_splits(ds, "random")
    assert [len(splits[k]) for k in ("train", "val", "test")] == [16, 5, 2]
    assert sorted(i for s in splits.values() for i in s.indices) == list(range(23))


def test_statistics_and_normalization(toy_samples):
    ds = _DS(toy_samples(12))
    train = Subset(ds, list(range(12)))
    stats = target_statistics(train, PROPS)
    normalize_targets(ds, stats)
    for i in range(len(PROPS)):
        y = torch.tensor([s["y"] for s in ds.samples if s["property_id"] == i])
        assert abs(float(y.mean())) < 1e-5 and abs(float(y.std(unbiased=False)) - 1) < 1e-4
    assert stats["cond_log_mean"][1] == 0.0 and stats["cond_log_mean"][0] != 0.0  # only conductivities


def test_collate_index_maps(toy_samples):
    samples = toy_samples(5)
    batch = collate(samples)
    assert batch["g2b"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] + [0, 1, 2, 3, 4]
    assert batch["n2g"].numel() == batch["graphs"].num_nodes + batch["salt_graphs"].num_nodes
    assert torch.equal(batch["n2b"], batch["g2b"][batch["n2g"]])
    assert batch["c"][2].item() == 0.0  # salt-free sample gets c = 0


def test_balanced_sampler_weights(toy_samples):
    samples = toy_samples(10)
    weights = balanced_sampler(Subset(_DS(samples), list(range(10)))).weights
    per_prop = [float(weights[[s["property_id"] == p for s in samples]].sum()) for p in range(3)]
    assert per_prop == pytest.approx([1.0, 1.0, 1.0])
