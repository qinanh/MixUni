import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROPS = ["IL_conductivity", "IL_viscosity", "MiscSolv_density"]


def _graph(torch, Data, n_atoms, seed, name):
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(n_atoms, 16)
    x[:, 0] = 0.6
    x[torch.arange(n_atoms), 2 + torch.randint(0, 14, (n_atoms,), generator=g)] = 1.0
    src, dst = zip(*[(i, j) for i in range(n_atoms) for j in range(n_atoms) if i != j]) if n_atoms > 1 else ((0,), (0,))
    edge_index = torch.tensor([src, dst])
    pos = torch.randn(n_atoms, 3, generator=g)
    return Data(x=x, pos=pos, edge_index=edge_index, edge_attr=torch.rand(edge_index.size(1), generator=g), name=name)


@pytest.fixture
def toy_samples():
    torch = pytest.importorskip("torch")
    Data = pytest.importorskip("torch_geometric.data").Data

    def make(n=24, properties=PROPS):
        samples = []
        for i in range(n):
            prop = i % len(properties)
            name = properties[prop]
            salts = [_graph(torch, Data, 3, 1000 + i, f"salt{i}")] if name.startswith(("IL_", "PE_")) else \
                [Data(x=torch.zeros(1, 16), pos=torch.zeros(1, 3), edge_index=torch.tensor([[0], [0]]),
                      edge_attr=torch.zeros(1), name="dummy_salt")]
            samples.append({
                "sample_id": f"s{i}", "property_name": name, "property_id": prop,
                "graphs": [_graph(torch, Data, 2 + i % 3, i, f"m{i}"), _graph(torch, Data, 4, 500 + i, f"n{i}")],
                "salt_graphs": salts,
                "T": 280.0 + i, "c": 0.1 if salts[0].name != "dummy_salt" else None,
                "y_raw": 1.0 + 0.5 * i, "y": 0.1 * i,
            })
        return samples

    return make
