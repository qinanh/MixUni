# MixUni

Code for **MixUni: Joint Multi-Property Prediction for Chemical Mixtures with
Physics-Informed Heads** (NeurIPS 2026).

MixUni trains one model on several properties of liquid mixtures: ionic-liquid
conductivity and viscosity, polymer-electrolyte conductivity, and the viscosity
and density of solvent mixtures, all from [CheMixHub](https://github.com/chemcognition-lab/chemixhub).

```
solvent graphs ── EGNN ──┐
                         ├─ mean over atoms ─ [GIL] ─ transformer ─ h ─┬─ shared MoE head ───────────┬─ y
salt graphs ───── EGNN ──┘                                             └─ stretched-Arrhenius head ──┘
                                                                         (conductivity, gated blend)
```

- Each component is encoded from its 3D structure by an EGNN, with its mole
  fraction as an atom feature; a coordinate-denoising loss regularises the encoder.
- A transformer over the components gives one embedding per mixture.
- A mixture-of-experts head shared by all properties predicts the queried property.
- For conductivity, the prediction is blended with a physics head,
  log σ = log A − B / T^γ, whose parameters are read from the mixture embedding.

## Setup

```bash
git clone https://github.com/qinanh/MixUni.git && cd MixUni
pip install -e ".[dev]"
```

Tested with Python 3.10–3.11, PyTorch 2.4 and PyTorch Geometric 2.7.

## Data

The processed datasets are built from the public CheMixHub CSVs and a
coordinate-cache archive (`mixuni_coord_caches.tar`). Download both as
described in [data/README.md](data/README.md), then:

```bash
python scripts/prepare_data.py   # writes data/processed/<property>.pt
python scripts/check_data.py     # same samples, in the same order, as ours
```

The split files in `data/splits/` index samples by position, so run the check
before using them.

## Training

```bash
python scripts/train.py configs/mixuni.yaml              # main model
python scripts/train.py configs/mixuni.yaml --seed 43    # another training seed
python scripts/train.py configs/ablations/no_moe.yaml    # any other experiment
```

A run writes `artifacts/runs/<config>[_seed<N>]/` with the full config, the
per-epoch train/val/test metrics (`metrics.json`) and `checkpoints/{best,latest}.pt`.
`best.pt` is the epoch with the lowest validation loss. One run of the main
model takes about 6 h on one A100.

Configs are short: `configs/mixuni.yaml` lists every option, and the other
configs start from it and change a few lines. For example:

```yaml
# configs/ablations/no_physics.yaml
base: ../mixuni.yaml
model:
  physics_head: none
```

`python scripts/train.py <config> --dry-run` prints the merged config.

## Pretrained models

The weights of the runs in the paper are attached to the
[v1.0.0 release](https://github.com/qinanh/MixUni/releases/tag/v1.0.0)
([checkpoints/MANIFEST.md](checkpoints/MANIFEST.md) lists every file):

```bash
wget https://github.com/qinanh/MixUni/releases/download/v1.0.0/mixuni_checkpoints.tar
tar -xf mixuni_checkpoints.tar -C checkpoints --strip-components=1
sha256sum -c checkpoints/SHA256SUMS
```

To evaluate one on its test split:

```bash
python scripts/evaluate.py checkpoints/mixuni/best.pt
```

or, from Python:

```python
from mixuni.model import load_model

model, ckpt = load_model("checkpoints/mixuni/best.pt")
```

## Reproducing the paper

| result | configs | command |
|---|---|---|
| main results, random split | `mixuni.yaml`, `single_task/*.yaml` | `bash scripts/reproduce.sh main` |
| ablations | `ablations/*.yaml` | `bash scripts/reproduce.sh ablations` |
| conductivity heads | `physics_heads/*.yaml` | `bash scripts/reproduce.sh physics_heads` |
| chemical-OOD split | `chem_ood/mixuni.yaml`, `chem_ood/single_task/*.yaml` | `bash scripts/reproduce.sh chem_ood` |
| five-seed error bars | the above with `--seed 41 … 45` | `bash scripts/reproduce.sh seeds` |

The random split is 70/20/10 with seed 42. The chemical-OOD split clusters
mixtures by fingerprint similarity (Butina, Tanimoto 0.4) and holds out whole
clusters; its indices are in `data/splits/chem_ood/`. The single-task
baselines are GeoMix models: the same encoder with GeoMix graph interaction
layers (GIL) and one MLP head.

## Code

```
configs/                 one YAML per experiment
src/mixuni/
  config.py              config loading (base + overrides)
  data.py                processed datasets, splits, target normalisation, batching
  model/
    encoder.py           EGNN and the mixture encoder
    gil.py               GeoMix graph interaction layer (single-task baselines)
    moe.py               shared mixture-of-experts head, per-property heads
    heads.py             MLP expert and the physics heads (Arrhenius, VTF, stretched Arrhenius)
    conductivity.py      blend of the shared and physics heads for conductivity
    mixuni.py            the full model, build_model, load_model
    layers.py            MLPs, scatter helpers, transformer pooling, local frames
  train/
    engine.py            training/evaluation loops, checkpoints
    losses.py            task weighting (uncertainty, GradNorm) and PCGrad
    metrics.py           MAE, RMSE, Pearson, Spearman
scripts/                 prepare_data.py, check_data.py, train.py, evaluate.py, reproduce.sh
data/splits/             chemical-OOD split indices
tests/
```

## License

MIT, see [LICENSE](LICENSE). Datasets and reimplemented methods are credited in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
