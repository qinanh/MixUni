# Third-party assets

The MIT license of this repository does not replace the licenses below.

## Data

| asset | license | source |
|---|---|---|
| CheMixHub (ionic liquids, polymer electrolytes, NIST-logV, logV, miscible solvents) | MIT | https://github.com/chemcognition-lab/chemixhub |

We do not redistribute the CheMixHub data or the processed files built from it.

## Methods reimplemented here

| method | file | reference |
|---|---|---|
| EGNN | `src/mixuni/model/encoder.py` | Satorras et al., E(n) Equivariant Graph Neural Networks, ICML 2021 |
| GeoMix graph interaction layer, single-task baselines | `src/mixuni/model/gil.py` | GeoMix, https://github.com/GLAD-RUC/GeoMix (MIT) |
| Sparse mixture of experts, load-balance loss | `src/mixuni/model/moe.py` | Shazeer et al., 2017; Fedus et al., Switch Transformer, 2022 |
| Uncertainty task weighting | `src/mixuni/train/losses.py` | Kendall et al., CVPR 2018 |
| GradNorm weight parameterisation | `src/mixuni/train/losses.py` | Chen et al., ICML 2018 |
| PCGrad | `src/mixuni/train/losses.py` | Yu et al., NeurIPS 2020 |
| Arrhenius, Vogel–Tammann–Fulcher, stretched Arrhenius | `src/mixuni/model/heads.py` | standard physical-chemistry forms |
| Butina clustering, Morgan fingerprints (chemical-OOD split) | split files in `data/splits/` | Butina, JCICS 1999; Rogers & Hahn, JCIM 2010 |

## Software

PyTorch (BSD-3), PyTorch Geometric (MIT), NumPy (BSD-3), pandas (BSD-3), PyYAML (MIT), pytest (MIT).
