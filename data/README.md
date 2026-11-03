# Data

Training reads one processed file per property from `data/processed/`:

| file | property | CheMixHub folder | samples |
|---|---|---|---:|
| `il_conductivity.pt` | `IL_conductivity` | `ionic-liquids` | 35,348 |
| `il_viscosity.pt` | `IL_viscosity` | `ionic-liquids` | 65,862 |
| `pe_conductivity.pt` | `PE_conductivity` | `polymer-electrolyte` | 10,704 |
| `nist_viscosity.pt` | `NIST_viscosity` | `nist-logV` | 235,369 |
| `logv_viscosity.pt` | `logV_viscosity` | `logV` | 31,464 |
| `miscsolv_density.pt` | `MiscSolv_density` | `miscible-solvent` | 30,061 |

We do not redistribute these files; build them from the two inputs below.

## 1. CheMixHub CSVs

From https://github.com/chemcognition-lab/chemixhub (MIT), commit `356c16b99e38b56a91b5219d33db410fc372d538`:

```bash
git clone https://github.com/chemcognition-lab/chemixhub.git /path/to/chemixhub
git -C /path/to/chemixhub checkout 356c16b99e38b56a91b5219d33db410fc372d538
mkdir -p data/raw/chemixhub && ln -s /path/to/chemixhub/datasets data/raw/chemixhub/datasets
```

Only `datasets/<folder>/processed_data/{processed_*.csv,compounds.csv}` are read:

| file | sha256 |
|---|---|
| `ionic-liquids/processed_data/processed_IlThermoData.csv` | `ec526a4e7491fca3268e0d2548bcb7d3aabcd43bf9453b37bba122610a878a4b` |
| `ionic-liquids/processed_data/compounds.csv` | `e4aba67dc12ae1dff3e10aa1b6f81099c07fd30f654aa091927889cad5e46ca7` |
| `polymer-electrolyte/processed_data/processed_PolymerElectrolyteData.csv` | `293907e2590aa784e372186d481b025c44908da8b8093fd6236cd7ffb74f9d6b` |
| `polymer-electrolyte/processed_data/compounds.csv` | `82503f37a87670968004f4723259567827f80e6acb6967670f3a7ad5a92cf324` |
| `nist-logV/processed_data/processed_NISTlogV.csv` | `f904ec1c16e21626b5a46c50aac833ebdeb6dc4069f5d41e10df6b7d69e573b8` |
| `nist-logV/processed_data/compounds.csv` | `48b7c1733a381ddcf71d88ac69858cc718d56022d224574c37ab1f52ce583166` |
| `logV/processed_data/processed_logV.csv` | `a92d126886603874e493484ef493d5f4eaa7c541e26919f2d09fdc308c78930c` |
| `logV/processed_data/compounds.csv` | `c3edcc99be57053bba1aed2139afa59c5e1e7425d1f1123c1175fe0520eb5174` |
| `miscible-solvent/processed_data/processed_MiscibleSolventData.csv` | `586c903cc848730c4d213d6357ecbffe9c359ff21bca746e5bf2ec298b275341` |
| `miscible-solvent/processed_data/compounds.csv` | `b5220b72868af6bf19a957e6ef893fc6cf63f936238e816b806e482698d8354c` |

## 2. Coordinate caches

CheMixHub has no 3D structures. `mixuni_coord_caches.tar` (attached to the
[v1.0.0 release](https://github.com/qinanh/MixUni/releases/tag/v1.0.0), sha256
`1daa2dfc3d6fbd34ccc0d7e54c7d4524d73bfbb608fdc0256def49861487b0f0`) holds one
conformer per compound with a 6 Å radius graph:

```bash
wget https://github.com/qinanh/MixUni/releases/download/v1.0.0/mixuni_coord_caches.tar
tar -xf mixuni_coord_caches.tar -C data/raw/chemixhub
# -> data/raw/chemixhub/coord_data/coords-chemixhub-{ionic-liquids,polymer-electrolyte,nist-logV,miscible-solvent}.npz
#    data/raw/chemixhub/coord_data_logv/coords-chemixhub-nist-logV.npz
```

Conformer generation is random, so caches you generate yourself will not
reproduce our files exactly.

## 3. Build and check

```bash
python scripts/prepare_data.py    # about 30 min and 5 GB of RAM (NIST_viscosity is the largest)
python scripts/check_data.py      # compares against data/fingerprints.json
```

`check_data.py` compares the ordered sample identities (id, components, mole
fractions, T, c, target) and the component graphs. The split files refer to
samples by position, so they are only valid if the identity check passes.

## Notes

- `NIST_viscosity` keeps rows with log-viscosity in [−8, 10] and drops mixtures
  whose only components are single atoms (no edges).
- `logV_viscosity` was built with an earlier NIST-logV coordinate cache
  (`coord_data_logv/`), and its component graphs are looked up by compound id in
  that cache. The `logV` and `nist-logV` folders number compounds independently,
  so most `logV` components receive the graph of a different molecule. All
  reported `logV_viscosity` results use this lookup, and `prepare_data.py`
  reproduces it to stay consistent with the splits and checkpoints.
- For polymer electrolytes the cached structure is the repeat unit, not the
  oligomer.
