# Checkpoints

`mixuni_checkpoints.tar` (112 MB, 52 weight files, SHA-256 `9bc5202a60d875cf71624311a3b18d9244768e5d2c08a4545c017cfd7b1876e8`) holds the
weights of the runs reported in the paper. Unpack it here and verify:

```bash
tar -xf mixuni_checkpoints.tar -C checkpoints --strip-components=1
sha256sum -c checkpoints/SHA256SUMS
```

Each run directory has the same name as the output of the training command in the table,
and contains `config.yaml` with `best.pt` (lowest validation loss) and, where saved,
`latest.pt` (last epoch). Load with `mixuni.model.load_model(path)` or evaluate with
`python scripts/evaluate.py <path>`. Optimizer states are not included.

`mixuni/` and `chem_ood/single_task/*/` are the runs of the original tables; the `_seed<N>`
directories are the five-seed runs (seed 42 among them is a separate run of the same config).
Weights were not kept for the runs of `ablations/pcgrad`, `physics_heads/{arrhenius,vtf,free_mlp}`,
and the random-split single-task baselines other than PE conductivity. `latest.pt` is left out where
the run diverged and its final weights are NaN.
`physics_heads/stretched_arrhenius` was launched with a longer epoch budget (500) and the same
cosine schedule (T_max 200), so up to its `best.pt` (epoch 119) it trained exactly as the config does.

| run | file | trained with | epoch | best epoch | SHA-256 (first 16) | MB |
|---|---|---|---|---|---|---|
| `mixuni` | `best.pt` | `configs/mixuni.yaml` | 94 | 94 | `8e4ba51342162a5a` | 1.9 |
| `mixuni_seed41` | `best.pt` | `configs/mixuni.yaml --seed 41` | 116 | 116 | `d11d9157dbfa512f` | 1.9 |
| `mixuni_seed41` | `latest.pt` | `configs/mixuni.yaml --seed 41` | 199 | 116 | `4e7992319624336d` | 1.9 |
| `mixuni_seed42` | `best.pt` | `configs/mixuni.yaml --seed 42` | 101 | 101 | `8d5203834da40d6c` | 1.9 |
| `mixuni_seed42` | `latest.pt` | `configs/mixuni.yaml --seed 42` | 199 | 101 | `2a410f14a7d5dc4a` | 1.9 |
| `mixuni_seed43` | `best.pt` | `configs/mixuni.yaml --seed 43` | 109 | 109 | `f07c1a34d7adf865` | 1.9 |
| `mixuni_seed43` | `latest.pt` | `configs/mixuni.yaml --seed 43` | 199 | 109 | `b418e1fe52ea657e` | 1.9 |
| `mixuni_seed44` | `best.pt` | `configs/mixuni.yaml --seed 44` | 113 | 113 | `48797ecaced80328` | 1.9 |
| `mixuni_seed44` | `latest.pt` | `configs/mixuni.yaml --seed 44` | 199 | 113 | `5bc975f4d73e9e50` | 1.9 |
| `mixuni_seed45` | `best.pt` | `configs/mixuni.yaml --seed 45` | 104 | 104 | `dd6a34e318325413` | 1.9 |
| `mixuni_seed45` | `latest.pt` | `configs/mixuni.yaml --seed 45` | 199 | 104 | `24fe04093c55aec6` | 1.9 |
| `ablations/gradnorm` | `best.pt` | `configs/ablations/gradnorm.yaml` | 91 | 91 | `2cfa47ab5e939b1a` | 1.9 |
| `ablations/no_denoise` | `best.pt` | `configs/ablations/no_denoise.yaml` | 101 | 101 | `4b0822a09bcbc12e` | 1.9 |
| `ablations/no_moe` | `best.pt` | `configs/ablations/no_moe.yaml` | 150 | 150 | `8de4ba6041deb9e5` | 1.9 |
| `ablations/no_moe` | `latest.pt` | `configs/ablations/no_moe.yaml` | 299 | 117 | `4fd3d6007ef9357a` | 1.9 |
| `ablations/no_physics` | `best.pt` | `configs/ablations/no_physics.yaml` | 92 | 92 | `e9d4fcad82495b8a` | 1.8 |
| `ablations/per_property_heads` | `best.pt` | `configs/ablations/per_property_heads.yaml` | 98 | 98 | `8bd86cda0f423fd5` | 1.8 |
| `ablations/plain` | `best.pt` | `configs/ablations/plain.yaml` | 198 | 198 | `6a1ce5b76024ea0b` | 1.9 |
| `physics_heads/stretched_arrhenius` | `best.pt` | `configs/physics_heads/stretched_arrhenius.yaml` | 119 | 119 | `0e63350aacd9ca14` | 1.9 |
| `single_task/pe_conductivity` | `best.pt` | `configs/single_task/pe_conductivity.yaml` | 183 | 183 | `82a46d225c0b771c` | 2.5 |
| `chem_ood/mixuni_seed41` | `best.pt` | `configs/chem_ood/mixuni.yaml --seed 41` | 5 | 5 | `a72b564f5526c7ad` | 1.9 |
| `chem_ood/mixuni_seed41` | `latest.pt` | `configs/chem_ood/mixuni.yaml --seed 41` | 199 | 5 | `f852f3b0ee47065f` | 1.9 |
| `chem_ood/mixuni_seed42` | `best.pt` | `configs/chem_ood/mixuni.yaml --seed 42` | 16 | 16 | `4c3765a6a0a7a312` | 1.9 |
| `chem_ood/mixuni_seed42` | `latest.pt` | `configs/chem_ood/mixuni.yaml --seed 42` | 199 | 16 | `7543313971137c9b` | 1.9 |
| `chem_ood/mixuni_seed43` | `best.pt` | `configs/chem_ood/mixuni.yaml --seed 43` | 7 | 7 | `6855dba0f4a51e9a` | 1.9 |
| `chem_ood/mixuni_seed43` | `latest.pt` | `configs/chem_ood/mixuni.yaml --seed 43` | 199 | 7 | `823ef91c2d38712b` | 1.9 |
| `chem_ood/mixuni_seed44` | `best.pt` | `configs/chem_ood/mixuni.yaml --seed 44` | 19 | 19 | `0d54ded3c7e16a8a` | 1.9 |
| `chem_ood/mixuni_seed44` | `latest.pt` | `configs/chem_ood/mixuni.yaml --seed 44` | 199 | 19 | `d8fa17c298a52446` | 1.9 |
| `chem_ood/mixuni_seed45` | `best.pt` | `configs/chem_ood/mixuni.yaml --seed 45` | 5 | 5 | `ec904b5ac2a00be3` | 1.9 |
| `chem_ood/mixuni_seed45` | `latest.pt` | `configs/chem_ood/mixuni.yaml --seed 45` | 199 | 5 | `4281b3f12fa5f46d` | 1.9 |
| `chem_ood/single_task/il_conductivity` | `best.pt` | `configs/chem_ood/single_task/il_conductivity.yaml` | 117 | 117 | `ee44b262889de708` | 2.5 |
| `chem_ood/single_task/il_conductivity_seed41` | `best.pt` | `configs/chem_ood/single_task/il_conductivity.yaml --seed 41` | 194 | 194 | `4bb33e65e1284f11` | 2.5 |
| `chem_ood/single_task/il_conductivity_seed41` | `latest.pt` | `configs/chem_ood/single_task/il_conductivity.yaml --seed 41` | 199 | 194 | `3eddc29137afc5a0` | 2.5 |
| `chem_ood/single_task/il_conductivity_seed42` | `best.pt` | `configs/chem_ood/single_task/il_conductivity.yaml --seed 42` | 110 | 110 | `5c909477f8725d0b` | 2.5 |
| `chem_ood/single_task/il_conductivity_seed42` | `latest.pt` | `configs/chem_ood/single_task/il_conductivity.yaml --seed 42` | 199 | 110 | `3c36ee9fd515c010` | 2.5 |
| `chem_ood/single_task/il_conductivity_seed43` | `best.pt` | `configs/chem_ood/single_task/il_conductivity.yaml --seed 43` | 15 | 15 | `384f6f861f15f545` | 2.5 |
| `chem_ood/single_task/il_conductivity_seed44` | `best.pt` | `configs/chem_ood/single_task/il_conductivity.yaml --seed 44` | 16 | 16 | `3b758edab8d82d14` | 2.5 |
| `chem_ood/single_task/il_conductivity_seed45` | `best.pt` | `configs/chem_ood/single_task/il_conductivity.yaml --seed 45` | 121 | 121 | `66962898e7d0d764` | 2.5 |
| `chem_ood/single_task/il_conductivity_seed45` | `latest.pt` | `configs/chem_ood/single_task/il_conductivity.yaml --seed 45` | 199 | 121 | `382ebc7084896ac8` | 2.5 |
| `chem_ood/single_task/il_viscosity` | `best.pt` | `configs/chem_ood/single_task/il_viscosity.yaml` | 183 | 183 | `813fffec763eba8c` | 2.5 |
| `chem_ood/single_task/logv_viscosity` | `best.pt` | `configs/chem_ood/single_task/logv_viscosity.yaml` | 199 | 199 | `630a5638a54746c0` | 2.5 |
| `chem_ood/single_task/miscsolv_density` | `best.pt` | `configs/chem_ood/single_task/miscsolv_density.yaml` | 198 | 198 | `308b7cc0b287bea6` | 2.5 |
| `chem_ood/single_task/pe_conductivity` | `best.pt` | `configs/chem_ood/single_task/pe_conductivity.yaml` | 38 | 38 | `dbd8e9b142fa904d` | 2.5 |
| `chem_ood/single_task/pe_conductivity_seed41` | `best.pt` | `configs/chem_ood/single_task/pe_conductivity.yaml --seed 41` | 17 | 17 | `a925bdbd09509816` | 2.5 |
| `chem_ood/single_task/pe_conductivity_seed42` | `best.pt` | `configs/chem_ood/single_task/pe_conductivity.yaml --seed 42` | 107 | 107 | `75316815293c56c3` | 2.5 |
| `chem_ood/single_task/pe_conductivity_seed42` | `latest.pt` | `configs/chem_ood/single_task/pe_conductivity.yaml --seed 42` | 199 | 107 | `16599bc01fd31d3f` | 2.5 |
| `chem_ood/single_task/pe_conductivity_seed43` | `best.pt` | `configs/chem_ood/single_task/pe_conductivity.yaml --seed 43` | 124 | 124 | `15c8f1a5322fbc55` | 2.5 |
| `chem_ood/single_task/pe_conductivity_seed43` | `latest.pt` | `configs/chem_ood/single_task/pe_conductivity.yaml --seed 43` | 199 | 124 | `17fd12c7a34b773e` | 2.5 |
| `chem_ood/single_task/pe_conductivity_seed44` | `best.pt` | `configs/chem_ood/single_task/pe_conductivity.yaml --seed 44` | 26 | 26 | `247848568dedc1eb` | 2.5 |
| `chem_ood/single_task/pe_conductivity_seed44` | `latest.pt` | `configs/chem_ood/single_task/pe_conductivity.yaml --seed 44` | 199 | 26 | `500d04c12c1571f8` | 2.5 |
| `chem_ood/single_task/pe_conductivity_seed45` | `best.pt` | `configs/chem_ood/single_task/pe_conductivity.yaml --seed 45` | 145 | 145 | `be50cfa0bf459e07` | 2.5 |
| `chem_ood/single_task/pe_conductivity_seed45` | `latest.pt` | `configs/chem_ood/single_task/pe_conductivity.yaml --seed 45` | 199 | 145 | `4992090c43f03e64` | 2.5 |
