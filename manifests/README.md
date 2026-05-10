# Manifests Directory

This directory stores frozen evaluation splits and reusable name lists.

Current canonical splits:

- `val_smoke4_seed42.txt`
- `val_probe40_seed42.txt`
- `val_full120_seed42.txt`

Important relationship:

- `val_probe40_seed42.txt` is a strict subset of `val_full120_seed42.txt`
- if a model is trained with `val_probe40_seed42` held out, then evaluating it on `val_full120_seed42` is not a clean validation because the other `80/120` `full120` images are in that training split

Rule:

- once a split becomes the comparison baseline, do not silently regenerate or replace it
- new experimental splits should use a clear filename and not overwrite the canonical ones
