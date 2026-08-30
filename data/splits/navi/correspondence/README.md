# NAVI-Wild dense-correspondence splits

The three `pairs_wildset_*.csv` files are the correspondence family of the
canonical NAVI-Wild joint split. Their image union is disjoint from the
Wild files in `../estimation/` across all angular bins.

Do not edit these CSVs independently. Regenerate both Wild families and their
shared audit metadata with:

```bash
python data/splits/navi/generate.py --overwrite
(cd data/splits/navi && sha256sum -c SHA256SUMS)
```

The shared protocol, image assignment, quotas, and checksums live one level
above this directory in `manifest.json`, `image_partition.csv`,
`partition_capacity.csv`, and `SHA256SUMS`.
