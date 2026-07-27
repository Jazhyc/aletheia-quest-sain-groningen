# staged/

Notebooks that are built and tested but are not the one being submitted.

The runner accepts exactly one notebook per package and `submit.py` refuses to
build a zip with more, so only the candidate actually going out lives in
`submission/`. This directory is in `OUR_EXCLUDE_DIRS`, so nothing here is ever
uploaded.

To swap which one is queued:

```bash
mv submission/<current>.ipynb staged/
mv staged/<next>.ipynb submission/
```

Then re-point the `DEFAULT_SOURCE` / `DEFAULT_OUTPUT` paths in the affected
`build_sonic_*_notebook.py` and the `SOURCE` in their tests, and check the probe
weights under `submission/whitebox_probe/` are the ones the promoted notebook
expects.

| notebook | status |
| --- | --- |
| `sonic_v3_4.ipynb` | Built, never submitted. v3.2's caps with the v3.3 probe. Still the only clean way to tell whether v3.3's Iris loss came from `MAX_CAP` or from the probe. Worth a run if that attribution is wanted; `sonic_v3_5.ipynb` overtook it as the candidate. |
