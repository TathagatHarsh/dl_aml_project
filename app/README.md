# EngineMind Phase 1 frontend

A responsive, dependency-free dashboard displaying the repository's real NASA C-MAPSS FD001 evaluation results. Includes engine search and filtering, six sensor histories per engine, prediction scatter plot, model comparison, methodology, and CSV export.

From the repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory app
```

Open http://127.0.0.1:8000. To refresh the snapshot after regenerating ML reports:

```sh
python3 app/prepare_data.py
```

The frontend uses saved predictions; it does not run live inference or upload files. All metrics on the overview use the capped target. Model comparison uses grouped cross-validation reports and identifies different targets. Fleet bands (≤30, >30–60, >60 cycles) are illustrative, not operational maintenance recommendations. Google Fonts is optional; system fonts work offline.

The data exporter checks that all 100 test engines have sensor histories ending at their reported last observed cycle. Existing Python model and training code are unchanged.
