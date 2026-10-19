# Stroke Outreach Prediction

An end-to-end stroke-risk prioritization project that turns a predictive model into an offline, interactive outreach dashboard for care-team leaders.

## Repository contents

- `stroke_prediction_analysis.ipynb` - primary analysis notebook covering data quality, EDA, preprocessing, model comparison, interpretation, and outreach thresholds.
- `build_dashboard_v3.py` - reproducible builder for the final dashboard and its embedded data.
- `stroke_prediction_dashboard_v3.html` - standalone interactive dashboard that works offline.
- `build_data_pipeline_dashboard.py` - reproducible builder for the simulated healthcare data-integration dashboard.
- `healthcare_data_pipeline_dashboard.html` - standalone interactive HIS/ICD/CPT ingestion, quality, transformation, and lineage dashboard.
- `src_stroke_prediction.csv` - source dataset used by the notebook and dashboard builder.
- `deliverable/` - assignment instructions and presentation artifact.

## Run locally

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Open `stroke_prediction_analysis.ipynb` in Jupyter or VS Code to run the analysis. Rebuild the dashboard with:

```powershell
python build_dashboard_v3.py
```

Then open `stroke_prediction_dashboard_v3.html` directly in a browser. The dashboard has no external JavaScript or internet dependency.

Build the separate data-pipeline dashboard with:

```powershell
python build_data_pipeline_dashboard.py
```

The stroke dashboard builder also writes `docs/index.html`, which is the GitHub Pages artifact for this repository. The workflow in `.github/workflows/pages.yml` publishes only that `docs/` directory after changes reach `main`.

The healthcare data-pipeline dashboard is intended for its own `healthcare-data-pipeline` repository and GitHub Pages site.

## Notes

- Patient identifiers are excluded from the dashboard data.
- Model performance is evaluated on a held-out test set; the operational dashboard scores the full cleaned population.
- Caregiver matching and program-outcome metrics are clearly labeled MVP/demo concepts rather than validated production measures.
- The separate HIS/ICD/CPT dashboard uses illustrative operational metadata to demonstrate ingestion, reconciliation, transformation, linkage, and field-level lineage.
