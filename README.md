# Care Risk Outreach

An end-to-end risk-prioritization project that turns a predictive model into an offline, interactive outreach planning dashboard for care-team leaders.

## Live dashboard

https://tedcui.github.io/care-risk-outreach/

## Dashboard workflow

1. **Outreach Planner** - converts staffing capacity into an achievable prioritized call list and expected event capture.
2. **Model & Subgroup Review** - combines held-out model performance, subgroup checks, classification review, and predictive drivers.
3. **Priority Patient List** - provides a sortable outreach queue with patient-level model context.
4. **Caregiver Match** - demonstrates capacity-aware routing to an available care-team member.
5. **Program Outcomes** - tracks patient engagement and operational follow-through after outreach.

## Repository contents

- `stroke_prediction_analysis.ipynb` - source analysis notebook covering data quality, preprocessing, model comparison, interpretation, and outreach thresholds.
- `build_care_risk_dashboard.py` - reproducible builder for the final dashboard and embedded data.
- `care_risk_outreach_dashboard.html` - standalone interactive dashboard that works offline.
- `index.html` - GitHub Pages homepage.
- `src_stroke_prediction.csv` - source dataset used by the notebook and dashboard builder.

The source dataset contains a specific binary clinical outcome. The dashboard presents it as a **high-risk event** for a broader operational MVP. Production use would require a formally governed outcome definition, prediction window, calibration review, and external validation.

## Run locally

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Open `stroke_prediction_analysis.ipynb` in Jupyter or VS Code to run the technical analysis. Rebuild the dashboard with:

```powershell
python build_care_risk_dashboard.py
```

The builder writes `care_risk_outreach_dashboard.html`, `index.html`, and `docs/index.html`. The dashboard has no external JavaScript or internet dependency.

## Notes

- Patient identifiers are excluded from dashboard data.
- Held-out predictions support model evaluation; the operational view scores the full cleaned population.
- Subgroup results are an initial performance assessment, not a fairness conclusion.
- Caregiver matching and program outcomes are clearly labeled MVP/demo concepts rather than validated production measures.

Related project: [Healthcare Data Pipeline](https://github.com/tedcui/healthcare-data-pipeline)
