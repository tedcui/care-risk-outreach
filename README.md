# Care Risk Outreach

An end-to-end risk-prioritization project that turns a predictive model into an interactive outreach planning dashboard for care-team leaders. The core dashboard remains usable without the AI service; optional Cloudflare Workers AI features add grounded explanations and aggregate text-to-SQL questions.

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
- `worker/src/index.js` - Cloudflare Worker for live insights and guarded text-to-SQL.
- `worker/schema.sql` - de-identified D1 analytical schema.
- `wrangler.jsonc` - local and deployment configuration for Workers AI and D1.
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

Open `stroke_prediction_analysis.ipynb` in Jupyter or VS Code to run the technical analysis. Rebuild the local dashboard preview and D1 seed with:

```powershell
python build_care_risk_dashboard.py
```

The default build writes `care_risk_outreach_dashboard.html` and generated files under `outputs/`. It does not change `index.html` or `docs/index.html`.

## Run the AI service locally

Install the JavaScript dependencies and authenticate Wrangler:

```powershell
npm.cmd install
npx.cmd wrangler login
```

Complete the one-time **Workers & Pages** onboarding in the Cloudflare dashboard and choose a `workers.dev` subdomain. Wrangler requires that account subdomain before it can proxy Workers AI during local development.

Initialize the local D1 database after building the dashboard:

```powershell
npm.cmd run ai:db:schema:local
npm.cmd run ai:db:seed:local
```

Start the Worker and a local dashboard server in separate terminals:

```powershell
npm.cmd run ai:dev
python -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/care_risk_outreach_dashboard.html`.

The AI service uses Cloudflare's `AI` binding and does not place an API key in the browser. Population questions generate read-only aggregate SQL against D1. The Worker blocks write operations, unapproved tables, patient-level rows, oversized results, and groups smaller than 10 patients.

## Publish after review

Publishing is explicit. After the Worker URL is available and the local preview is approved, run:

```powershell
npm.cmd run ai:deploy
npm.cmd run ai:db:schema:remote
npm.cmd run ai:db:seed:remote
python build_care_risk_dashboard.py --publish --ai-api-base https://care-risk-outreach-ai.<workers-subdomain>.workers.dev
```

Review the generated diff before committing or pushing.

## Notes

- Patient identifiers are excluded from dashboard data.
- Workers AI receives only the selected de-identified model or routing context.
- LLM output explains existing model results; it does not calculate risk, diagnose patients, or make final staffing decisions.
- Held-out predictions support model evaluation; the operational view scores the full cleaned population.
- Subgroup results are an initial performance assessment, not a fairness conclusion.
- Caregiver matching and program outcomes are clearly labeled MVP/demo concepts rather than validated production measures.

Related project: [Healthcare Data Pipeline](https://github.com/tedcui/healthcare-data-pipeline)
