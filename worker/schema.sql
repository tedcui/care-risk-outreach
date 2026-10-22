PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS scored_patients (
  patient_rank INTEGER PRIMARY KEY,
  predicted_risk REAL NOT NULL,
  observed_event INTEGER NOT NULL CHECK (observed_event IN (0, 1)),
  gender TEXT NOT NULL,
  age REAL NOT NULL,
  age_group TEXT NOT NULL,
  hypertension INTEGER NOT NULL CHECK (hypertension IN (0, 1)),
  heart_disease INTEGER NOT NULL CHECK (heart_disease IN (0, 1)),
  residence_type TEXT NOT NULL,
  avg_glucose_level REAL NOT NULL,
  glucose_band TEXT NOT NULL,
  bmi REAL,
  bmi_band TEXT NOT NULL,
  smoking_status TEXT NOT NULL,
  error_group TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scored_risk ON scored_patients(predicted_risk DESC);
CREATE INDEX IF NOT EXISTS idx_scored_age_group ON scored_patients(age_group);
CREATE INDEX IF NOT EXISTS idx_scored_gender ON scored_patients(gender);
CREATE INDEX IF NOT EXISTS idx_scored_residence ON scored_patients(residence_type);

CREATE TABLE IF NOT EXISTS model_metrics (
  model TEXT PRIMARY KEY,
  roc_auc REAL,
  pr_auc REAL,
  sensitivity REAL,
  specificity REAL,
  precision REAL,
  top10_precision REAL,
  top10_recall REAL,
  top10_events_reached INTEGER
);

CREATE TABLE IF NOT EXISTS subgroup_metrics (
  population TEXT PRIMARY KEY,
  patient_count INTEGER NOT NULL,
  event_rate REAL,
  roc_auc REAL,
  pr_auc REAL,
  recall_top_10 REAL,
  global_top_10_share REAL
);

CREATE TABLE IF NOT EXISTS capacity_metrics (
  capacity_fraction REAL PRIMARY KEY,
  patients_contacted INTEGER NOT NULL,
  events_found INTEGER NOT NULL,
  precision_at_k REAL,
  recall_at_k REAL,
  lift REAL,
  minimum_score REAL
);

CREATE TABLE IF NOT EXISTS model_coefficients (
  feature TEXT PRIMARY KEY,
  coefficient REAL NOT NULL,
  odds_ratio REAL NOT NULL
);
