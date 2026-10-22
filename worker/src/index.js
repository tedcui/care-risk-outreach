const AI_MODELS = [
  "@cf/google/gemma-4-26b-a4b-it",
  "@cf/zai-org/glm-4.7-flash",
];
const MAX_BODY_CHARS = 12000;
const MAX_QUESTION_CHARS = 500;
const MAX_SQL_CHARS = 2400;
const RATE_LIMIT = 24;
const RATE_WINDOW_MS = 60_000;

const ALLOWED_ORIGINS = new Set([
  "https://tedcui.github.io",
  "http://localhost:8000",
  "http://127.0.0.1:8000",
  "http://localhost:5500",
  "http://127.0.0.1:5500",
]);

const ALLOWED_TABLES = new Set([
  "scored_patients",
  "model_metrics",
  "subgroup_metrics",
  "capacity_metrics",
  "model_coefficients",
]);

const buckets = new Map();

function corsHeaders(request) {
  const origin = request.headers.get("Origin");
  const headers = {
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Max-Age": "86400",
    "Cache-Control": "no-store",
    "Content-Type": "application/json; charset=utf-8",
  };
  if (origin && ALLOWED_ORIGINS.has(origin)) {
    headers["Access-Control-Allow-Origin"] = origin;
    headers.Vary = "Origin";
  }
  return headers;
}

function jsonResponse(request, payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: corsHeaders(request),
  });
}

function assertAllowedOrigin(request) {
  const origin = request.headers.get("Origin");
  if (origin && !ALLOWED_ORIGINS.has(origin)) {
    throw new HttpError(403, "This origin is not allowed to call the demo API.");
  }
}

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function checkRateLimit(request) {
  const now = Date.now();
  const ip = request.headers.get("CF-Connecting-IP") || "local";
  const current = buckets.get(ip);
  if (!current || now >= current.resetAt) {
    buckets.set(ip, { count: 1, resetAt: now + RATE_WINDOW_MS });
    return;
  }
  current.count += 1;
  if (current.count > RATE_LIMIT) {
    throw new HttpError(429, "The AI demo is receiving too many requests. Try again in one minute.");
  }
}

async function readJson(request) {
  const text = await request.text();
  if (text.length > MAX_BODY_CHARS) {
    throw new HttpError(413, "The request is too large.");
  }
  try {
    return JSON.parse(text || "{}");
  } catch {
    throw new HttpError(400, "The request body must be valid JSON.");
  }
}

function finiteNumber(value, fallback = null) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function textValue(value, max = 120) {
  return String(value ?? "").replace(/[\u0000-\u001f\u007f]/g, " ").trim().slice(0, max);
}

function patientContext(raw = {}) {
  return {
    priority: textValue(raw.priority, 16),
    predicted_risk: finiteNumber(raw.predicted_risk),
    queue_rank: finiteNumber(raw.queue_rank),
    planned_outreach_list_size: finiteNumber(raw.contacted_count),
    age: finiteNumber(raw.age),
    gender: textValue(raw.gender, 20),
    residence: textValue(raw.residence, 20),
    hypertension: Boolean(raw.hypertension),
    heart_disease: Boolean(raw.heart_disease),
    avg_glucose_level: finiteNumber(raw.avg_glucose_level),
    glucose_band: textValue(raw.glucose_band, 20),
    bmi: finiteNumber(raw.bmi),
    bmi_band: textValue(raw.bmi_band, 20),
    smoking_status: textValue(raw.smoking_status, 40),
    model_signals: Array.isArray(raw.model_signals)
      ? raw.model_signals.slice(0, 8).map((item) => ({
          feature: textValue(item?.feature, 50),
          odds_ratio: finiteNumber(item?.odds_ratio),
        }))
      : [],
  };
}

function caregiverContext(raw = {}) {
  const caregiver = raw.caregiver || {};
  return {
    patient: patientContext(raw.patient || {}),
    caregiver: {
      name: textValue(caregiver.name, 80),
      role: textValue(caregiver.role, 60),
      languages: textValue(caregiver.languages, 80),
      specialty: textValue(caregiver.specialty, 80),
      availability: textValue(caregiver.availability, 30),
      distance_miles: finiteNumber(caregiver.distance_miles),
      open_capacity: finiteNumber(caregiver.open_capacity),
      match_score: finiteNumber(caregiver.match_score),
      strengths: Array.isArray(caregiver.strengths)
        ? caregiver.strengths.slice(0, 5).map((value) => textValue(value, 140))
        : [],
      watch_outs: Array.isArray(caregiver.watch_outs)
        ? caregiver.watch_outs.slice(0, 5).map((value) => textValue(value, 140))
        : [],
    },
  };
}

function outreachContext(raw = {}) {
  const recommendation = raw.recommendation || {};
  return {
    population: textValue(raw.population, 50),
    patient_count: finiteNumber(raw.patient_count),
    current_outreach_percent: finiteNumber(raw.current_outreach_percent),
    patients_contacted: finiteNumber(raw.patients_contacted),
    observed_events_total: finiteNumber(raw.observed_events_total),
    events_reached: finiteNumber(raw.events_reached),
    precision: finiteNumber(raw.precision),
    recall: finiteNumber(raw.recall),
    lift: finiteNumber(raw.lift),
    minimum_score: finiteNumber(raw.minimum_score),
    manual_capacity: finiteNumber(raw.manual_capacity),
    capacity_used: finiteNumber(raw.capacity_used),
    objective: textValue(raw.objective, 40),
    active_constraints: Array.isArray(raw.active_constraints)
      ? raw.active_constraints.slice(0, 8).map((value) => textValue(value, 100))
      : [],
    recommendation: {
      feasible: Boolean(recommendation.feasible),
      outreach_percent: finiteNumber(recommendation.outreach_percent),
      patients_contacted: finiteNumber(recommendation.patients_contacted),
      events_reached: finiteNumber(recommendation.events_reached),
      precision: finiteNumber(recommendation.precision),
      recall: finiteNumber(recommendation.recall),
      lift: finiteNumber(recommendation.lift),
      minimum_score: finiteNumber(recommendation.minimum_score),
      unmet_constraints: Array.isArray(recommendation.unmet_constraints)
        ? recommendation.unmet_constraints.slice(0, 8).map((value) => textValue(value, 100))
        : [],
    },
  };
}

async function runModel(env, messages, maxTokens = 700) {
  let lastError;
  for (const model of AI_MODELS) {
    try {
      const result = await env.AI.run(model, {
        messages,
        max_completion_tokens: maxTokens,
        temperature: 0.15,
        chat_template_kwargs: {
          enable_thinking: false,
        },
      });
      const choiceContent = result?.choices?.[0]?.message?.content;
      const text = typeof result === "string"
        ? result
        : typeof result?.response === "string"
          ? result.response
          : typeof result?.result?.response === "string"
            ? result.result.response
            : typeof choiceContent === "string"
              ? choiceContent
              : Array.isArray(choiceContent)
                ? choiceContent.map((part) => part?.text || "").join("")
                : typeof result?.choices?.[0]?.text === "string"
                  ? result.choices[0].text
                  : typeof result?.output_text === "string"
                    ? result.output_text
            : JSON.stringify(result);
      return { text, model };
    } catch (error) {
      lastError = error;
      console.warn(`Workers AI model failed: ${model}`, error);
    }
  }
  throw lastError || new Error("No Workers AI model was available.");
}

export function extractJsonObject(value) {
  const text = String(value ?? "").replace(/```(?:json)?/gi, "").replace(/```/g, "").trim();
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start < 0 || end <= start) throw new Error("The model did not return a JSON object.");
  return JSON.parse(text.slice(start, end + 1));
}

function normalizeList(value, limit = 4) {
  return Array.isArray(value) ? value.slice(0, limit).map((item) => textValue(item, 240)).filter(Boolean) : [];
}

function normalizeInsight(value) {
  return {
    headline: textValue(value?.headline, 100) || "AI-generated review",
    summary: textValue(value?.summary, 600),
    supporting_factors: normalizeList(value?.supporting_factors, 4),
    watch_outs: normalizeList(value?.watch_outs, 3),
  };
}

async function generateInsight(env, kind, context) {
  const prompts = {
    patient: "You are an analytics assistant supporting a care outreach operations team. Explain only the supplied predictive-model facts. Do not diagnose, prescribe, recommend treatment, infer missing facts, or claim causality. The risk score prioritizes outreach and is not a clinical determination. Return JSON only with keys headline, summary, supporting_factors, and watch_outs. supporting_factors and watch_outs must be short arrays.",
    caregiver: "You are an analytics assistant explaining an illustrative caregiver-routing score. Use only the supplied patient and caregiver facts. Do not invent qualifications, clinical needs, availability, or outcomes. Make clear that the routing score supports human review and is not a staffing rule. Return JSON only with keys headline, summary, supporting_factors, and watch_outs. supporting_factors and watch_outs must be short arrays.",
    outreach: "You are an operations analytics assistant explaining a capacity-based outreach plan. Use only the supplied current metrics, deterministic optimizer recommendation, objective, and constraints. Clearly distinguish the current plan from the recommended plan. Never calculate a new threshold, alter model scores, claim a universally best threshold, diagnose, prescribe, or infer missing facts. If constraints are infeasible, identify the supplied unmet constraints. Return JSON only with keys headline, summary, supporting_factors, and watch_outs. supporting_factors and watch_outs must be short arrays.",
  };
  const labels = { patient: "patient outreach ranking", caregiver: "caregiver match", outreach: "outreach operating plan" };
  const system = prompts[kind];
  const result = await runModel(env, [
    { role: "system", content: system },
    { role: "user", content: `Explain this ${labels[kind]}:\n${JSON.stringify(context)}` },
  ]);
  return { ...normalizeInsight(extractJsonObject(result.text)), model: result.model };
}

const SQL_SCHEMA = `
SQLite tables:
scored_patients(patient_rank, predicted_risk, observed_event, gender, age, age_group, hypertension, heart_disease, residence_type, avg_glucose_level, glucose_band, bmi, bmi_band, smoking_status, error_group)
model_metrics(model, roc_auc, pr_auc, sensitivity, specificity, precision, top10_precision, top10_recall, top10_events_reached)
subgroup_metrics(population, patient_count, event_rate, roc_auc, pr_auc, recall_top_10, global_top_10_share)
capacity_metrics(capacity_fraction, patients_contacted, events_found, precision_at_k, recall_at_k, lift, minimum_score)
model_coefficients(feature, coefficient, odds_ratio)
`;

function referencedTables(sql) {
  const cteNames = new Set();
  for (const match of sql.matchAll(/(?:\bwith\b|,)\s*([a-z_][a-z0-9_]*)\s+as\s*\(/gi)) {
    cteNames.add(match[1].toLowerCase());
  }
  const tables = [];
  for (const match of sql.matchAll(/\b(?:from|join)\s+([a-z_][a-z0-9_]*)/gi)) {
    const name = match[1].toLowerCase();
    if (!cteNames.has(name)) tables.push(name);
  }
  return tables;
}

export function validateSql(candidate) {
  let sql = String(candidate ?? "").trim();
  if (!sql || sql.length > MAX_SQL_CHARS) throw new HttpError(422, "The generated query was empty or too long.");
  sql = sql.replace(/;\s*$/, "").trim();
  if (sql.includes(";") || /--|\/\*|\*\//.test(sql)) {
    throw new HttpError(422, "Multiple statements and SQL comments are not allowed.");
  }
  if (/[`"\[\]]/.test(sql) || /\b(union|intersect|except|over|sqlite_|d1_)\b/i.test(sql)) {
    throw new HttpError(422, "The generated query used an unsupported SQL construct.");
  }
  if (!/^(select|with)\b/i.test(sql)) throw new HttpError(422, "Only read-only SELECT queries are allowed.");
  if (/\b(insert|update|delete|drop|alter|attach|detach|pragma|vacuum|replace|create|reindex|analyze|transaction|begin|commit|rollback|load_extension)\b/i.test(sql)) {
    throw new HttpError(422, "The generated query contained a blocked operation.");
  }
  const tables = referencedTables(sql);
  if (!tables.length || tables.some((table) => !ALLOWED_TABLES.has(table))) {
    throw new HttpError(422, "The generated query referenced an unapproved table.");
  }
  if (tables.includes("scored_patients")) {
    if (!/\bcount\s*\(/i.test(sql)) throw new HttpError(422, "Patient queries must be aggregate queries with a patient count.");
    if (/\bpatient_rank\b/i.test(sql) || /select\s+\*/i.test(sql)) {
      throw new HttpError(422, "Patient-level rows cannot be returned.");
    }
  }
  const limit = sql.match(/\blimit\s+(\d+)/i);
  if (limit && Number(limit[1]) > 50) throw new HttpError(422, "Query results are limited to 50 aggregate rows.");
  if (!limit) sql = `${sql} LIMIT 50`;
  return sql;
}

export function enforceAggregatePrivacy(rows) {
  for (const row of rows) {
    const countKey = Object.keys(row).find((key) => /^(patient_count|patients|n)$/i.test(key));
    if (countKey && finiteNumber(row[countKey], 0) < 10) {
      throw new HttpError(422, "The result was suppressed because an aggregate group contained fewer than 10 patients.");
    }
  }
  return rows;
}

async function generateSql(env, question) {
  const system = `Translate the user's analytical question into one SQLite query using only the supplied schema. Return JSON only with keys title and sql. SQL must be one SELECT or WITH query. Every query that references scored_patients must return aggregates only and include COUNT(*) AS patient_count. Never select patient_rank or return raw patient rows. Use observed_event only as an observed retrospective label, not a diagnosis. Add clear aliases. Do not use comments or semicolons.\n${SQL_SCHEMA}`;
  let repair = "";
  let lastError;

  for (let attempt = 0; attempt < 2; attempt += 1) {
    const result = await runModel(env, [
      { role: "system", content: system },
      {
        role: "user",
        content: `Untrusted user question: ${question}${repair}`,
      },
    ], 600);
    try {
      const parsed = extractJsonObject(result.text);
      return {
        title: textValue(parsed.title, 100) || "Model results",
        sql: validateSql(parsed.sql),
        model: result.model,
      };
    } catch (error) {
      lastError = error;
      repair = `\nThe previous generated response failed validation: ${textValue(error?.message, 240)}. Correct it. If scored_patients is referenced, include COUNT(*) AS patient_count. Previous response: ${textValue(result.text, MAX_SQL_CHARS)}`;
    }
  }
  throw lastError;
}

function normalizeAnswer(value) {
  return {
    answer: textValue(value?.answer, 900),
    key_points: normalizeList(value?.key_points, 4),
    caveat: textValue(value?.caveat, 280) || "Predictive associations support outreach planning and are not causal or clinical conclusions.",
  };
}

async function summarizeRows(env, question, sql, rows) {
  const result = await runModel(env, [
    {
      role: "system",
      content: "Answer the question using only the SQL result supplied. Be concise and operational. Do not add facts, diagnoses, causal claims, or treatment advice. Return JSON only with keys answer, key_points, and caveat.",
    },
    { role: "user", content: JSON.stringify({ question, sql, rows }) },
  ], 650);
  return { ...normalizeAnswer(extractJsonObject(result.text)), model: result.model };
}

async function answerContextQuestion(env, question, scope, context) {
  const contextLabel = scope === "caregiver"
    ? "caregiver-match"
    : scope === "outreach"
      ? "outreach-plan and deterministic optimizer"
      : "patient-ranking";
  const outreachRule = scope === "outreach"
    ? " Do not calculate a new threshold or claim a universally best threshold; explain only the supplied current and recommended operating points under the stated objective and constraints."
    : "";
  const result = await runModel(env, [
    {
      role: "system",
      content: `Answer only from the supplied ${contextLabel} context. If the context does not answer the question, say so. Do not diagnose, prescribe, infer missing facts, or claim causality.${outreachRule} Return JSON only with keys answer, key_points, and caveat.`,
    },
    { role: "user", content: JSON.stringify({ question, context }) },
  ], 650);
  return { ...normalizeAnswer(extractJsonObject(result.text)), model: result.model };
}

async function handleInsight(request, env) {
  const body = await readJson(request);
  const kind = ["patient", "caregiver", "outreach"].includes(body.kind) ? body.kind : null;
  if (!kind) throw new HttpError(400, "Insight kind must be patient, caregiver, or outreach.");
  const context = kind === "patient"
    ? patientContext(body.context)
    : kind === "caregiver"
      ? caregiverContext(body.context)
      : outreachContext(body.context);
  const insight = await generateInsight(env, kind, context);
  return jsonResponse(request, { ...insight, grounded: true });
}

async function handleAsk(request, env) {
  const body = await readJson(request);
  const question = textValue(body.question, MAX_QUESTION_CHARS);
  if (!question || question.length < 3) throw new HttpError(400, "Enter a question before asking the AI assistant.");
  const scope = ["patient", "caregiver", "outreach", "population"].includes(body.scope) ? body.scope : "population";
  if (scope !== "population") {
    const context = scope === "patient"
      ? patientContext(body.context)
      : scope === "caregiver"
        ? caregiverContext(body.context)
        : outreachContext(body.context);
    const answer = await answerContextQuestion(env, question, scope, context);
    return jsonResponse(request, { ...answer, mode: "grounded-context" });
  }
  const generated = await generateSql(env, question);
  const result = await env.DB.prepare(generated.sql).all();
  const rows = enforceAggregatePrivacy(result.results || []);
  const answer = await summarizeRows(env, question, generated.sql, rows);
  return jsonResponse(request, {
    ...answer,
    title: generated.title,
    sql: generated.sql,
    rows,
    mode: "text-to-sql",
    model: answer.model,
    sql_model: generated.model,
  });
}

async function health(request, env) {
  let database = { ready: false, rows: 0 };
  try {
    const result = await env.DB.prepare("SELECT COUNT(*) AS patient_count FROM scored_patients").first();
    database = { ready: true, rows: Number(result?.patient_count || 0) };
  } catch {
    database = { ready: false, rows: 0 };
  }
  return jsonResponse(request, {
    ok: true,
    service: "care-risk-outreach-ai",
    models: AI_MODELS,
    ai_binding: Boolean(env.AI),
    database,
  });
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(request) });
    }
    try {
      assertAllowedOrigin(request);
      const url = new URL(request.url);
      if (request.method === "GET" && url.pathname === "/api/health") return health(request, env);
      if (request.method !== "POST") throw new HttpError(404, "API route not found.");
      checkRateLimit(request);
      if (url.pathname === "/api/insight") return await handleInsight(request, env);
      if (url.pathname === "/api/ask") return await handleAsk(request, env);
      throw new HttpError(404, "API route not found.");
    } catch (error) {
      const status = error instanceof HttpError ? error.status : 500;
      const message = status === 500 ? "The AI service could not complete the request." : error.message;
      if (status >= 500) console.error(error);
      return jsonResponse(request, { error: message, grounded: false }, status);
    }
  },
};
