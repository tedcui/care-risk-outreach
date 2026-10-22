import assert from "node:assert/strict";
import test from "node:test";

import {
  enforceAggregatePrivacy,
  extractJsonObject,
  validateSql,
} from "../src/index.js";
import worker from "../src/index.js";

function postRequest(path, body, origin = "http://127.0.0.1:8000") {
  return new Request(`https://worker.test${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Origin: origin },
    body: JSON.stringify(body),
  });
}

test("extractJsonObject accepts fenced JSON", () => {
  const result = extractJsonObject('```json\n{"answer":"grounded"}\n```');
  assert.equal(result.answer, "grounded");
});

test("validateSql accepts aggregate patient queries", () => {
  const sql = validateSql(
    "SELECT age_group, COUNT(*) AS patient_count, AVG(predicted_risk) AS average_risk FROM scored_patients GROUP BY age_group"
  );
  assert.match(sql, /LIMIT 50$/);
});

test("validateSql accepts aggregate metric tables", () => {
  const sql = validateSql("SELECT model, pr_auc FROM model_metrics ORDER BY pr_auc DESC");
  assert.match(sql, /^SELECT model/);
});

test("validateSql rejects patient-level rows", () => {
  assert.throws(() => validateSql("SELECT * FROM scored_patients"), /aggregate queries/);
  assert.throws(
    () => validateSql("SELECT patient_rank, COUNT(*) AS patient_count FROM scored_patients GROUP BY patient_rank"),
    /Patient-level rows/
  );
});

test("validateSql rejects writes and multiple statements", () => {
  assert.throws(() => validateSql("DELETE FROM scored_patients"), /SELECT queries/);
  assert.throws(
    () => validateSql("SELECT COUNT(*) AS patient_count FROM scored_patients; DROP TABLE scored_patients"),
    /Multiple statements/
  );
});

test("validateSql rejects unapproved tables", () => {
  assert.throws(() => validateSql("SELECT COUNT(*) AS patient_count FROM users"), /unapproved table/);
});

test("validateSql rejects quoted identifiers, unions, and window functions", () => {
  assert.throws(
    () => validateSql('SELECT COUNT(*) AS patient_count FROM "scored_patients"'),
    /unsupported SQL construct/
  );
  assert.throws(
    () => validateSql("SELECT COUNT(*) AS patient_count FROM scored_patients UNION SELECT COUNT(*) FROM model_metrics"),
    /unsupported SQL construct/
  );
  assert.throws(
    () => validateSql("SELECT age, COUNT(*) OVER () AS patient_count FROM scored_patients"),
    /unsupported SQL construct/
  );
});

test("enforceAggregatePrivacy suppresses small groups", () => {
  assert.throws(() => enforceAggregatePrivacy([{ age_group: "Example", patient_count: 4 }]), /fewer than 10/);
  assert.deepEqual(enforceAggregatePrivacy([{ patient_count: 20 }]), [{ patient_count: 20 }]);
});

test("patient insight endpoint returns grounded structured content", async () => {
  const env = {
    AI: {
      run: async () => ({
        response: JSON.stringify({
          headline: "Early outreach candidate",
          summary: "The patient is highly ranked by the model.",
          supporting_factors: ["Older age", "Hypertension"],
          watch_outs: ["Confirm context"],
        }),
      }),
    },
  };
  const response = await worker.fetch(
    postRequest("/api/insight", {
      kind: "patient",
      context: { predicted_risk: 0.42, age: 72, hypertension: true },
    }),
    env
  );
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.grounded, true);
  assert.equal(payload.headline, "Early outreach candidate");
  assert.equal(response.headers.get("Access-Control-Allow-Origin"), "http://127.0.0.1:8000");
});

test("outreach insight endpoint explains a deterministic recommendation", async () => {
  let modelPrompt = "";
  const env = {
    AI: {
      run: async (_model, options) => {
        modelPrompt = options.messages.map((message) => message.content).join("\n");
        return {
          response: JSON.stringify({
            headline: "Feasible outreach plan",
            summary: "The recommended operating point satisfies the supplied capacity and recall constraints.",
            supporting_factors: ["Capacity requirement met", "Recall target met"],
            watch_outs: ["Validate operational assumptions"],
          }),
        };
      },
    },
  };
  const response = await worker.fetch(
    postRequest("/api/insight", {
      kind: "outreach",
      context: {
        population: "All patients",
        patients_contacted: 511,
        objective: "max_events",
        active_constraints: ["No more than 550 contacts", "Recall at least 40%"],
        recommendation: {
          feasible: true,
          outreach_percent: 10,
          patients_contacted: 511,
          events_reached: 105,
          recall: 0.422,
          precision: 0.205,
          lift: 4.2,
        },
      },
    }),
    env
  );
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.grounded, true);
  assert.equal(payload.headline, "Feasible outreach plan");
  assert.match(modelPrompt, /deterministic optimizer recommendation/i);
  assert.match(modelPrompt, /outreach operating plan/i);
});

test("outreach questions use grounded context without querying D1", async () => {
  let modelCalls = 0;
  const env = {
    AI: {
      run: async () => {
        modelCalls += 1;
        return {
          response: JSON.stringify({
            answer: "The supplied feasible plan contacts 511 patients at the 10 percent operating point.",
            key_points: ["The recommendation is conditional on the selected constraints."],
            caveat: "This is an operational planning recommendation, not a clinical conclusion.",
          }),
        };
      },
    },
    DB: {
      prepare() {
        throw new Error("D1 should not be queried for outreach-context questions.");
      },
    },
  };
  const response = await worker.fetch(
    postRequest("/api/ask", {
      scope: "outreach",
      question: "What is the best threshold under these constraints?",
      context: {
        population: "All patients",
        objective: "max_events",
        recommendation: { feasible: true, outreach_percent: 10, patients_contacted: 511 },
      },
    }),
    env
  );
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.mode, "grounded-context");
  assert.equal(modelCalls, 1);
});

test("population question generates SQL, queries D1, and summarizes results", async () => {
  const modelResponses = [
    {
      response: JSON.stringify({
        title: "Patients by age group",
        sql: "SELECT age_group, COUNT(*) AS patient_count, AVG(predicted_risk) AS average_risk FROM scored_patients GROUP BY age_group",
      }),
    },
    {
      response: JSON.stringify({
        answer: "Older patients have the higher average predicted risk in this extract.",
        key_points: ["Both groups exceed the reporting minimum."],
        caveat: "This is a predictive association.",
      }),
    },
  ];
  let executedSql = "";
  const env = {
    AI: { run: async () => modelResponses.shift() },
    DB: {
      prepare(sql) {
        executedSql = sql;
        return {
          all: async () => ({
            results: [
              { age_group: "Age <65", patient_count: 3800, average_risk: 0.02 },
              { age_group: "Age >=65", patient_count: 1309, average_risk: 0.13 },
            ],
          }),
        };
      },
    },
  };
  const response = await worker.fetch(
    postRequest("/api/ask", {
      scope: "population",
      question: "Compare average predicted risk by age group.",
    }),
    env
  );
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(payload.mode, "text-to-sql");
  assert.equal(payload.rows.length, 2);
  assert.match(executedSql, /COUNT\(\*\) AS patient_count/i);
  assert.match(payload.sql, /LIMIT 50$/);
});

test("population question repairs SQL that fails the aggregate guardrail", async () => {
  const modelResponses = [
    {
      response: JSON.stringify({
        title: "Average risk by age group",
        sql: "SELECT age_group, AVG(predicted_risk) AS average_risk FROM scored_patients GROUP BY age_group",
      }),
    },
    {
      response: JSON.stringify({
        title: "Average risk by age group",
        sql: "SELECT age_group, COUNT(*) AS patient_count, AVG(predicted_risk) AS average_risk FROM scored_patients GROUP BY age_group",
      }),
    },
    {
      response: JSON.stringify({
        answer: "Average predicted risk is higher in the older age group.",
        key_points: ["The result is aggregate."],
        caveat: "This is a predictive association.",
      }),
    },
  ];
  let modelCalls = 0;
  const env = {
    AI: {
      run: async () => {
        modelCalls += 1;
        return modelResponses.shift();
      },
    },
    DB: {
      prepare: () => ({
        all: async () => ({
          results: [{ age_group: "Age >=65", patient_count: 1309, average_risk: 0.13 }],
        }),
      }),
    },
  };
  const response = await worker.fetch(
    postRequest("/api/ask", {
      scope: "population",
      question: "Compare average predicted risk by age group.",
    }),
    env
  );
  const payload = await response.json();
  assert.equal(response.status, 200);
  assert.equal(modelCalls, 3);
  assert.match(payload.sql, /COUNT\(\*\) AS patient_count/i);
});

test("API rejects unapproved browser origins", async () => {
  const response = await worker.fetch(
    postRequest("/api/ask", { scope: "population", question: "Count patients" }, "https://example.com"),
    {}
  );
  assert.equal(response.status, 403);
});
