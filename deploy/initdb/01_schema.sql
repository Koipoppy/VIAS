-- VIAS pilot metadata schema (PostgreSQL 14)
-- Run automatically by the postgres container on first start.

CREATE TABLE IF NOT EXISTS cases (
    case_id           TEXT PRIMARY KEY,
    source_case_id    TEXT,
    synthetic         BOOLEAN NOT NULL DEFAULT TRUE,
    not_real_identity BOOLEAN NOT NULL DEFAULT TRUE,
    ground_truth      JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evidence_units (
    evidence_id       TEXT PRIMARY KEY,
    case_id           TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
    schema_version    TEXT NOT NULL DEFAULT '0.1',
    modality          TEXT NOT NULL,
    source_type       TEXT NOT NULL,
    source_name       TEXT,
    content           JSONB,
    gender_probs      JSONB,
    age_probs         JSONB,
    credibility       JSONB,
    credibility_score DOUBLE PRECISION,
    conflict_flag     BOOLEAN NOT NULL DEFAULT FALSE,
    conflict_degree   DOUBLE PRECISION NOT NULL DEFAULT 0,
    injected          BOOLEAN NOT NULL DEFAULT FALSE,
    synthetic         BOOLEAN NOT NULL DEFAULT TRUE,
    pii_redacted      BOOLEAN NOT NULL DEFAULT TRUE,
    payload           JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_evidence_case ON evidence_units(case_id);
CREATE INDEX IF NOT EXISTS idx_evidence_source ON evidence_units(source_type);

CREATE TABLE IF NOT EXISTS case_results (
    case_id             TEXT PRIMARY KEY REFERENCES cases(case_id) ON DELETE CASCADE,
    method              TEXT,
    predicted_gender    TEXT,
    predicted_age_bucket TEXT,
    gender_posterior    JSONB,
    age_posterior       JSONB,
    conflict_detected   BOOLEAN,
    max_conflict        DOUBLE PRECISION,
    payload             JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS inference_jobs (
    job_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id    TEXT,
    status     TEXT NOT NULL DEFAULT 'queued',
    error      TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON inference_jobs(status);
