/**
 * The dashboard's single data entry point.
 *
 * This is the only module that reads the generated payload, and it is the only
 * place in the front end where a data path appears. Everything rendered on the
 * page descends from the object returned here, which is produced by
 * scripts/export_dashboard_data.py from the approved analytics modules and
 * verified by scripts/validate_dashboard_data.py before deployment.
 *
 * scripts/check_no_hardcoded_numbers.py enforces that this file holds the only
 * such reference, so there is exactly one place to audit.
 */

const DATA_URL = new URL("../../data/dashboard_data.json", import.meta.url);

const SUPPORTED_SCHEMA = 2;

/** Trading days past the latest observation before the data reads as stale. */
const STALE_AFTER_DAYS = 10;

export class DataError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = "DataError";
    this.cause = cause;
  }
}

/**
 * Fetch and lightly validate the payload.
 *
 * Validation here is a guard against serving a payload the page cannot render
 * (wrong schema, missing blocks) - the numerical validation happens in CI, in
 * scripts/validate_dashboard_data.py, where it can fail the build.
 */
export async function loadDashboardData() {
  let response;
  try {
    response = await fetch(DATA_URL, { cache: "no-cache" });
  } catch (error) {
    throw new DataError(
      "Could not reach the analytics payload. Check your connection and reload.",
      error
    );
  }

  if (!response.ok) {
    throw new DataError(
      `The analytics payload responded with ${response.status}. The pipeline may not have published yet.`
    );
  }

  let payload;
  try {
    payload = await response.json();
  } catch (error) {
    throw new DataError("The analytics payload could not be parsed.", error);
  }

  assertShape(payload);
  return payload;
}

function assertShape(payload) {
  for (const key of ["meta", "windows", "series", "signals", "monte_carlo"]) {
    if (!payload || !payload[key]) {
      throw new DataError(`The analytics payload is missing its "${key}" block.`);
    }
  }

  const version = payload.meta.schema_version;
  if (version !== SUPPORTED_SCHEMA) {
    throw new DataError(
      `This page reads payload schema ${SUPPORTED_SCHEMA} but received ${version}. ` +
        `Redeploy the dashboard so the page and the payload match.`
    );
  }

  if (!payload.windows[payload.meta.default_window]) {
    throw new DataError("The payload's default window is not present in its window set.");
  }
}

/**
 * Classify freshness for the status pill. The pipeline runs weekly after the
 * Friday US close, so a gap beyond ten days means a refresh has been missed
 * rather than a holiday having shifted the last session.
 */
export function pipelineHealth(meta, daysSinceLatest) {
  if (meta.pipeline_status !== "ok") {
    return { level: "error", label: "Pipeline error" };
  }
  if (daysSinceLatest === null || daysSinceLatest === undefined) {
    return { level: "ok", label: "Pipeline healthy" };
  }
  if (daysSinceLatest > STALE_AFTER_DAYS) {
    return {
      level: "stale",
      label: `Data ${daysSinceLatest} days old`,
    };
  }
  return { level: "ok", label: "Pipeline healthy" };
}

export { STALE_AFTER_DAYS };
