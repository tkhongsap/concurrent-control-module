# ⚡ Mini-PRD: Azure OpenAI Load-Test Script

## Goal

Fire **200 total requests** to an Azure OpenAI deployment while ensuring **no more than 20 requests are in-flight at any moment**. Collect basic performance metrics for team review.

## Scope

* One self-contained testing module (Python or Node.js) runnable from the command line.
* No changes to production code or infrastructure.

## Inputs

Accept via CLI flags or environment variables:

* `AZURE_OPENAI_ENDPOINT`
* `AZURE_OPENAI_KEY`
* `AZURE_OPENAI_DEPLOYMENT`
* `--total` (default `200`)
* `--concurrency` (default `20`)

## Core Logic

1. Use an async semaphore (`asyncio.Semaphore` in Python or `p-limit` in Node) to cap parallel calls at **20**.
2. For each request:

   * Send a minimal prompt (e.g. `"Say hello"`).
   * On HTTP **429** or **503**, retry up to **3** times with exponential back-off.
3. Record start/end timestamps, status code, and any token-usage metadata returned by Azure OpenAI.

## Outputs

* Optional real-time console progress bar.
* End-of-run summary (JSON or CSV) containing:

  * Total requests
  * Successful vs. failed
  * Average latency
  * 95th-percentile latency
  * Retry counts

## Success Criteria

* Script completes all **200** calls.
* Peak concurrent requests never exceeds **20** (verified by internal counter).
* At least **100 %** of calls succeed without manual intervention.

## Non-Goals

* Production queuing or auto-scaling logic.
* Any UI beyond basic console output.

## Timeline

* **Day 1** – draft script.
* **Day 2** – peer review and minor fixes.
* Same week – run test and share metrics.

## Repository Setup

* New Git repo.
* `README.md` with usage examples.
* `requirements.txt` (Python) or `package.json` (Node).
* Example `.env` file.

## Nice-to-Have Extensions (defer if tight on time)

* Flag to vary prompt size for token-throughput testing.
* Metrics export compatible with Grafana/Prometheus.
* Dockerfile for reproducible runs.
