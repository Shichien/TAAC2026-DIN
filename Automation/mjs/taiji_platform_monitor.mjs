#!/usr/bin/env node
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const TAIJI_ORIGIN = "https://taiji.algo.qq.com";
const DEFAULT_CREATOR = "ams_2026_1029731869646210001";
const DEFAULT_COOKIE_FILE = "taiji-output/secrets/taiji-cookie.txt";
const DEFAULT_STATE_FILE = "taiji-output/monitor/taiji-watch-state.json";
const DEFAULT_SUMMARY_FILE = "taiji-output/monitor/last-summary.json";
const DEFAULT_PAGE_SIZE = 200;
const DEFAULT_INSTANCE_PAGE_SIZE = 20;
function usage() {
  return `Usage:
  node Automation/mjs/taiji_platform_monitor.mjs [options]

Options:
  --cookie-file <file>       Cookie header or Copy-as-cURL text.
  --creator <creator>        Training job creator. Default: ${DEFAULT_CREATOR}
  --state-file <file>        Persistent watch state. Default: ${DEFAULT_STATE_FILE}
  --summary-file <file>      Latest summary output. Default: ${DEFAULT_SUMMARY_FILE}
  --watch-eval <id>          Eval task id to monitor. Repeatable.
  --watch-training <taskId>  Training task id to monitor. Repeatable.
  --seed-training <taskId>   Scope sync-active to jobs created at or after this training task.
  --sync-active              Add all current non-terminal training jobs for the creator into the watch set.
  --apply                    Actually restart failed jobs with the platform start API.
  --page-size <n>            Training job page size. Default: ${DEFAULT_PAGE_SIZE}
  --instance-page-size <n>   Instance page size. Default: ${DEFAULT_INSTANCE_PAGE_SIZE}
  --help                     Show this help.

Dry-run is the default. It inspects status and writes summary/state, but does not restart jobs.`;
}

function parseArgs(argv) {
  const args = {
    cookieFile: DEFAULT_COOKIE_FILE,
    creator: DEFAULT_CREATOR,
    stateFile: DEFAULT_STATE_FILE,
    summaryFile: DEFAULT_SUMMARY_FILE,
    watchEval: [],
    watchTraining: [],
    seedTraining: null,
    syncActive: false,
    apply: false,
    pageSize: DEFAULT_PAGE_SIZE,
    instancePageSize: DEFAULT_INSTANCE_PAGE_SIZE,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      args.help = true;
      continue;
    }
    if (arg === "--sync-active") {
      args.syncActive = true;
      continue;
    }
    if (arg === "--apply") {
      args.apply = true;
      continue;
    }
    if (arg === "--watch-eval") {
      const value = argv[i + 1];
      if (!value || value.startsWith("--")) throw new Error("Missing value for --watch-eval");
      args.watchEval.push(String(value));
      i += 1;
      continue;
    }
    if (arg === "--watch-training") {
      const value = argv[i + 1];
      if (!value || value.startsWith("--")) throw new Error("Missing value for --watch-training");
      args.watchTraining.push(String(value));
      i += 1;
      continue;
    }
    if (arg === "--seed-training") {
      const value = argv[i + 1];
      if (!value || value.startsWith("--")) throw new Error("Missing value for --seed-training");
      args.seedTraining = String(value);
      i += 1;
      continue;
    }
    if (arg.startsWith("--")) {
      const key = arg.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      const value = argv[i + 1];
      if (!value || value.startsWith("--")) throw new Error(`Missing value for ${arg}`);
      if (key === "pageSize" || key === "instancePageSize") args[key] = Number(value);
      else args[key] = value;
      i += 1;
      continue;
    }
    throw new Error(`Unexpected argument: ${arg}`);
  }

  return args;
}

function assertSafeRelativePath(targetPath) {
  if (!path.isAbsolute(targetPath) && String(targetPath).split(/[\\/]+/).includes("..")) {
    throw new Error("Relative paths must not contain '..'.");
  }
}

function resolveRepoPath(targetPath) {
  assertSafeRelativePath(targetPath);
  return path.isAbsolute(targetPath) ? targetPath : path.resolve(targetPath);
}

async function readJsonIfExists(filePath, fallbackValue) {
  try {
    return JSON.parse(await readFile(filePath, "utf8"));
  } catch {
    return fallbackValue;
  }
}

async function writeJsonFile(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function extractCookieHeader(fileContent) {
  const text = fileContent.trim();
  const headerLine = text.match(/^cookie:\s*(.+)$/im);
  if (headerLine) return headerLine[1].trim();
  const curlHeader = text.match(/(?:-H|--header)\s+(['"])cookie:\s*([\s\S]*?)\1/i);
  if (curlHeader) return curlHeader[2].trim();
  return text.replace(/^cookie:\s*/i, "").trim();
}

function taijiHeaders(cookieHeader, refererPath) {
  return {
    accept: "application/json, text/plain, */*",
    "content-type": "application/json",
    cookie: cookieHeader,
    referer: `${TAIJI_ORIGIN}${refererPath}`,
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147 Safari/537.36",
  };
}

async function fetchJson(cookieHeader, endpoint, options = {}) {
  const url = new URL(endpoint, TAIJI_ORIGIN);
  if (options.params) {
    for (const [key, value] of Object.entries(options.params)) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    }
  }

  const method = options.method || "GET";
  const response = await fetch(url, {
    method,
    headers: taijiHeaders(cookieHeader, options.refererPath || "/training"),
    body: options.body === undefined ? undefined : JSON.stringify(options.body),
  });
  const text = await response.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${url.pathname}: ${String(text).slice(0, 500)}`);
  }
  return body;
}

function extractRows(response) {
  if (Array.isArray(response?.data)) return response.data;
  if (Array.isArray(response?.data?.data)) return response.data.data;
  if (Array.isArray(response?.data?.list)) return response.data.list;
  if (Array.isArray(response?.list)) return response.list;
  if (Array.isArray(response?.results)) return response.results;
  return [];
}

function extractTotal(response) {
  return (
    response?.data?.totalCount ??
    response?.data?.total ??
    response?.data?.count ??
    response?.totalCount ??
    response?.total ??
    response?.count ??
    null
  );
}

function isTerminalJob(job) {
  const status = String(job?.status ?? "").toUpperCase();
  const jzStatus = String(job?.jzStatus ?? "").toUpperCase();
  return jzStatus === "END" || ["SUCCEED", "FAILED", "KILLED", "CANCELED", "CANCELLED"].includes(status);
}

function isSuccessfulJob(job) {
  const status = String(job?.status ?? "").toUpperCase();
  return status === "SUCCEED";
}

function isRunningLikeJob(job) {
  if (isTerminalJob(job)) return false;
  const jzStatus = String(job?.jzStatus ?? "").toUpperCase();
  return jzStatus !== "JOB_DELETE";
}

function parseJobCreateTime(job) {
  const rawValue = job?.createTime ?? job?.createdAt ?? job?.create_time ?? job?.gmtCreate ?? null;
  if (!rawValue) return null;
  const time = Date.parse(String(rawValue));
  return Number.isFinite(time) ? time : null;
}

function defaultState(args) {
  return {
    version: 1,
    creator: args.creator,
    scope: null,
    training: {},
    evals: {},
  };
}

function ensureTrainingEntry(state, taskId) {
  if (!state.training[taskId]) {
    state.training[taskId] = {
      taskId,
      firstSeenAt: new Date().toISOString(),
      done: false,
    };
  }
  return state.training[taskId];
}

function ensureEvalEntry(state, evalId) {
  if (!state.evals[evalId]) {
    state.evals[evalId] = {
      evalId,
      firstSeenAt: new Date().toISOString(),
      done: false,
    };
  }
  return state.evals[evalId];
}

function buildTrainingScope(args, trainingJobs) {
  if (!args.seedTraining) return null;
  const seedJob = trainingJobs.find((job) => String(job?.taskID ?? "") === args.seedTraining);
  if (!seedJob) {
    throw new Error(`Seed training task not found for creator ${args.creator}: ${args.seedTraining}`);
  }
  const minCreateTime = parseJobCreateTime(seedJob);
  if (minCreateTime == null) {
    throw new Error(`Seed training task is missing a parseable create time: ${args.seedTraining}`);
  }
  return {
    kind: "seed_training",
    seedTaskId: args.seedTraining,
    minCreateTime,
    minCreateTimeIso: new Date(minCreateTime).toISOString(),
  };
}

function isJobWithinScope(job, scope) {
  if (!scope) return true;
  if (scope.kind === "seed_training") {
    const createTime = parseJobCreateTime(job);
    return createTime != null && createTime >= scope.minCreateTime;
  }
  return true;
}

function shouldSyncTrainingJob(job, scope) {
  if (!isJobWithinScope(job, scope)) return false;
  if (scope) return !isSuccessfulJob(job);
  return !isTerminalJob(job);
}

async function fetchTrainingJobs(cookieHeader, pageSize, creator) {
  const jobs = [];
  for (let pageNum = 0; ; pageNum += 1) {
    const response = await fetchJson(cookieHeader, "/taskmanagement/api/v1/webtasks/external/task", {
      params: { pageNum, pageSize },
      refererPath: "/training",
    });
    const rows = extractRows(response);
    jobs.push(...rows);
    const total = extractTotal(response);
    if (!rows.length || rows.length < pageSize || (total != null && jobs.length >= total)) break;
  }
  return jobs.filter((job) => String(job?.creator ?? "") === creator);
}

async function fetchJobInstances(cookieHeader, taskId, pageSize) {
  const instances = [];
  for (let pageNum = 0; ; pageNum += 1) {
    const response = await fetchJson(cookieHeader, "/taskmanagement/api/v1/instances/list", {
      method: "POST",
      body: { desc: true, orderBy: "create", task_id: taskId, page: pageNum, size: pageSize },
      refererPath: "/training",
    });
    const rows = extractRows(response);
    instances.push(...rows);
    const total = extractTotal(response);
    if (!rows.length || rows.length < pageSize || (total != null && instances.length >= total)) break;
  }
  return instances;
}

async function startTrainingTask(cookieHeader, taskId) {
  return fetchJson(cookieHeader, `/taskmanagement/api/v1/webtasks/${taskId}/start`, {
    method: "POST",
    body: {},
    refererPath: "/training",
  });
}

async function fetchEvalTask(cookieHeader, evalId) {
  return fetchJson(cookieHeader, `/aide/api/evaluation_tasks/${evalId}/`, {
    refererPath: "/evaluation",
  });
}

function isFinishedEval(evalTask) {
  const status = String(evalTask?.status ?? "").toLowerCase();
  if (evalTask?.score !== null && evalTask?.score !== undefined) return true;
  return ["success", "succeed", "finished", "done"].includes(status);
}

function isFailedEval(evalTask) {
  const status = String(evalTask?.status ?? "").toLowerCase();
  return ["failed", "error", "cancelled", "canceled", "terminated"].includes(status) || Boolean(evalTask?.error_msg);
}

async function inspectTrainingTask(cookieHeader, job, stateEntry, args) {
  stateEntry.jobInternalId = job?.id ?? stateEntry.jobInternalId ?? null;
  stateEntry.name = job?.name ?? stateEntry.name ?? "";
  stateEntry.lastSeenAt = new Date().toISOString();
  stateEntry.lastStatus = job?.status ?? null;
  stateEntry.lastJzStatus = job?.jzStatus ?? null;

  const instances = await fetchJobInstances(cookieHeader, stateEntry.taskId, args.instancePageSize);
  const latestInstance = instances[0] ?? null;
  stateEntry.latestInstanceId = latestInstance?.id ?? null;
  stateEntry.instanceCount = instances.length;

  const result = {
    type: "training",
    taskId: stateEntry.taskId,
    name: stateEntry.name,
    jobInternalId: stateEntry.jobInternalId,
    status: stateEntry.lastStatus,
    jzStatus: stateEntry.lastJzStatus,
    latestInstanceId: stateEntry.latestInstanceId,
    outcome: "pending",
    restarted: false,
  };

  if (isSuccessfulJob(job)) {
    stateEntry.done = true;
    stateEntry.doneReason = "success";
    result.outcome = "done";
    return result;
  }

  if (isRunningLikeJob(job)) {
    stateEntry.done = false;
    result.outcome = "running";
    return result;
  }

  stateEntry.lastInstanceInnerStatus = latestInstance.inner_status ?? latestInstance.status ?? null;
  result.latestInstanceInnerStatus = stateEntry.lastInstanceInnerStatus;

  const failedStatus = String(job?.status ?? "").toUpperCase() === "FAILED";
  if (!failedStatus) {
    stateEntry.done = false;
    stateEntry.needsAttention = true;
    stateEntry.lastIssue = "terminal_non_failed";
    result.outcome = "blocked";
    result.issue = "terminal_non_failed";
    return result;
  }

  if (stateEntry.lastRestartedFromInstanceId !== latestInstance?.id) {
    result.outcome = args.apply ? "restart_submitted" : "restart_needed";
    stateEntry.lastIssue = "failed";
    if (args.apply) {
      const startResponse = await startTrainingTask(cookieHeader, stateEntry.taskId);
      stateEntry.lastRestartedFromInstanceId = latestInstance?.id ?? null;
      stateEntry.lastRestartAt = new Date().toISOString();
      stateEntry.lastRestartResponse = startResponse;
      result.restarted = true;
    }
  } else {
    result.outcome = "failed_already_restarted";
  }

  stateEntry.done = false;
  return result;
}

async function inspectEvalTask(cookieHeader, stateEntry) {
  const evalTask = await fetchEvalTask(cookieHeader, stateEntry.evalId);
  stateEntry.name = evalTask?.name ?? stateEntry.name ?? "";
  stateEntry.lastSeenAt = new Date().toISOString();
  stateEntry.lastStatus = evalTask?.status ?? null;
  stateEntry.lastScore = evalTask?.score ?? null;
  stateEntry.lastError = evalTask?.error_msg ?? null;

  const result = {
    type: "eval",
    evalId: stateEntry.evalId,
    name: stateEntry.name,
    status: stateEntry.lastStatus,
    score: stateEntry.lastScore,
    outcome: "pending",
  };

  if (isFinishedEval(evalTask)) {
    stateEntry.done = true;
    stateEntry.doneReason = "success";
    result.outcome = "done";
    return result;
  }

  if (isFailedEval(evalTask)) {
    stateEntry.done = false;
    stateEntry.needsAttention = true;
    stateEntry.lastIssue = "eval_failed";
    result.outcome = "blocked";
    return result;
  }

  stateEntry.done = false;
  result.outcome = "running";
  return result;
}

function buildSummary(args, state, results) {
  const counts = {
    training: { watched: 0, done: 0, running: 0, restartNeeded: 0, restarted: 0, blocked: 0 },
    eval: { watched: 0, done: 0, running: 0, blocked: 0 },
  };

  for (const result of results) {
    if (result.type === "training") {
      counts.training.watched += 1;
      if (result.outcome === "done") counts.training.done += 1;
      else if (result.outcome === "running" || result.outcome === "failed_already_restarted") counts.training.running += 1;
      else if (result.outcome === "restart_needed") counts.training.restartNeeded += 1;
      else if (result.outcome === "restart_submitted") counts.training.restarted += 1;
      else if (result.outcome === "blocked") counts.training.blocked += 1;
    } else {
      counts.eval.watched += 1;
      if (result.outcome === "done") counts.eval.done += 1;
      else if (result.outcome === "running") counts.eval.running += 1;
      else if (result.outcome === "blocked") counts.eval.blocked += 1;
    }
  }

  const pendingResults = results.filter((result) => result.outcome !== "done");
  return {
    generatedAt: new Date().toISOString(),
    apply: args.apply,
    creator: state.creator,
    allDone: pendingResults.length === 0,
    counts,
    pendingResults,
    results,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }

  const cookiePath = resolveRepoPath(args.cookieFile);
  const statePath = resolveRepoPath(args.stateFile);
  const summaryPath = resolveRepoPath(args.summaryFile);
  const cookieHeader = extractCookieHeader(await readFile(cookiePath, "utf8"));
  const state = await readJsonIfExists(statePath, defaultState(args));
  state.creator = args.creator;

  for (const taskId of args.watchTraining) ensureTrainingEntry(state, taskId);
  for (const evalId of args.watchEval) ensureEvalEntry(state, evalId);

  const trainingJobs = await fetchTrainingJobs(cookieHeader, args.pageSize, args.creator);
  const jobsByTaskId = new Map(trainingJobs.map((job) => [String(job.taskID), job]));
  const trainingScope = buildTrainingScope(args, trainingJobs);
  state.scope = trainingScope;

  if (args.syncActive) {
    for (const job of trainingJobs) {
      if (shouldSyncTrainingJob(job, trainingScope)) {
        ensureTrainingEntry(state, String(job.taskID));
      }
    }
  }

  const results = [];

  for (const taskId of Object.keys(state.training)) {
    const stateEntry = ensureTrainingEntry(state, taskId);
    const job = jobsByTaskId.get(taskId);
    if (!job) {
      stateEntry.lastSeenAt = new Date().toISOString();
      stateEntry.lastIssue = "job_not_found";
      stateEntry.done = false;
      results.push({
        type: "training",
        taskId,
        name: stateEntry.name ?? "",
        status: null,
        jzStatus: null,
        outcome: "blocked",
        issue: "job_not_found",
      });
      continue;
    }
    results.push(await inspectTrainingTask(cookieHeader, job, stateEntry, args));
  }

  for (const evalId of Object.keys(state.evals)) {
    const stateEntry = ensureEvalEntry(state, evalId);
    results.push(await inspectEvalTask(cookieHeader, stateEntry));
  }

  const summary = buildSummary(args, state, results);
  await writeJsonFile(statePath, state);
  await writeJsonFile(summaryPath, summary);

  const lines = [
    `apply=${args.apply}`,
    `allDone=${summary.allDone}`,
    `training watched=${summary.counts.training.watched} done=${summary.counts.training.done} running=${summary.counts.training.running} restartNeeded=${summary.counts.training.restartNeeded} restarted=${summary.counts.training.restarted} blocked=${summary.counts.training.blocked}`,
    `eval watched=${summary.counts.eval.watched} done=${summary.counts.eval.done} running=${summary.counts.eval.running} blocked=${summary.counts.eval.blocked}`,
    `stateFile=${statePath}`,
    `summaryFile=${summaryPath}`,
  ];

  if (summary.pendingResults.length) {
    lines.push("pending:");
    for (const result of summary.pendingResults) {
      if (result.type === "training") {
        lines.push(`training ${result.taskId} ${result.name} outcome=${result.outcome} status=${result.status ?? "null"} jzStatus=${result.jzStatus ?? "null"} latestInstance=${result.latestInstanceId ?? "null"}`);
      } else {
        lines.push(`eval ${result.evalId} ${result.name} outcome=${result.outcome} status=${result.status ?? "null"} score=${result.score ?? "null"}`);
      }
    }
  }

  console.log(lines.join("\n"));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error?.stack || error);
    process.exitCode = 1;
  });
}
