#!/usr/bin/env node
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const TAIJI_ORIGIN = "https://taiji.algo.qq.com";

function usage() {
  return `Usage:
  node Automation/mjs/publish_model.mjs --instance-id <id> --ckpt <dir> --name <name> --description <desc> --cookie-file <file> [--out <file>]`;
}

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") args.help = true;
    else if (arg.startsWith("--")) {
      const key = arg.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      const value = argv[i + 1];
      if (!value || value.startsWith("--")) throw new Error(`Missing value for ${arg}`);
      args[key] = value;
      i += 1;
    } else {
      throw new Error(`Unexpected argument: ${arg}`);
    }
  }
  return args;
}

function required(value, message) {
  if (value === undefined || value === null || value === "") throw new Error(message);
  return value;
}

function safeName(value) {
  return String(value || "published_model").replace(/[^a-zA-Z0-9_.-]/g, "_") || "published_model";
}

function resolveOutputPath(outFile, name) {
  const target = outFile || path.join("taiji-output", "publish", `${safeName(name)}_publish_result.json`);
  if (path.isAbsolute(target)) return target;
  return path.resolve(target);
}

function normalizeCkptPath(value) {
  return String(value || "")
    .replace(/\\/g, "/")
    .replace(/^\.\//, "")
    .replace(/^\/+/, "")
    .replace(/\/+/g, "/")
    .trim();
}

async function findLocalLogFilesForInstance(instanceId) {
  const logsRoot = path.resolve("taiji-output", "logs");
  let entries = [];
  try {
    entries = await readdir(logsRoot, { withFileTypes: true });
  } catch {
    return [];
  }

  const hits = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const candidate = path.join(logsRoot, entry.name, `${instanceId}.txt`);
    try {
      await readFile(candidate, "utf8");
      hits.push(candidate);
    } catch {
      // ignore missing instance logs
    }
  }
  return hits;
}

function extractCheckpointPathsFromLog(logText) {
  const matches = logText.matchAll(/Saved checkpoint to .*?[\\/]ckpt[\\/](.+?)[\\/]model\.pt/g);
  const ordered = [];
  const seen = new Set();
  for (const match of matches) {
    const rel = normalizeCkptPath(match[1]);
    if (!rel || seen.has(rel)) continue;
    seen.add(rel);
    ordered.push(rel);
  }
  return ordered;
}

async function resolveCheckpointArgument(instanceId, rawCkpt) {
  const requested = normalizeCkptPath(rawCkpt);
  if (!requested) {
    throw new Error("Missing --ckpt");
  }
  if (requested.includes("/")) {
    return {
      requested,
      resolved: requested,
      resolution: "explicit_relative_path",
      candidates: [],
      logFiles: [],
    };
  }

  const logFiles = await findLocalLogFilesForInstance(instanceId);
  if (!logFiles.length) {
    return {
      requested,
      resolved: requested,
      resolution: "unvalidated_leaf_without_local_logs",
      candidates: [],
      logFiles: [],
    };
  }

  const allCandidates = [];
  for (const logFile of logFiles) {
    const text = await readFile(logFile, "utf8");
    allCandidates.push(...extractCheckpointPathsFromLog(text));
  }

  const deduped = [...new Set(allCandidates)];
  const exact = deduped.find((item) => item === requested);
  if (exact) {
    return {
      requested,
      resolved: exact,
      resolution: "exact_log_match",
      candidates: deduped,
      logFiles,
    };
  }

  const basenameMatches = deduped.filter((item) => path.posix.basename(item) === requested);
  if (basenameMatches.length === 1) {
    return {
      requested,
      resolved: basenameMatches[0],
      resolution: "resolved_from_log_basename",
      candidates: deduped,
      logFiles,
    };
  }

  if (basenameMatches.length > 1) {
    throw new Error(
      `Checkpoint leaf ${requested} is ambiguous for instance ${instanceId}. ` +
      `Candidates: ${basenameMatches.join(", ")}`
    );
  }

  throw new Error(
    `Checkpoint leaf ${requested} was not found in local logs for instance ${instanceId}. ` +
    `Known saved checkpoints: ${deduped.join(", ")}`
  );
}

function extractCookieHeader(fileContent) {
  const text = fileContent.trim();
  const headerLine = text.match(/^cookie:\s*(.+)$/im);
  if (headerLine) return headerLine[1].trim();
  const curlHeader = text.match(/(?:-H|--header)\s+(['"])cookie:\s*([\s\S]*?)\1/i);
  if (curlHeader) return curlHeader[2].trim();
  return text.replace(/^cookie:\s*/i, "").trim();
}

async function fetchJson(cookieHeader, instanceId, payload) {
  const url = `${TAIJI_ORIGIN}/taskmanagement/api/v1/instances/external/${instanceId}/release_ckpt`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      accept: "application/json, text/plain, */*",
      "content-type": "application/json",
      cookie: cookieHeader,
      referer: `${TAIJI_ORIGIN}/training`,
      "user-agent": "Mozilla/5.0",
    },
    body: JSON.stringify(payload),
  });
  const text = await response.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${String(text).slice(0, 500)}`);
  return { status: response.status, body };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }

  const instanceId = required(args.instanceId, "Missing --instance-id");
  const ckptResolution = await resolveCheckpointArgument(
    instanceId,
    required(args.ckpt, "Missing --ckpt")
  );
  const payload = {
    name: required(args.name, "Missing --name"),
    desc: required(args.description, "Missing --description"),
    ckpt: ckptResolution.resolved,
  };
  const cookieFile = required(args.cookieFile, "Missing --cookie-file");
  const outPath = resolveOutputPath(args.out, payload.name);
  const cookieHeader = extractCookieHeader(await readFile(cookieFile, "utf8"));
  const result = await fetchJson(cookieHeader, instanceId, payload);

  await mkdir(path.dirname(outPath), { recursive: true });
  const record = {
    status: result.status,
    payload,
    instanceId,
    ckptResolution,
    body: result.body,
    savedAt: new Date().toISOString(),
  };
  await writeFile(outPath, `${JSON.stringify(record, null, 2)}\n`, "utf8");
  console.log(`Published model request saved: ${outPath}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error?.stack || error);
    process.exitCode = 1;
  });
}
