#!/usr/bin/env node
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const TAIJI_ORIGIN = "https://taiji.algo.qq.com";

function usage() {
  return `Usage:
  node Automation/mjs/taiji_eval_logs.mjs --task-id <eval-task-id> --cookie-file <file> [--out <dir>]`;
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

function extractCookieHeader(fileContent) {
  const text = fileContent.trim();
  const headerLine = text.match(/^cookie:\s*(.+)$/im);
  if (headerLine) return headerLine[1].trim();
  const curlHeader = text.match(/(?:-H|--header)\s+(['"])cookie:\s*([\s\S]*?)\1/i);
  if (curlHeader) return curlHeader[2].trim();
  return text.replace(/^cookie:\s*/i, "").trim();
}

function safeName(value) {
  return String(value || "eval").replace(/[^a-zA-Z0-9_.-]/g, "_") || "eval";
}

function resolveOutputDir(outDir, taskId) {
  const stamp = new Date().toISOString().replace(/[-:TZ.]/g, "").slice(0, 14);
  const target = outDir || path.join("evaluation-submit", `event_log_${safeName(taskId)}_${stamp}`);
  if (!path.isAbsolute(target) && target.split(/[\\/]+/).includes("..")) {
    throw new Error("Relative output paths must not contain '..'.");
  }
  if (path.isAbsolute(target)) return target;
  if (target.split(/[\\/]/)[0] === "taiji-output") return path.resolve(target);
  return path.resolve("taiji-output", target);
}

async function fetchJson(cookieHeader, endpoint, params) {
  const url = new URL(endpoint, TAIJI_ORIGIN);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, String(value));
  }
  const response = await fetch(url.href, {
    headers: {
      accept: "application/json, text/plain, */*",
      cookie: cookieHeader,
      referer: `${TAIJI_ORIGIN}/evaluation`,
      "user-agent": "Mozilla/5.0",
    },
  });
  const text = await response.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = text;
  }
  if (!response.ok) throw new Error(`HTTP ${response.status} ${url.pathname}: ${String(text).slice(0, 500)}`);
  return body;
}

function eventRows(result) {
  return result?.data?.list || result?.list || [];
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }
  const taskId = required(args.taskId, "Missing --task-id");
  const cookieFile = required(args.cookieFile, "Missing --cookie-file");
  const outDir = resolveOutputDir(args.out, taskId);
  const cookieHeader = extractCookieHeader(await readFile(cookieFile, "utf8"));
  const result = await fetchJson(cookieHeader, "/aide/api/evaluation_tasks/event_log/", { task_id: taskId });
  const rows = eventRows(result);

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "event_log.json"), `${JSON.stringify(result, null, 2)}\n`, "utf8");
  await writeFile(
    path.join(outDir, "event_log.txt"),
    `${rows.map((row) => `${row.time || ""}\t${row.message || JSON.stringify(row)}`).join("\n")}\n`,
    "utf8",
  );
  console.log(`Saved Eval event log: ${outDir}`);
  console.log(`Event rows: ${rows.length}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error?.stack || error);
    process.exitCode = 1;
  });
}
