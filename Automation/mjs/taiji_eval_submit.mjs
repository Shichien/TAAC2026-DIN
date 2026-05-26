#!/usr/bin/env node
import { createReadStream } from "node:fs";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const localRequire = createRequire(import.meta.url);
const skillRequire = createRequire("file:///C:/Users/26421/.codex/skills/taac2026-cli/package.json");
let COS;
try {
  COS = localRequire("cos-nodejs-sdk-v5");
} catch {
  COS = skillRequire("cos-nodejs-sdk-v5");
}

const TAIJI_ORIGIN = "https://taiji.algo.qq.com";
const BUCKET = "hunyuan-external-1258344706";
const REGION = "ap-guangzhou";
const DEFAULT_COS_PREFIX = "2026_AMS_ALGO_Competition/ams_2026_1029731869646210001/infer";

function usage() {
  return `Usage:
  node Automation/mjs/taiji_eval_submit.mjs --mould-id <id> --name <eval-name> --infer-dir <dir> --cookie-file <file> [options]

Options:
  --file <path[=name]>       Add or override one inference file. Repeatable.
  --creator <user>           Defaults to ams_2026_1029731869646210001.
  --image-name <name>        Optional platform image name.
  --cos-prefix <prefix>      COS prefix for uploaded infer files.
  --execute --yes            Upload files and create the Eval task.
  --out <dir>                Output directory. Relative paths are placed under taiji-output/.
  --help                     Show this help.

Dry-run writes a plan and does not upload or create an Eval task.`;
}

function parseArgs(argv) {
  const args = { files: [], execute: false, yes: false };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") args.help = true;
    else if (arg === "--execute") args.execute = true;
    else if (arg === "--yes") args.yes = true;
    else if (arg === "--file") {
      const value = argv[i + 1];
      if (!value || value.startsWith("--")) throw new Error("Missing value for --file");
      args.files.push(value);
      i += 1;
    } else if (arg.startsWith("--")) {
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
  return String(value || "eval").replace(/[^a-zA-Z0-9_.-]/g, "_") || "eval";
}

function resolveOutputDir(outDir, name) {
  const target = outDir || path.join("evaluation-submit", `${safeName(name)}_${Date.now()}`);
  if (!path.isAbsolute(target) && target.split(/[\\/]+/).includes("..")) {
    throw new Error("Relative output paths must not contain '..'.");
  }
  if (path.isAbsolute(target)) return target;
  if (target.split(/[\\/]/)[0] === "taiji-output") return path.resolve(target);
  return path.resolve("taiji-output", target);
}

function extractCookieHeader(fileContent) {
  const text = fileContent.trim();
  const headerLine = text.match(/^cookie:\s*(.+)$/im);
  if (headerLine) return headerLine[1].trim();
  const curlHeader = text.match(/(?:-H|--header)\s+(['"])cookie:\s*([\s\S]*?)\1/i);
  if (curlHeader) return curlHeader[2].trim();
  return text.replace(/^cookie:\s*/i, "").trim();
}

function taijiHeaders(cookieHeader) {
  return {
    accept: "application/json, text/plain, */*",
    "content-type": "application/json",
    cookie: cookieHeader,
    referer: `${TAIJI_ORIGIN}/evaluation`,
    "user-agent": "Mozilla/5.0",
  };
}

async function fetchJson(cookieHeader, endpoint, options = {}) {
  const url = new URL(endpoint, TAIJI_ORIGIN);
  if (options.params) {
    for (const [key, value] of Object.entries(options.params)) {
      if (value !== undefined && value !== null && value !== "") url.searchParams.set(key, String(value));
    }
  }
  const init = { method: options.method || "GET", headers: taijiHeaders(cookieHeader) };
  if (options.body !== undefined) init.body = JSON.stringify(options.body);
  const response = await fetch(url.href, init);
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

async function getFederationToken(cookieHeader) {
  const token = await fetchJson(cookieHeader, "/aide/api/evaluation_tasks/get_federation_token/");
  for (const key of ["id", "key", "Token"]) {
    if (!token?.[key]) throw new Error(`Federation token missing ${key}`);
  }
  return token;
}

function putObject(cos, params) {
  return new Promise((resolve, reject) => {
    cos.putObject(params, (error, data) => {
      if (error) reject(error);
      else resolve(data);
    });
  });
}

function contentTypeForFile(name) {
  if (name.endsWith(".py")) return "text/x-python";
  if (name.endsWith(".json")) return "application/json";
  if (name.endsWith(".txt")) return "text/plain";
  return "";
}

function formatTaijiTime(date) {
  const utc = date.getTime() + date.getTimezoneOffset() * 60_000;
  const bj = new Date(utc + 8 * 60 * 60_000);
  const pad = (n) => String(n).padStart(2, "0");
  return `${bj.getFullYear()}-${pad(bj.getMonth() + 1)}-${pad(bj.getDate())} ${pad(bj.getHours())}:${pad(bj.getMinutes())}:${pad(bj.getSeconds())}`;
}

function parseFileSpec(spec) {
  const eq = spec.lastIndexOf("=");
  if (eq > 0) return { path: spec.slice(0, eq), name: spec.slice(eq + 1) };
  return { path: spec, name: path.basename(spec) };
}

async function listInferFiles(args) {
  const byName = new Map();
  if (args.inferDir) {
    for (const name of ["dataset.py", "model.py", "infer.py"]) {
      const filePath = path.resolve(args.inferDir, name);
      await stat(filePath);
      byName.set(name, { name, path: filePath });
    }
  }
  for (const spec of args.files) {
    const parsed = parseFileSpec(spec);
    const filePath = path.resolve(parsed.path);
    await stat(filePath);
    byName.set(parsed.name, { name: parsed.name, path: filePath });
  }
  if (!byName.size) throw new Error("Provide --infer-dir or at least one --file.");
  return [...byName.values()];
}

async function sha256(filePath) {
  const hash = createHash("sha256");
  const stream = createReadStream(filePath);
  for await (const chunk of stream) hash.update(chunk);
  return hash.digest("hex");
}

function newInferKey(prefix, filename) {
  return `${prefix.replace(/\/+$/, "")}/local--${randomUUID().replaceAll("-", "")}/${filename}`;
}

async function buildUploadedFiles(localFiles, cosPrefix) {
  const rows = [];
  for (const file of localFiles) {
    const s = await stat(file.path);
    rows.push({
      name: file.name,
      path: newInferKey(cosPrefix, file.name),
      size: s.size,
      mtime: formatTaijiTime(s.mtime),
      localPath: file.path,
      sha256: await sha256(file.path),
    });
  }
  return rows;
}

async function uploadToCos(cookieHeader, row) {
  const token = await getFederationToken(cookieHeader);
  const cos = new COS({
    SecretId: token.id,
    SecretKey: token.key,
    SecurityToken: token.Token,
  });
  await putObject(cos, {
    Bucket: BUCKET,
    Region: REGION,
    Key: row.path,
    Body: createReadStream(row.localPath),
    ContentLength: row.size,
    ContentType: contentTypeForFile(row.name),
  });
  return { key: row.path, bytes: row.size };
}

function publicFile(row) {
  return { name: row.name, path: row.path, mtime: row.mtime, size: row.size };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    console.log(usage());
    return;
  }
  const mouldId = Number(required(args.mouldId, "Missing --mould-id"));
  if (!Number.isFinite(mouldId)) throw new Error("--mould-id must be numeric");
  const name = required(args.name, "Missing --name");
  const cookieFile = required(args.cookieFile, "Missing --cookie-file");
  if (args.execute && !args.yes) throw new Error("--execute requires --yes");

  const outDir = resolveOutputDir(args.out, name);
  const localFiles = await listInferFiles(args);
  const uploadedFiles = await buildUploadedFiles(localFiles, args.cosPrefix || DEFAULT_COS_PREFIX);
  const payload = {
    mould_id: mouldId,
    name,
    image_name: args.imageName || "",
    creator: args.creator || "ams_2026_1029731869646210001",
    files: uploadedFiles.map(publicFile),
  };

  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, "payload.json"), `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  await writeFile(path.join(outDir, "uploaded_files_with_hash.json"), `${JSON.stringify(uploadedFiles, null, 2)}\n`, "utf8");

  if (!args.execute) {
    console.log(`Wrote Eval dry-run plan: ${path.join(outDir, "payload.json")}`);
    console.log("No upload/create happened. Add --execute --yes to run live.");
    return;
  }

  const cookieHeader = extractCookieHeader(await readFile(cookieFile, "utf8"));
  const uploadResults = [];
  for (const row of uploadedFiles) uploadResults.push(await uploadToCos(cookieHeader, row));
  const result = await fetchJson(cookieHeader, "/aide/api/evaluation_tasks/", {
    method: "POST",
    body: payload,
  });
  await writeFile(path.join(outDir, "upload_results.json"), `${JSON.stringify(uploadResults, null, 2)}\n`, "utf8");
  await writeFile(path.join(outDir, "result.json"), `${JSON.stringify(result, null, 2)}\n`, "utf8");
  console.log(`Created Eval task: ${result?.id ?? result?.data?.id ?? "unknown"}`);
  console.log(`Wrote Eval result: ${path.join(outDir, "result.json")}`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error?.stack || error);
    process.exitCode = 1;
  });
}
