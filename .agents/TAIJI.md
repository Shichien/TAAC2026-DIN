# Taiji Cookbook

本仓库优先使用 `Automation/ps1/` 下的包装脚本操作平台。它们会自动从仓库根目录执行，并在找不到全局 `taac2026` 命令时回退到本机 Codex skill 里的 CLI。

默认 Cookie 文件是：

```text
taiji-output/secrets/taiji-cookie.txt
```

Cookie 是密钥，只能本地使用，不要提交、打印或写进文档正文。

## 常用入口

| 任务 | 推荐脚本 | 说明 |
| -- | -- | -- |
| 提交训练 Job | `Automation/ps1/submit_job.ps1` | 准备 bundle、运行 doctor、上传并创建 Job，可选立即启动 |
| 发布模型 | `Automation/ps1/publish_model.ps1` | 从训练实例发布指定 checkpoint |
| 提交 Eval | `Automation/ps1/submit_eval.ps1` | 上传 infer 文件并创建 Eval task |
| 拉日志和指标 | `Automation/ps1/fetch_logs.ps1` | 同步训练 Job、ckpt、metrics、代码文件，或拉 Eval event log |
| 监控平台任务 | `Automation/mjs/taiji_platform_monitor.mjs` | 监控训练和 Eval；带 `--apply` 时可重启失败训练 |

`.ps1` 是日常入口，`.mjs` 是底层实现。不要直接改平台接口脚本来绕过包装器，除非是在修复自动化本身。

## 提交训练

先 dry-run 或 doctor，确认 bundle 内容无误；只有用户明确要求提交并启动时才加 `-Execute -RunAfterSubmit`。

```powershell
powershell -ExecutionPolicy Bypass -File Automation/ps1/submit_job.ps1 `
  -TemplateJobUrl "https://taiji.algo.qq.com/training/instances/<task>/<template_job_internal_id>" `
  -TemplateJobInternalId "<template_job_internal_id>" `
  -FileDir "<experiment>/train" `
  -Name "<job_name>" `
  -Description "<job_description>" `
  -Bundle "taiji-output/submit-bundle" `
  -Execute `
  -RunAfterSubmit
```

要点：

- `-FileDir` 适合目录里已有 `code.zip`、`config.yaml`、`run.sh` 或平台模板同名文件。
- `-Zip`、`-Config`、`-RunSh` 适合显式指定单个主文件。
- `-RunAfterSubmit` 必须和 `-Execute` 一起使用。
- 默认会运行 `submit doctor`；只有明确知道原因时才使用 `-SkipDoctor`。
- 只有需要新增模板里不存在的 trainFiles 时才用 `-AllowAddFile`。

## 拉训练日志、代码和指标

同步单个 Job：

```powershell
powershell -ExecutionPolicy Bypass -File Automation/ps1/fetch_logs.ps1 `
  -JobInternalId <job_internal_id>
```

增量同步全部 Job：

```powershell
powershell -ExecutionPolicy Bypass -File Automation/ps1/fetch_logs.ps1 -All
```

拉单个 ckpt 页面：

```powershell
powershell -ExecutionPolicy Bypass -File Automation/ps1/fetch_logs.ps1 `
  -CkptUrl "<ckpt_page_url>"
```

输出会写到 `taiji-output/`，常用文件是：

```text
taiji-output/jobs-summary.csv
taiji-output/all-metrics-long.csv
taiji-output/all-checkpoints.csv
taiji-output/jobs.json
taiji-output/logs/
taiji-output/code/
```

## 发布模型

发布前最好先同步一次对应训练 Job 日志，因为脚本会用本地日志校验 checkpoint 名称，避免发布错 epoch。

```powershell
powershell -ExecutionPolicy Bypass -File Automation/ps1/publish_model.ps1 `
  -InstanceId "<instance_id>" `
  -Ckpt "global_step21744.layer=2.head=4.hidden=64.best_model" `
  -Name "<model_name>" `
  -Description "<description>"
```

`-Ckpt` 可以传 checkpoint 叶子目录名；如果本地日志里有唯一匹配，脚本会解析成平台需要的路径。若匹配不到或有歧义，应先重新拉日志，不要猜。

## 提交 Eval

Eval 使用已发布模型的 `MouldId`，并上传本地 infer 目录。`InferDir` 至少应包含 `dataset.py`、`model.py`、`infer.py`。

```powershell
powershell -ExecutionPolicy Bypass -File Automation/ps1/submit_eval.ps1 `
  -MouldId <model_id> `
  -Name "<eval_name>" `
  -InferDir "<experiment>/infer" `
  -Execute
```

不加 `-Execute` 时只写 dry-run plan，不会上传或创建 Eval。

拉 Eval event log：

```powershell
powershell -ExecutionPolicy Bypass -File Automation/ps1/fetch_logs.ps1 `
  -EvalTaskId <eval_task_id>
```

## 监控和重启

只查看状态：

```powershell
node Automation/mjs/taiji_platform_monitor.mjs `
  --watch-training <task_id> `
  --watch-eval <eval_task_id>
```

同步当前账号下未结束训练任务到 watch set：

```powershell
node Automation/mjs/taiji_platform_monitor.mjs --sync-active
```

真正重启失败训练必须显式加：

```powershell
node Automation/mjs/taiji_platform_monitor.mjs --sync-active --apply
```

## 底层 CLI

只有在包装脚本不覆盖需求时，才直接使用 `taac2026`：

```powershell
taac2026 scrape --all --incremental --cookie-file taiji-output/secrets/taiji-cookie.txt --direct
taac2026 diagnose job --job-internal-id <job_internal_id> --json
taac2026 submit verify --bundle taiji-output/submit-bundle --job-internal-id <job_internal_id>
```

如果没有全局 `taac2026`，使用：

```powershell
node C:/Users/26421/.codex/skills/taac2026-cli/bin/taac2026.mjs <args>
```

## 清理策略

保留：

- `taiji-output/secrets`
- `taiji-output/jobs.json`
- `taiji-output/jobs-summary.csv`
- `taiji-output/all-metrics-long.csv`
- `taiji-output/all-checkpoints.csv`
- `taiji-output/submit-bundle`
- `taiji-output/submit-live`
- `taiji-output/publish`
- `taiji-output/evaluation-submit`
- 当前仍在分析的 `taiji-output/logs`、`taiji-output/code`

可以清理：

- `taiji-output/browser-profile`
- 过期的 dry-run bundle 和 submit-live 记录
- 已经整理进 docs 的旧日志、旧代码缓存和临时 EDA 输出
- `local-smoke-*`、临时 patch、临时 diff

清理前先确认对应结果已经写入 `docs/RUSH.md`、`docs/Thinking.md` 或其他复盘文档。

