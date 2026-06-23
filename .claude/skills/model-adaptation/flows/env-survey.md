# Environment Survey Flow

## Goal

只做环境勘测，判断当前环境是否足够支撑后续 `plan` 和 `bringup`。不安装依赖，不修改目标框架代码。

## Read Order

1. `tasks/<task>/task.yaml`
2. `tasks/<task>/status.md`
3. `tasks/<task>/notes.md`
4. `tasks/<task>/references.md`
5. 最近一次 `runs/<task>/<ts>-model/result.json`
6. 用户提供的机器、容器、镜像、权重路径、目标仓路径

## Checklist

- 目标仓路径是否存在
- 目标仓 branch / commit / dirty 状态
- 目标仓框架版本
- `task.yaml` 中的目标基线版本是否和目标仓匹配
- Python 版本
- PyTorch 版本
- vLLM / SGLang / 目标框架是否可 import
- P800 / XPU 设备是否可见
- 驱动、KLX、XPytorch、算子库版本
- 当前镜像名、镜像 digest 或基础镜像来源
- 代理是否可用
- Hugging Face / BOS / 内部文档访问是否可用
- 模型权重路径是否可访问
- 磁盘空间是否足够
- 最小 tensor / device 检查是否可运行

## Behavior Rules

- 只采集事实，不安装依赖。
- 不修改目标仓代码。
- 不启动大模型。
- 不下载完整权重。
- 不做 P800 bringup。
- 不做性能测试。
- 命令失败时记录命令、退出码、日志路径和错误签名，不写原因分析。
- 缺少权限、机器或容器时，记录为 blocker，不编造环境状态。
- 访问外网资料、下载文件或调用需要联网的命令时，必须使用下面的代理环境变量。

## Network Proxy

```bash
HTTP_PROXY=http://agent.baidu.com:8891 \
HTTPS_PROXY=http://agent.baidu.com:8891 \
http_proxy=http://agent.baidu.com:8891 \
https_proxy=http://agent.baidu.com:8891 \
<command>
```

## Result (PASS / FAIL)

`runs/<task>/<ts>-env/result.json` 记录环境事实。`status` 为 `PASS` 或 `FAIL`。

env-survey 使用固定字段，因为环境事实跨任务同质，后续 `plan` 需要稳定读取。

固定字段不得写空对象。无法确认的字段写入 `missing`，不得编造。关键字段必须带 `evidence`，指向命令、日志或文件路径。

`status=PASS` 表示环境勘测完成，并且可以进入 `plan`。它不等于环境已经可以直接 `bringup`。

PASS 时至少包含这些字段和最低内容：

```json
{
  "status": "PASS",
  "ready_for_plan": true,
  "target_repo": {
    "path": "../vllm-kunlun",
    "exists": true,
    "branch": "main",
    "commit": "abcdef0",
    "dirty": false,
    "framework_version": "0.21.0",
    "evidence": "logs/git-status.log"
  },
  "python": {
    "executable": "/usr/bin/python",
    "version": "3.10.12",
    "evidence": "logs/python.log"
  },
  "framework": {
    "name": "vllm",
    "version": "0.21.0",
    "importable": true,
    "evidence": "logs/framework-import.log"
  },
  "hardware": {
    "name": "P800",
    "visible": true,
    "evidence": "logs/device.log"
  },
  "image": {
    "name": "image:tag",
    "digest": "sha256:...",
    "base": "base-image:tag",
    "evidence": "logs/image.log"
  },
  "network": {
    "proxy_ok": true,
    "hf_access_ok": true,
    "evidence": "logs/network.log"
  },
  "storage": {
    "free_gb": 100,
    "model_path_accessible": true,
    "evidence": "logs/storage.log"
  },
  "missing": [],
  "blockers": [],
  "bringup_blockers": []
}
```

FAIL 时只写事实，最小形状（禁写字段见 `CLAUDE.md`）：

```json
{
  "status": "FAIL",
  "ready_for_plan": false,
  "failed_check": "framework_import",
  "error_signature": "vllm-import-failed",
  "evidence": ["logs/framework-import.log"],
  "blockers": ["target framework is not importable"]
}
```

## ready_for_plan Rules

`ready_for_plan=true` 当且仅当 `blockers` 为空。

写入 `blockers` 的情况：

- `tasks/<task>/task.yaml` 无法读取。
- 最近一次 `model-research` 结果无法读取。
- 目标仓路径不存在或不可访问。
- 目标仓 branch / commit / framework version 无法确认。
- 目标框架无法 import。
- Python 环境无法确认或无法执行最小命令。
- 目标基线版本无法和目标仓状态建立关系。

写入 `bringup_blockers` 但不阻止 `plan` 的情况：

- P800 / XPU 设备不可见。
- 镜像、驱动、KLX、XPytorch 或算子库缺失。
- 模型权重路径不可访问。
- 磁盘空间不足以运行后续 bringup。

写入 `missing` 但不阻止 `plan` 的情况：

- Hugging Face 不可访问，但已有本地配置、技术报告或 model-research 结果。
- 镜像 digest 无法确认，但镜像名和运行环境可确认。
- 当前阶段不需要的可选路径或可选资料缺失。

## Update Targets

更新这些文件：

- `tasks/<task>/status.md`：当前进度、本轮 `result.json` 路径、PASS/FAIL、是否可进入 plan。
- `tasks/<task>/notes.md`：按 `SKILL.md` 的 `Task Notes Rules` 更新。
- `runs/<task>/<ts>-env/result.json`：环境事实、blocker、bringup_blocker、missing 和 evidence。

## Output Style

- 结论要短。
- 每条结论要能回到命令输出、日志或文件路径。
- 不写长篇环境背景介绍。
- 不写未验证原因分析。
