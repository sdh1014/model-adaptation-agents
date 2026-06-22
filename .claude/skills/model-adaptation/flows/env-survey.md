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

PASS 时至少包含：

```json
{
  "status": "PASS",
  "ready_for_plan": true,
  "target_repo": {},
  "python": {},
  "framework": {},
  "hardware": {},
  "image": {},
  "network": {},
  "storage": {},
  "missing": [],
  "blockers": []
}
```

FAIL 时只写事实，最小形状（禁写字段见 `CLAUDE.md`）：

```json
{
  "status": "FAIL",
  "ready_for_plan": false,
  "failed_check": "xpu_visible",
  "error_signature": "p800-device-not-visible",
  "evidence": ["logs/device-check.log"],
  "blockers": ["P800 device not visible"]
}
```

## Update Targets

更新这些文件：

- `tasks/<task>/status.md`：当前进度、本轮 `result.json` 路径、PASS/FAIL、是否可进入 plan。
- `tasks/<task>/notes.md`：按 `SKILL.md` 的 `Task Notes Rules` 更新。
- `runs/<task>/<ts>-env/result.json`：环境事实、blocker、missing 和 evidence。

## Output Style

- 结论要短。
- 每条结论要能回到命令输出、日志或文件路径。
- 不写长篇环境背景介绍。
- 不写未验证原因分析。
