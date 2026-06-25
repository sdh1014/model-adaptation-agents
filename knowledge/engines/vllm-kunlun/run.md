# vLLM-Kunlun 运行知识

## Runbook 关注点

`env.sh` 通常记录：

- Kunlun 设备与运行时环境变量；
- vLLM、vLLM-Kunlun 和目标仓 Python 路径；
- 模型路径、dtype、TP、最大长度；
- 服务 host、port 和 endpoint。

`start.sh` 应保留完整、已验证的 vLLM-Kunlun 启动命令。不要由 Skill 根据模型分析结果拼装未知参数。

## 运行前检查

- Python 实际 import 的 `vllm` 和 Kunlun 插件路径；
- 目标仓 commit 与当前实现工作项一致；
- P800 设备数量满足 TP；
- 服务端口未被占用；
- 启动命令保持前台。

## 就绪与 Smoke

优先使用引擎提供的健康接口。健康接口不可用时，`ready.sh` 可执行一个成本较低且不会改变代码的探测。

Smoke 只证明：

```text
服务启动成功
请求协议可用
至少一个最小生成完成
```

它不证明 logits、权重映射或多卡结果正确。
