# 模型发现与结构接入

## 需要比较的模型事实

- `architectures`；
- `model_type`；
- 任务类型：generation、embedding、reward、multimodal 等；
- config 类和自定义字段；
- 主模型类与输出头；
- remote code 依赖；
- tokenizer、processor 和服务协议。

## 引擎检查项

1. 模型解析入口如何从 config 选择模型类；
2. 是否存在通用 Transformers/fallback 路径；
3. fallback 是否支持目标任务、量化、并行和 P800 backend；
4. 是否存在模型专用 override；
5. 注册发生在上游引擎、硬件插件还是 fork 内；
6. 同名或同系列模型是否真的具有相同计算语义。

## 支持证据

强证据：

- 当前 commit 的注册表、模型类和测试；
- baseline 日志明确选中了预期模型类；
- config 字段被实际消费。

弱证据：

- README 的支持列表；
- 文件名相似；
- 同系列旧模型可运行。

## 常见误判

- architecture 已注册，但模型类缺少新字段；
- fallback 能加载 config，但 P800 backend 不支持实际算子；
- 模型类可实例化，但权重映射仍沿用旧模型；
- fork 中存在代码，但当前 Python 使用的是其他安装版本。
