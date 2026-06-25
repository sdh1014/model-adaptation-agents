---
scope: hardware/p800
status: verified
---

# P800 环境知识

## 使用原则

1. 以当前 target commit 的安装文档和 requirements 为版本依据；
2. 同时记录“安装版本”和“实际 import 路径”，防止加载到其他虚拟环境；
3. 设备可见性必须用实际运行时验证，不只依赖设备管理工具；
4. 插件式引擎要确认上游引擎、插件和本地算子库三者兼容；
5. 环境事实保存在 run 目录，本文只保存跨任务确认过的规则。

## 常见环境问题类别

- Python 环境与执行命令不一致；
- 上游引擎与 Kunlun 插件版本不匹配；
- 本地源码已修改，但运行时加载 site-packages 中的另一份包；
- patch、Python 包和本地动态库来自不同版本；
- 同类算子包同时安装导致符号或注册冲突；
- 动态库搜索路径缺失；
- P800 通过兼容接口暴露，单独检查 `torch.xpu` 得到误判；
- TP 需求超过实际可见设备；
- `/dev/shm` 或模型文件系统空间不足。

## 评估输出

环境勘测报告至少保留：

```text
Python executable
package versions and locations
isolated import results
torch device probe
device tools and nodes
model/repo path facts
disk and shared-memory facts
whitelisted runtime environment variables
```

不得记录访问令牌、私钥或完整进程环境。
