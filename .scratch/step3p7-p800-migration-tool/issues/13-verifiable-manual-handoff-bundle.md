# 实现可验证的人工交接包

Type: task
Status: ready-for-agent
Blocked by: 12

## What to build

实现 `handoff_bundle.py` 的 build 和 verify 能力，把一个已经自回放通过的 Golden Run 打成可人工复制的交接包。工具只负责生成和校验，CUDA 与 P800 机器之间的复制继续由人完成。

manifest 必须列出允许出现的完整文件集合，并为每个文件记录相对路径、大小和 SHA-256。CUDA 端构建后先校验一次；人工复制后，P800 端使用同一规则再次校验。缺文件、多文件、文件内容变化、Spec 绑定不一致都必须失败。

## Acceptance criteria

- build 只接受已经完成自回放并处于可封存状态的 Golden Run。
- manifest 覆盖交接包中的每个文件，且不允许 manifest 未声明的额外文件。
- CUDA 端构建完成后可以立即 verify；模拟复制后的目录也可以用同一命令 verify。
- 删除文件、增加文件、修改内容、修改大小或替换 Spec 时，verify 都会给出明确失败原因。
- bundle 内包含 P800 恢复所需的下一状态 Spec，但工具不自行修改当前 Working State。
- Agent 只有在 build 结果、CUDA 端 verify 结果和 Spec 绑定都可信后，才把状态原子地推进到 `WAITING / HANDOFF`。
- 工具不实现 SSH、远程复制、对象存储上传或自动登录另一台机器。
- 有自动化测试覆盖正常交接、缺失、额外、篡改和错误 Contract。

## Comments

真实跨机器复制是人工步骤，不属于工具权限。
