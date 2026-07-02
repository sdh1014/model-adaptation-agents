.PHONY: help test tree

help:
	@printf '%s\n' \
	  '常用入口：' \
	  '  Claude Code 中使用 /model-analyze、/adapt-assess、/adapt-implement、/model-run、/adapt-validate、/adapt-benchmark' \
	  '  独立文档生成使用 /model-inference-release-docs <input-file-or-notes>' \
	  '' \
	  '维护命令：' \
	  '  make test   运行自动测试' \
	  '  make tree   查看仓库目录' \
	  '' \
	  '手工初始化 Runbook：' \
	  '  python scripts/model_runtime.py init <model>/<target>'

test:
	bash tests/run.sh

tree:
	find . -path './.git' -prune -o -print | sort
