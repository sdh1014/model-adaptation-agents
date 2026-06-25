.PHONY: test tree

test:
	bash tests/run.sh

tree:
	find . -path './.git' -prune -o -print | sort
