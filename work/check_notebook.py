# -*- coding: utf-8 -*-
"""Validate a Colab notebook: JSON shape + every code cell parses as Python.

IPython shell escapes (a line starting with '!') are replaced by 'pass' so the
surrounding code still parses. A '%' at the start of a continuation line is the
string-format operator, not a magic, so it is left alone.
"""
import ast
import json
import re
import sys

SHELL = re.compile(r"^\s*!")
MAGIC = re.compile(r"^\s*%(matplotlib|cd|load_ext|run|time|timeit|pip)\b")


def strip_magics(src):
    out = []
    for ln in src.split("\n"):
        if SHELL.match(ln) or MAGIC.match(ln):
            out.append(" " * (len(ln) - len(ln.lstrip())) + "pass")
        else:
            out.append(ln)
    return "\n".join(out)


def main(path):
    nb = json.load(open(path, encoding="utf-8"))
    assert nb["nbformat"] == 4, "nbformat must be 4"
    cells = nb["cells"]
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print("nbformat %d.%d | markdown %d | code %d | 共 %d cells"
          % (nb["nbformat"], nb["nbformat_minor"], n_md, n_code, len(cells)))
    print("accelerator:", nb["metadata"].get("accelerator"))

    bad = 0
    defined, used_before = set(), []
    for i, c in enumerate(cells):
        for key in ("cell_type", "metadata", "source"):
            assert key in c, "cell %d missing %s" % (i, key)
        if c["cell_type"] == "code":
            assert "outputs" in c and "execution_count" in c, \
                "code cell %d missing outputs/execution_count" % i
            src = "".join(c["source"])
            try:
                tree = ast.parse(strip_magics(src))
            except SyntaxError as e:
                bad += 1
                print("\n[FAIL] cell %d  line %s: %s" % (i, e.lineno, e.msg))
                for n, ln in enumerate(src.split("\n"), 1):
                    if abs(n - (e.lineno or 0)) <= 2:
                        print("   %3d| %s" % (n, ln[:110]))
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    defined.add(node.name)
                elif isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            defined.add(t.id)

    print("\n語法錯誤 cell 數: %d" % bad)
    print("頂層定義的名稱 %d 個" % len(defined))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
