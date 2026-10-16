"""Test strictly verifying m4dash layering rules."""
import os
import ast
import pytest


def test_m4dash_has_zero_engine_and_bench_imports():
    """Verify that m4dash never imports m4, m4bench, or benchmarks."""
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src", "m4dash")
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            assert not alias.name.startswith("m4."), f"{path} illegally imports {alias.name}"
                            assert not alias.name == "m4", f"{path} illegally imports m4"
                            assert not alias.name.startswith("m4bench"), f"{path} illegally imports {alias.name}"
                            assert not alias.name.startswith("benchmarks"), f"{path} illegally imports {alias.name}"
                    elif isinstance(node, ast.ImportFrom):
                        mod = node.module or ""
                        assert not mod.startswith("m4."), f"{path} illegally imports from {mod}"
                        assert not mod == "m4", f"{path} illegally imports from m4"
                        assert not mod.startswith("m4bench"), f"{path} illegally imports from {mod}"
                        assert not mod.startswith("benchmarks"), f"{path} illegally imports from {mod}"
