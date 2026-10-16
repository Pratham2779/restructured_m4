"""Layering and architectural isolation enforcement tests."""
import os
import ast
import pytest


def get_m4_python_files():
    """Discover all python files inside packages/m4."""
    m4_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    py_files = []
    for root, _, files in os.walk(m4_root):
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))
    return py_files


def test_m4_has_no_cross_imports():
    """
    Acceptance Criterion 2 & Layering Rules:
    m4 must import nothing from m4bench, benchmarks, m4dash, or any notebook.
    """
    forbidden_modules = {"m4bench", "benchmarks", "m4dash", "streamlit"}
    py_files = get_m4_python_files()
    assert len(py_files) > 0, "No python files found in packages/m4"

    violations = []
    for path in py_files:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_mod = alias.name.split(".")[0]
                    if root_mod in forbidden_modules:
                        violations.append((path, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_mod = node.module.split(".")[0]
                    if root_mod in forbidden_modules:
                        violations.append((path, node.lineno, node.module))

    assert not violations, f"Layering violation! Forbidden imports found in packages/m4: {violations}"


def test_m4_no_prohibited_mentions():
    """
    Acceptance Criterion 7:
    Nothing in packages/m4 mentions LoCoMo, MSC, Colab, Streamlit or Cloudflare.
    """
    prohibited_keywords = ["locomo", "msc", "colab", "streamlit", "cloudflare"]
    m4_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    violations = []
    for root, _, files in os.walk(m4_root):
        if "__pycache__" in root:
            continue
        for f in files:
            if f == "test_layering.py" or f.endswith(".pyc"):
                continue
            file_path = os.path.join(root, f)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
                content_lower = content.lower()
                for kw in prohibited_keywords:
                    # Look for whole word or exact token match
                    if kw in content_lower:
                        violations.append((file_path, kw))

    assert not violations, f"Prohibited keywords found in packages/m4: {violations}"
