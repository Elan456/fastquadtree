"""Regression tests for pygame staying an optional dependency.

The main pygame integration suite uses ``pytest.importorskip("pygame")``,
which is correct for behavior tests but means those tests disappear when
pygame is not installed. This file simulates a client machine without pygame
installed and verifies that every non-pygame integration API still imports and
works independently. Only ``fastquadtree.pygame`` itself should require pygame.
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest


def test_core_modules_import_without_pygame_installed(monkeypatch):
    real_import = builtins.__import__
    previous_pygame_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "pygame" or name.startswith("pygame.")
    }

    def block_pygame_import(name, *args, **kwargs):
        if name == "pygame" or name.startswith("pygame."):
            raise ImportError("blocked pygame import")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_pygame_import)
    for module_name in previous_pygame_modules:
        sys.modules.pop(module_name, None)

    import fastquadtree

    for name in ("QuadTree", "QuadTreeObjects", "RectQuadTree", "RectQuadTreeObjects"):
        assert getattr(fastquadtree, name) is not None

    for module_name in (
        "fastquadtree.point_quadtree",
        "fastquadtree.point_quadtree_objects",
        "fastquadtree.rect_quadtree",
        "fastquadtree.rect_quadtree_objects",
    ):
        assert importlib.import_module(module_name) is not None

    previous_pygame_module = sys.modules.pop("fastquadtree.pygame", None)
    had_pygame_attr = hasattr(fastquadtree, "pygame")
    previous_pygame_attr = getattr(fastquadtree, "pygame", None)
    if had_pygame_attr:
        delattr(fastquadtree, "pygame")

    try:
        with pytest.raises(ImportError, match="Pygame is not installed") as exc_info:
            importlib.import_module("fastquadtree.pygame")
        assert isinstance(exc_info.value.__cause__, ImportError)
        assert "blocked pygame import" in str(exc_info.value.__cause__)
    finally:
        sys.modules.pop("fastquadtree.pygame", None)
        for module_name in list(sys.modules):
            if module_name == "pygame" or module_name.startswith("pygame."):
                sys.modules.pop(module_name, None)
        sys.modules.update(previous_pygame_modules)
        if previous_pygame_module is not None:
            sys.modules["fastquadtree.pygame"] = previous_pygame_module
            fastquadtree.pygame = previous_pygame_module
        elif had_pygame_attr:
            fastquadtree.pygame = previous_pygame_attr
