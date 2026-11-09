# -*- coding: utf-8 -*-
"""Loader for the vendored, UNMODIFIED MyUncle core (third_party/myuncle/MyUncle_core.py, MIT, github.com/sandlerleon/MyUncle).

MyUncleVision never edits that file; `verify()` compares its SHA-256 with the recorded value and the tests fail if they differ."""
import hashlib
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(ROOT, "third_party", "myuncle")
CORE_PATH = os.path.join(CORE_DIR, "MyUncle_core.py")
_mod = None


def sha256(path=CORE_PATH):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def verify():
    """True when the vendored core is byte-identical to the recorded release."""
    rec = open(os.path.join(CORE_DIR, "MyUncle_core.sha256")).read().strip()
    return sha256() == rec


def load():
    """the MyUncle_core module (Config, Bridge, MeanField, MC, ...)"""
    global _mod
    if _mod is None:
        spec = importlib.util.spec_from_file_location("MyUncle_core", CORE_PATH)
        _mod = importlib.util.module_from_spec(spec)
        sys.modules["MyUncle_core"] = _mod
        spec.loader.exec_module(_mod)
    return _mod
