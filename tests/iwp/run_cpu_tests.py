#!/usr/bin/env python3
"""Run inherited converter/energy tests and compiled normal derivatives without pytest."""
from pathlib import Path
import importlib.util
import subprocess
import tempfile
ROOT = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as directory:
    exe = Path(directory) / "normal_test"
    subprocess.run(["g++", "-std=c++14", "-O2", "-Wall", "-Wextra",
                    str(ROOT/"tests/iwp/test_iwp_normal.cpp"), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
    for filename in ("test_convert_fitted_iwp.py", "test_iwp_energy_decomposition.py"):
        spec = importlib.util.spec_from_file_location(filename[:-3], ROOT/"tests/iwp"/filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            if name.startswith("test_"):
                function = getattr(module, name)
                import inspect
                if inspect.signature(function).parameters:
                    function(Path(directory))
                else:
                    function()
                print("PASS:", name)
