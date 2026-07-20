# geomflow Python Module — Implementation Plan

## Goal
Distribute the C++ header-only intrinsic Riemannian-manifold CNF library as a
Python module called `geomflow`, using pybind11, PyTorch (for the neural-network
vector field), `uv` as package manager, `pytest` for unit testing, and a data
preprocessing pipeline converting `ArrayLike` → `torch.Tensor`.

## Architecture (Option 3 — Hybrid)
- **Python**: defines the neural network (PyTorch `nn.Module`), preprocesses
  input data (ArrayLike → tensor), passes the module + data to C++.
- **C++**: uses libtorch to evaluate the vector field and runs the intrinsic
  ODE solver (forward + adjoint) against the PyTorch module, returning
  gradients to Python for training.

## Tasks

### 1. Clean up artifacts
- [x] Remove broken `python/pygeomflow.cpp`
- [x] Fix CMakeLists.txt: target name, source file, syntax errors
- [x] Verify `python/bindings.cpp` module name (`geomflow`)

### 2. Build system
- [ ] Create `pyproject.toml` for `uv` (scikit-build + cmake)
- [ ] Add `geomflow.__init__.py` with preprocessing pipeline
- [ ] Add Torch-based vector field wrapper

### 3. Python package
- [ ] `python/geomflow/__init__.py` — preprocessing + CNF high-level API
- [ ] `python/geomflow/_preprocess.py` — ArrayLike → torch.Tensor converter
- [ ] `python/geomflow/_vector_field.py` — PyTorch nn.Module vector field

### 4. Tests
- [ ] Rewrite `tests/test_flow.py` with proper pytest tests
- [ ] Test preprocessing pipeline
- [ ] Test Torch vector field integration

### 5. Verify
- [ ] Build C++ Catch2 tests → all pass
- [ ] Build Python module → `import geomflow` works
- [ ] Run pytest → all pass