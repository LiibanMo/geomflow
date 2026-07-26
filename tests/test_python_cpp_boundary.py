import pytest
import torch


def test_cpu_cpp_value_constructor_rejects_tensor() -> None:
    geomflow = pytest.importorskip("geomflow")
    if not hasattr(geomflow, "TangentVector2D"):
        pytest.skip("CPU C++ extension is not installed")

    with pytest.raises(TypeError, match="CPU-only C\\+\\+ API"):
        geomflow.TangentVector2D(torch.tensor([1.0, 2.0]))


def test_cpu_cpp_value_constructor_accepts_scalar_list() -> None:
    geomflow = pytest.importorskip("geomflow")
    if not hasattr(geomflow, "TangentVector2D"):
        pytest.skip("CPU C++ extension is not installed")

    assert geomflow.TangentVector2D([1.0, 2.0]).to_list() == [1.0, 2.0]
