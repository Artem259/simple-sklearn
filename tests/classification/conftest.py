from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture(scope="package")
def dataset_4f() -> tuple[NDArray[Any], NDArray[Any]]:
    current_dir = Path(__file__).parent
    data_path = current_dir.parent.parent / "data" / "classification_4f.csv"

    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    X = data[:, :-1].astype(int)
    y = np.char.add("y", data[:, -1].astype(int).astype(str))
    return X, y
