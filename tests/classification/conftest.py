from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture(scope="package")
def dataset_4f() -> tuple[NDArray[np.int64], NDArray[np.str_]]:
    current_dir = Path(__file__).parent
    data_path = current_dir.parent.parent / "data" / "classification_4f.csv"

    data = np.loadtxt(data_path, delimiter=",", skiprows=1)
    X = data[:, :-1].astype(np.int64)
    y = np.char.add("y", data[:, -1].astype(np.int64).astype(str))
    return X, y
