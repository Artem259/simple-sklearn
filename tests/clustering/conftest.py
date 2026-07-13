from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture(scope="package")
def dataset_2f() -> NDArray[np.int64]:
    current_dir = Path(__file__).parent
    data_path = current_dir.parent.parent / "data" / "clustering_2f.csv"

    X = np.loadtxt(data_path, delimiter=",", skiprows=1, usecols=(0, 1)).astype(np.int64)
    return X
