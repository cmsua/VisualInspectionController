import numpy as np
import pylibdmtx


def scan_data_matrix(image: np.typing.ArrayLike) -> str:
    return str(pylibdmtx.decode(image, max_count=1).data)
