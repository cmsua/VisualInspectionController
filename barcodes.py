import numpy as np
from pylibdmtx.pylibdmtx import decode

def scan_data_matrix(image: np.typing.ArrayLike) -> str:
    return str(decode(image, max_count=1).data)
