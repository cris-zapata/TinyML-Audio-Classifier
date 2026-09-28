import gzip, struct
import numpy as np
import matplotlib.pyplot as plt
f = gzip.open(path, "rb")
header = f.read(16)
pixels = f.read()
magic, n, rows, cols = struct.unpack(">IIII", header)
arr = np.frombuffer(pixels, dtype=np.uint8)
arr = arr.reshape(n, rows, cols)