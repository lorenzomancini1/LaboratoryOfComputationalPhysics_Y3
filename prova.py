import os
import numpy as np
import math
#os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
from numba import jit, njit, vectorize, cuda

@cuda.jit
def increment_kernel(io_array):
    pos = cuda.grid(1)
    # x, y = cuda.grid(2) # For 2D array
    if pos < io_array.size:
        io_array[pos] += 1 # do the computation

# Create the data array - usually initialized some other way
data = np.ones(256)

# Set the number of threads in a block
threadsperblock = 32 

# Calculate the number of thread blocks in the grid
blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock

# Now start the kernel
increment_kernel[blockspergrid, threadsperblock](data)

# Print the result
print(data)