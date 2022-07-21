import numpy as np

def diff(x):
    return np.insert(np.diff(x), 0, 0, axis=0)
	
	
def subtraction(x1, x2):
    return x1 - x2