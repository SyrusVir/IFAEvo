import sys
from os.path import dirname
sys.path.append(dirname(__file__))
import matrix as mat

rows = 5
cols = 5

m = [[i + j*cols for i in range(cols)] for j in range(rows)]
print(m)
print(mat.matrix_find(m,lambda x: x<5))