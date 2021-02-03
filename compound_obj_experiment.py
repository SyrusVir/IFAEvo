import copy
import random

def experiment(m,rows, cols):
    ind_r = (random.randint(0,rows-1), random.randint(0,cols-1))
    m[ind_r[0]][ind_r[1]] = 'X'
    
    return m


if __name__ == "__main__":
    
    rows = 3
    cols = 3

    m = [[random.randint(1,5) for c in range(cols)] for i in range(rows)]
    print(m)
    print(experiment(m,rows, cols))