import csv

def transpose(m):
    l = [list(x) for x in zip(*m)]
    return l
    
def matrix_edge_finder(m,pad_sentinel):
    '''
    Finds indices of entries (laterally or diagonally) adjacent to non-zero entries of 'm'.
    'm' is a 2D list. 'pad_sentinel' is an integer value. Matrix indices containing this value will be
    included in the return list.

    Returns a list of tuples of row/column index pairs.
    '''
    ind_non0 = matrix_find(m,lambda x: x>0)
    ind_sentinel = matrix_find(m,lambda x: x == pad_sentinel)
    
    row_size = len(m)
    col_size = len(m[0])

    ind_adjacent = []
    for x in [-1,0,1]:
        for y in [-1,0,1]:
                for ind in ind_non0:
                    r = ind[0] + x
                    c = ind[1] + y

                    bound1 = r >= 0 and r < row_size
                    bound2 = c >= 0 and c < col_size
                    unique1 = (r,c) not in ind_adjacent
                    unique2 = (r,c) not in ind_non0

                    if bound1 and bound2 and unique1 and unique2:
                        ind_adjacent.append([r,c])

    #convert to list of tuple index pairs; likley unnecessary
    ind_adjacent.extend(ind_sentinel)
    ind_adjacent = [tuple(ind) for ind in ind_adjacent]
    
    return ind_adjacent

def matrix_find(m, criteria):
    '''
    Find elements of 'm' that satisfy 'criteria'
    'm' is a matrix (i.e. a 2D list)
    Returns a list of tuples, each are a row/column index pair
    '''
    ind = [(i,j) for i,r in enumerate(m) for j,c in enumerate(r) if criteria(c)]
    return ind
    
def update_matrix_from_position(position, matrix, ind_adj):
    '''
    The entries of 'matrix' at the indices contained in 'ind_adj'
    are replaced with the bits of 'position'.

    'position' and 'ind_adj' are corresponding lists
    '''
    pos_bits = [int(x) for x in bin(position)[2:]] #integer converted into list of integre bits

    for bit,ind in zip(pos_bits,ind_adj):
        matrix[ind[0]][ind[1]] = bit
    
    return matrix

def write_matrix_file(m,path):
    with open(path,mode='wb') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(m)

if __name__ == "__main__":
    import random

    rows = 15
    cols = 16

    m = [[0 for i in range(cols)] for j in range(rows)]
    print('initial:\n',m)
    
    
    m[12][12:-1] = [1]*len(m[12][12:-2]) # 12th row; from 12th column to 2nd-to-last
    m = [list(x) for x in zip(*m)]       #transpose
    m[12][:-1] = [1]*len(m[12][:-2])
    m = [list(x) for x in zip(*m)]       #transpose
    #m[feed_ind[0]][feed_ind[1]] = feed_sentinel  # add feed
    #for ind in seed_inds:                                   # add any other seed patches 
    #   m[ind[0]][ind[1]] = 1
    print('final:\n',m)