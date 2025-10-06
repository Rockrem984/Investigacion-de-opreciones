import numpy as np 
ESP=1e-9
def pivot (table, row,col):
    table[row, :]=table[row, :]/table[row,col]
    m,n=table.shape
    for r in range(m):
        if r !=row:
            table[r, :]-=table[r, col]*table[row, :]
            def