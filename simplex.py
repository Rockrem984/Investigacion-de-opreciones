import numpy as np 
ESP=1e-9
def pivot (table, row,col):
    table[row, :]=table[row, :]/table[row,col]
    m,n=table.shape
    for r in range(m):
        if r !=row:
            table[r, :]-=table[r, col]*table[row, :]

def find_entering_col(table):
    last = table.shape[0]-1
    row = table[last, :-1]
    min_val = np.min(row)
    if min_val <-EPS:
        return int(np.argmin(row))
    return None

def find_leaving_row(table, col):
    m = table.shape[0]-1
    ratios =[]
    for i in range(m):
        a = table[i, col] 
        if a > EPS:
            ratios.append(table[i,-1]/a)
        else:
            ratios.append(np.inf)
    ratios = np.array(ratios)
    if np.all(np.isinf(ratios)):
        return None
    return int(np.argmin(ratios))

def simplex_max(table):
    while True:
        col = find_entering_col(table)
        if col is None:
            return , table
        row = find_leaving_row(table, col)
        if row is None:
            return ,table
        pivot(table, row, col)        

def extract_solution(table, num_decision_vars, basis_col ):
    m = table.shape[0]-1
    solution = np.zeros(num_decision_vars)
    for j in range (num_decision_vars):
        col = table[:m,j]
        if np.count_nonzero(np.abs(col - 1.0)<EPS) == 1 and np.count_nonzero(np.abs(col)> EPS) == 1:
            i = int (np.where(np.abs(col -1.0)<EPS)[0][0])
            solution[j] = table[i,-1]
        else:
            solution[j] = 0.0
    objective_value = table[-1,-1]
    return solution, objective_value

def build_initial_table(n, m, A, b, relation, c, maximize=True):
    A=np.array(A, dtype=float)
    b=np.array(b, dtype=float)
    c=np.array(c, dtype=float)

    slack_cols = []
    art_cols = []
    basis = [-1] * m

    col = n

    slack_count = 0
    art_count = 0

    for i in range(m):
        if relation[i] == '<=' :
            
