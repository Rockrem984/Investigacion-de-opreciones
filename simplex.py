import numpy as np
EPS = 1e-9
def pivot(table, row, col):
    table[row, :] = table[row, :] / table[row, col]
    m, n = table.shape
    for r in range(m):
        if r != row:
            table[r, :] -= table[r, col] * table[row, :]
#_
def find_entering_col(table):
    last = table.shape[0] - 1
    row = table[last, :-1]
    min_val = np.min(row)
    if min_val < -EPS:
        return int(np.argmin(row))
    return None

def find_leaving_row(table, col):
    m = table.shape[0] - 1
    ratios = []
    for i in range(m):
        a = table[i, col]
        if a > EPS:
            ratios.append(table[i, -1] / a)
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
            return "optimal", table
        row = find_leaving_row(table, col)
        if row is None:
            return "unbounded", table
        pivot(table, row, col)

def extract_solution(table, num_decision_vars, basis_cols):
    m = table.shape[0] - 1
    solution = np.zeros(num_decision_vars)
    for j in range(num_decision_vars):
        col = table[:m, j]
        if np.count_nonzero(np.abs(col - 1.0) < EPS) == 1 and np.count_nonzero(np.abs(col) > EPS) == 1:
            i = int(np.where(np.abs(col - 1.0) < EPS)[0][0])
            solution[j] = table[i, -1]
        else:
            solution[j] = 0.0
    objective_value = table[-1, -1]
    return solution, objective_value

def build_initial_table(n, m, A, b, relations, c, maximize=True):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    slack_cols = []
    art_cols = []
    basis = [-1] * m
    cols = n
    slack_count = 0
    art_count = 0
    for i in range(m):
        if relations[i] == '<=':
            slack_count += 1
        elif relations[i] == '>=':
            slack_count += 1  
            art_count += 1
        elif relations[i] == '=':
            art_count += 1

    total_cols = n + slack_count + art_count + 1  
    table = np.zeros((m + 1, total_cols))

    table[:m, :n] = A
    col_ptr = n
    slack_ptr = []
    art_ptr = []

    for i in range(m):
        rel = relations[i]
        if rel == '<=':
            table[i, col_ptr] = 1.0
            slack_ptr.append(col_ptr)
            basis[i] = col_ptr
            col_ptr += 1
        elif rel == '>=':
            table[i, col_ptr] = -1.0
            slack_ptr.append(col_ptr)
            col_ptr += 1
            table[i, col_ptr] = 1.0
            art_ptr.append(col_ptr)
            basis[i] = col_ptr
            col_ptr += 1
        elif rel == '=':
            table[i, col_ptr] = 1.0
            art_ptr.append(col_ptr)
            basis[i] = col_ptr
            col_ptr += 1
        else:
            raise ValueError("Relación inválida: usar '<=', '>=', o '='")

    table[:m, -1] = b
    if maximize:
        table[-1, :n] = -c
    else:
        table[-1, :n] = c  

    return table, slack_ptr, art_ptr, basis