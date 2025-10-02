"""
Simplex con Método de Dos Fases (maneja <=, >=, =)
Compatible con Python 3.13
Requiere: numpy
Uso:
    python simplex_2phase.py
El programa pide:
 - tipo (max/min)
 - número de variables n
 - número de restricciones m
 - coeficientes de la función objetivo (n)
 - para cada restricción: coeficientes (n), tipo (<=, >=, =) y RHS (b)

Devuelve solución (si existe) y valor óptimo.
"""

import numpy as np

EPS = 1e-9

def pivot(table, row, col):
    """Realiza el pivoteo en la tabla (in-place)."""
    table[row, :] = table[row, :] / table[row, col]
    m, n = table.shape
    for r in range(m):
        if r != row:
            table[r, :] -= table[r, col] * table[row, :]

def find_entering_col(table):
    """Para maximización: busca columna con coeficiente negativo en fila objetivo.
       Devuelve índice o None si óptimo."""
    last = table.shape[0] - 1
    row = table[last, :-1]
    min_val = np.min(row)
    if min_val < -EPS:
        return int(np.argmin(row))
    return None

def find_leaving_row(table, col):
    """Calcula la razón mínima para la columna pivote. Devuelve fila o None si no acotado."""
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
    """Ejecuta simplex para MAXIMIZACIÓN sobre la tabla dada.
       La tabla tiene la última fila como -c (objetivo) y última columna b.
       Devuelve (status, table) con status in {"optimal","unbounded"}."""
    while True:
        col = find_entering_col(table)
        if col is None:
            return "optimal", table
        row = find_leaving_row(table, col)
        if row is None:
            return "unbounded", table
        pivot(table, row, col)

def extract_solution(table, num_decision_vars, basis_cols):
    """Extrae solución de variables originales (num_decision_vars) dado la tabla final.
       basis_cols: lista de indices de columnas que están en base (uno por fila)."""
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
    """
    Construye la tabla inicial con:
     - columnas: [x (n), slacks/surplus (s), artificials (a), RHS]
    Devuelve:
     - table (numpy array)
     - list indices: idx_slack, idx_artificial
     - basis: lista de columnas que son básicas al inicio (uno por restricción)
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    slack_cols = []
    art_cols = []
    basis = [-1] * m

    # inicialmente ponemos columnas de variables x
    cols = n
    # contadores
    slack_count = 0
    art_count = 0

    # primer pase para contar cuántas columnas adicionales necesitamos
    for i in range(m):
        if relations[i] == '<=':
            slack_count += 1
        elif relations[i] == '>=':
            slack_count += 1  # tendremos columna de surplus (usamos igual posición)
            art_count += 1
        elif relations[i] == '=':
            art_count += 1

    total_cols = n + slack_count + art_count + 1  # +1 para RHS
    table = np.zeros((m + 1, total_cols))

    # Llenar la parte A (coeficientes de x)
    table[:m, :n] = A
    col_ptr = n
    slack_ptr = []
    art_ptr = []

    # Agregar columnas de slack/surplus y artificiales, y decidir la base inicial
    for i in range(m):
        rel = relations[i]
        if rel == '<=':
            # agregar variable de holgura positiva
            table[i, col_ptr] = 1.0
            slack_ptr.append(col_ptr)
            basis[i] = col_ptr
            col_ptr += 1
        elif rel == '>=':
            # agregar variable surplus (-1) y variable artificial (+1)
            table[i, col_ptr] = -1.0
            slack_ptr.append(col_ptr)
            col_ptr += 1
            table[i, col_ptr] = 1.0
            art_ptr.append(col_ptr)
            basis[i] = col_ptr
            col_ptr += 1
        elif rel == '=':
            # agregar variable artificial (+1) (no slack)
            table[i, col_ptr] = 1.0
            art_ptr.append(col_ptr)
            basis[i] = col_ptr
            col_ptr += 1
        else:
            raise ValueError("Relación inválida: usar '<=', '>=', o '='")

    # RHS
    table[:m, -1] = b

    # Fila objetivo (última fila): para maximización guardamos -c en las primeras n columnas
    if maximize:
        table[-1, :n] = -c
    else:
        table[-1, :n] = c  # para minimizar, vamos a trasformar más tarde (lo manejamos por signo)

    return table, slack_ptr, art_ptr, basis

def make_phase1_objective(table, art_cols):
    """Construye la función objetivo de la fase 1 (maximizar -sum(artificials))
       y ajusta la fila objetivo restando filas básicas que contienen artificials."""
    m, n = table.shape
    # queremos maximizar -sum(art), por lo tanto en la fila objetivo ponemos -1 para cada artificial
    for col in art_cols:
        table[-1, col] = -1.0
    table[-1, -1] = 0.0

    # Si una artificial está en base (hay una fila i donde table[i, col]==1),
    # hay que restar esa fila de la fila objetivo (hacer objetivo consistente)
    for col in art_cols:
        # buscar fila donde columna tiene 1
        rows_with_one = np.where(np.abs(table[:m-1, col] - 1.0) < EPS)[0]
        if rows_with_one.size == 1:
            i = int(rows_with_one[0])
            table[-1, :] -= table[i, :]

def remove_artificial_columns(table, art_cols):
    """Elimina columnas artificiales de la tabla (devuelve tabla reducida)."""
    keep_cols = [j for j in range(table.shape[1]-1) if j not in art_cols] + [-1]
    new_table = table[:, keep_cols]
    return new_table

def run_two_phase(n, m, A, b, relations, c, maximize=True):
    """
    Ejecuta método de dos fases.
    Retorna: status ('optimal','infeasible','unbounded'), solution (or None), objective value (or None)
    """
    # 1) construir tabla inicial con slacks/surplus/artificials
    table, slack_cols, art_cols, basis = build_initial_table(n, m, A, b, relations, c, maximize=True)
    art_cols = list(art_cols)

    # Guardar índice de columnas iniciales de variables de decisión (0..n-1)
    decision_range = list(range(n))

    # 2) FASE 1: Maximizar -sum(artificials)
    if art_cols:
        make_phase1_objective(table, art_cols)
        status, table = simplex_max(table)
        # obtener valor óptimo de fase 1 (recordar que se maximiza -sum(art))
        phase1_value = table[-1, -1]
        # si phase1_value != 0 entonces problema infactible (porque min sum(art) > 0)
        if phase1_value < -EPS:
            # fase1 max -sum(art) < 0  => sum(art) > 0 -> infactible
            return "infeasible", None, None
        # remover columnas de artificiales antes de fase 2
        # identificar si alguna artificial quedó en base; si quedó, hay que intentar eliminarla (columna casi cero) o pivotear
        # Para simplicidad, eliminamos columnas artificiales de la tabla (si están exactamente cero)
        # pero si hay columnas artificiales con valores no-zero en coeficientes (por degeneración), las eliminamos de todas formas.
        table = remove_artificial_columns(table, art_cols)
    else:
        # No hay artificials: la tabla ya es factible — simplemente aseguramos la fila objetivo original
        pass

    # 3) FASE 2: Restaurar objetivo original y resolver
    # Construir fila objetivo original: para maximización: -c en primeras n columnas
    # Notar: la tabla actual puede tener menos columnas (si removimos artificiales). Tenemos que ubicar donde están las variables originales.
    # Reconstruir vector c alineado con las columnas de la tabla
    cols_now = table.shape[1] - 1
    # crear nuevo c_aligned con zeros
    c_aligned = np.zeros(cols_now)
    # las primeras n columnas originalmente correspondían a variables x, y su posición se mantiene en construcción.
    limit = min(n, cols_now)
    # las variables de decisión siguen ocupando los primeros n columnas (si no se eliminaron)
    c_arr = np.array(c, dtype=float)
    if maximize:
        c_aligned[:limit] = -c_arr[:limit]
    else:
        # si original era minimizar, convertimos min -> max multiplicando por -1
        # min c^T x  <=>  max (-c)^T x
        c_aligned[:limit] = c_arr[:limit]  # recordamos que en build_initial_table guardamos -c para max; pero para min lo guardamos como c. Ajustamos:
        c_aligned[:limit] = -c_arr[:limit]

    # establecer la fila objetivo
    table[-1, :-1] = 0.0
    table[-1, :c_aligned.size] = c_aligned[:c_aligned.size]
    table[-1, -1] = 0.0

    # IMPORTANT: Debemos ajustar la fila objetivo usando las filas que actualmente están en base.
    # Para cada fila i, si hay una columna j con 1 y esa columna tiene coeficiente no-cero en objetivo,
    # restamos coef*row_i de la fila objetivo.
    m_table = table.shape[0] - 1
    n_table = table.shape[1] - 1
    for i in range(m_table):
        # buscar columna básica (columna con 1 en la fila i y 0 en otras filas)
        for j in range(n_table):
            col = table[:m_table, j]
            if np.abs(col[i] - 1.0) < EPS and np.count_nonzero(np.abs(col) > EPS) == 1:
                # j es columna básica de fila i
                coef = table[-1, j]
                if abs(coef) > EPS:
                    table[-1, :] -= coef * table[i, :]
                break

    # Ejecutar simplex en fase 2 (maximización)
    status, table = simplex_max(table)
    if status == "unbounded":
        return "unbounded", None, None

    # Extraer solución de las primeras n variables
    solution, obj_value = extract_solution(table, n, None)
    # Si original era minimización, convertir signo del valor objetivo
    if not maximize:
        obj_value = -obj_value

    return "optimal", solution, obj_value

def input_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except:
            print("Valor inválido. Intenta de nuevo.")

def main():
    print("Simplex - Método de Dos Fases (≤, ≥, =)")
    tipo = input("¿Maximizar o Minimizar? (max/min): ").strip().lower()
    maximize = True
    if tipo == 'min':
        maximize = False
    elif tipo != 'max':
        print("Opción no reconocida, asumiendo 'max'.")
        maximize = True

    n = int(input("Número de variables (n): "))
    m = int(input("Número de restricciones (m): "))

    print("\nIntroduce los coeficientes de la función objetivo (vector c):")
    c = [input_float(f"c[{i+1}]: ") for i in range(n)]

    A = []
    b = []
    relations = []
    print("\nAhora ingresa cada restricción:")
    for i in range(m):
        print(f"\nRestricción {i+1}: (forma: a1 a2 ... an  [rel]  b )")
        row = []
        for j in range(n):
            row.append(input_float(f"  Coeficiente a[{i+1},{j+1}]: "))
        rel = input("  Relación (<=, >=, =): ").strip()
        while rel not in ['<=', '>=', '=']:
            print("  Relación inválida. Usa <= , >= o =")
            rel = input("  Relación (<=, >=, =): ").strip()
        rhs = input_float("  Lado derecho b: ")
        A.append(row)
        relations.append(rel)
        b.append(rhs)

    status, solution, value = run_two_phase(n, m, A, b, relations, c, maximize=maximize)

    print("\n====== RESULTADO ======")
    if status == "optimal":
        print("Solución óptima (variables x1..xn):")
        for i, val in enumerate(solution, 1):
            print(f" x{i} = {val:.6f}")
        print(f"Valor óptimo Z = {value:.6f}")
    elif status == "infeasible":
        print("El problema es INFACIL (no existe solución factible).")
    elif status == "unbounded":
        print("El problema es NO ACOtado (unbounded).")
    else:
        print("Estado desconocido:", status)

if __name__ == "__main__":
    main()
