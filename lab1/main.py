import re
from pathlib import Path
from copy import deepcopy
from pprint import pprint
import numpy as np

"""1 ЧТЕНИЕ И ПОДГОТОВКА ДАННЫХ"""

def parse_file(path: str):
    """Считывает задачу ЛП из текстового файла"""
    lines = [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("Файл должен содержать минимум 3 строки: MIN/MAX, коэффициенты, число ограничений.")

    sense = lines[0].lower()
    if sense not in ("min", "max"):
        raise ValueError("Первая строка должна быть 'MIN' или 'MAX'.")

    objective = [float(x) for x in lines[1].split()]
    n_constraints = int(lines[2])

    constraints = []
    for i in range(3, 3 + n_constraints):
        line = lines[i]
        match = re.search(r'(<=|>=|=)', line)
        if not match:
            raise ValueError(f"В строке ограничения отсутствует знак (<=, >=, =): {line}")
        rel = match.group(1)
        left, right = line.split(rel)
        coeffs = [float(x) for x in left.split()]
        rhs = float(right.strip())
        constraints.append({"coeffs": coeffs, "sense": rel, "rhs": rhs})

    nvars = len(objective)
    for c in constraints:
        if len(c["coeffs"]) != nvars:
            raise ValueError("Количество коэффициентов в ограничении не совпадает с числом переменных.")

    return {"sense": sense, "objective": objective, "constraints": constraints, "nvars": nvars}


def to_canonical_form(lp_data):
    """Приводит задачу ЛП к каноническому виду"""
    data = deepcopy(lp_data)
    sense = data["sense"]
    objective = data["objective"]
    constraints = data["constraints"]
    nvars = data["nvars"]

    # Преобразование задачи MAX -> MIN
    if sense == "max":
        objective = [-c for c in objective]
        sense = "min"

    slack_count = 0
    new_constraints = []

    for con in constraints:
        coeffs = con["coeffs"].copy()
        sign = con["sense"]

        if sign == "<=":
            coeffs.extend([0.0] * slack_count + [1.0])
            slack_count += 1
        elif sign == ">=":
            coeffs.extend([0.0] * slack_count + [-1.0])
            slack_count += 1
        elif sign == "=":
            coeffs.extend([0.0] * slack_count)
        else:
            raise ValueError(f"Неизвестный знак ограничения: {sign}")

        new_constraints.append({"coeffs": coeffs, "rhs": con["rhs"], "sense": "="})

    # Выравнивание длин
    max_len = max(len(c["coeffs"]) for c in new_constraints)
    for c in new_constraints:
        c["coeffs"].extend([0.0] * (max_len - len(c["coeffs"])))
    objective.extend([0.0] * (max_len - len(objective)))

    return {"sense": sense, "objective": objective, "constraints": new_constraints, "nvars": max_len}


""" 2 ВСПОМОГАТЕЛЬНАЯ ЗАДАЧА"""

def expand_variables_for_negativity(lp_data):
    """Заменяет каждую переменную x_i на (x_i+ - x_i-), чтобы учесть отрицательные значения"""
    data = deepcopy(lp_data)
    n = data["nvars"]
    new_objective = []
    new_constraints = []

    # каждая x_i заменяется на (x_i+ - x_i-)
    for c in data["objective"]:
        new_objective.extend([c, -c])

    for con in data["constraints"]:
        new_coeffs = []
        for a in con["coeffs"]:
            new_coeffs.extend([a, -a])
        new_constraints.append({
            "coeffs": new_coeffs,
            "sense": con["sense"],
            "rhs": con["rhs"]
        })

    data["objective"] = new_objective
    data["constraints"] = new_constraints
    data["nvars"] = n * 2
    data["var_mapping"] = [f"x{i+1}+" if j % 2 == 0 else f"x{i+1}-"
                           for i in range(n) for j in range(2)]
    return data

def build_auxiliary_problem(canonical_data):
    """Формирует вспомогательную задачу для первого этапа симплекс-метода"""
    data = deepcopy(canonical_data)
    constraints = data["constraints"]
    nvars = data["nvars"]
    m = len(constraints)

    # Добавляем искусственные переменные
    for i, con in enumerate(constraints):
        artificial = [0.0] * m
        artificial[i] = 1.0
        con["coeffs"].extend(artificial)

    total_vars = nvars + m
    aux_objective = [0.0] * nvars + [1.0] * m

    tableau = [con["coeffs"] + [con["rhs"]] for con in constraints]
    w_row = [-sum(row[j] for row in tableau) for j in range(total_vars + 1)]
    tableau.append(w_row)

    basis = [f"r{i+1}" for i in range(m)]
    return {"tableau": tableau, "basis": basis, "objective": aux_objective, "nvars": total_vars, "m": m}


def print_table(tableau, basis):
    """Выводит текущую симплекс-таблицу в читаемом виде"""
    print("\nТекущая симплекс-таблица:")
    for i, row in enumerate(tableau):
        tag = basis[i] if i < len(basis) else " W "
        print(f"{tag:>3} |", "  ".join(f"{x:>8.3f}" for x in row))
    print()


def find_pivot(tableau):
    """Находит разрешающий элемент (pivot) для очередного шага"""
    last_row = tableau[-1][:-1]
    pivot_col = np.argmin(last_row)
    if last_row[pivot_col] >= 0:
        return None, None  # оптимум найден

    # вычисляем отношения b_i / a_ij
    ratios = []
    for i in range(len(tableau) - 1):
        aij = tableau[i][pivot_col]
        bi = tableau[i][-1]
        if aij > 0:
            ratios.append(bi / aij)
        else:
            ratios.append(float("inf"))

    if all(r == float("inf") for r in ratios):
        # нет допустимых строк - неограниченность
        return -1, pivot_col

    pivot_row = int(np.argmin(ratios))
    return pivot_row, pivot_col


def pivot_step(tableau, basis, pivot_row, pivot_col):
    """Выполняет один симплекс-шаг (обновляет таблицу по разрешающему элементу)"""
    pivot_elem = tableau[pivot_row][pivot_col]
    tableau[pivot_row] = [x / pivot_elem for x in tableau[pivot_row]]

    for i in range(len(tableau)):
        if i != pivot_row:
            factor = tableau[i][pivot_col]
            tableau[i] = [tableau[i][j] - factor * tableau[pivot_row][j] for j in range(len(tableau[i]))]

    basis[pivot_row] = f"x{pivot_col + 1}"


def simplex_solve(aux_data):
    """Решает вспомогательную задачу симплекс-методом"""
    tableau = [row[:] for row in aux_data["tableau"]]
    basis = aux_data["basis"]

    print_table(tableau, basis)
    iteration = 1
    MAX_ITERS = 1000

    while iteration <= MAX_ITERS:
        pivot_row, pivot_col = find_pivot(tableau)
        if pivot_row is None:
            print("Вспомогательная задача решена.")
            return tableau, basis, True
        if pivot_row == -1:
            print("Вспомогательная задача не имеет решения (неограничена).")
            return tableau, basis, False

        print(f"Итерация {iteration}: разрешающий элемент ({pivot_row + 1}, {pivot_col + 1})")
        pivot_step(tableau, basis, pivot_row, pivot_col)
        print_table(tableau, basis)
        iteration += 1

    print("Превышено максимальное число итераций.")
    return tableau, basis, False


""" 3 ОСНОВНАЯ ЗАДАЧА"""
def transition_to_main_task(final_tableau, final_basis, original_obj, num_real_vars):
    """Переход к основной задаче после решения вспомогательной"""
    tableau = [row[:num_real_vars] + [row[-1]] for row in final_tableau[:-1]]
    z_row = original_obj[:] + [0.0]

    for i, base_var in enumerate(final_basis):
        if base_var.startswith("x"):
            idx = int(base_var[1:]) - 1
            coef = original_obj[idx]
            for j in range(len(z_row)):
                z_row[j] -= coef * tableau[i][j]

    tableau.append(z_row)

    print("\nСимплекс-таблица основной задачи (начальная):")
    print_table(tableau, final_basis)
    return tableau, final_basis


def simplex_iterations(tableau, basis):
    """Решение основной задачи"""
    iteration = 1
    MAX_ITERS = 1000

    while iteration <= MAX_ITERS:
        last_row = tableau[-1][:-1]
        pivot_col = np.argmin(last_row)
        if last_row[pivot_col] >= 0:
            print("\nОптимум найден!")
            return tableau, basis, True

        # отношения
        ratios = [tableau[i][-1] / tableau[i][pivot_col] if tableau[i][pivot_col] > 0 else float("inf")
                  for i in range(len(tableau) - 1)]
        if all(r == float("inf") for r in ratios):
            print("\nЗадача не имеет ограничений снизу — неограниченное решение.")
            return tableau, basis, False

        pivot_row = int(np.argmin(ratios))
        pivot_elem = tableau[pivot_row][pivot_col]

        print(f"\nИтерация {iteration}: разрешающий элемент ({pivot_row + 1}, {pivot_col + 1}) = {pivot_elem:.3f}")
        tableau[pivot_row] = [x / pivot_elem for x in tableau[pivot_row]]

        for i in range(len(tableau)):
            if i != pivot_row:
                factor = tableau[i][pivot_col]
                tableau[i] = [tableau[i][j] - factor * tableau[pivot_row][j] for j in range(len(tableau[i]))]

        basis[pivot_row] = f"x{pivot_col + 1}"
        print_table(tableau, basis)
        iteration += 1

    print("Превышено число итераций. Возможно, задача не имеет конечного оптимума.")
    return tableau, basis, False


def extract_solution(tableau, basis, num_vars, success, task_type="min"):
    """Извлечение оптимального решения"""
    x = [0.0] * num_vars
    for i, base_var in enumerate(basis):
        if base_var.startswith("x"):
            idx = int(base_var[1:]) - 1
            if idx < num_vars:
                x[idx] = tableau[i][-1]

    z = tableau[-1][-1]
    if task_type == "min" and z < 0:
        z = -z
    if not success:
      print("\nРешение не найдено — задача не имеет оптимального решения.")
    else:
      print("\n=== Итоговое решение ===")
      print("Оптимальные значения переменных:", x[:num_vars])
      print(f"Значение целевой функции: {z}")
    return x[:num_vars], z


""" 4 ОСНОВНАЯ ПРОГРАММА"""
if __name__ == "__main__":
    data = parse_file("task.txt")
    print("Содержимое файла успешно прочитано:\n")
    pprint(data)

    # учитываем переменные, которые могут быть отрицательными
    data = expand_variables_for_negativity(data)
    print("\nПосле разбиения переменных (x = x+ - x-):\n")
    pprint(data)

    canonical_form = to_canonical_form(data)
    print("\nЗадача в каноническом виде:\n")
    pprint(canonical_form)
    
    aux = build_auxiliary_problem(canonical_form)
    print("\nВспомогательная задача сформирована:\n")
    pprint(aux)

    print("\n-----------------------\n")

    final_table, final_basis, success = simplex_solve(aux)

    print("\n=== Переход к основной задаче ===")
    original_obj = canonical_form['objective']
    num_real_vars = canonical_form['nvars']

    tableau, basis = transition_to_main_task(final_table, final_basis, original_obj, num_real_vars)
    final_tableau, final_basis, success2 = simplex_iterations(tableau, basis)
    extract_solution(final_tableau, final_basis, num_real_vars, success and success2)