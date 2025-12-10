# Размеры пакетов (шаг дискретизации)
paket_cb1 = 25
paket_cb2 = 200
paket_dep = 100

# Комиссии
komissiya_buy = {
    "cb1": 0.04,
    "cb2": 0.07,
    "dep": 0.05
}

komissiya_sell = {
    "cb1": 0.04,
    "cb2": 0.07,
    "dep": 0.05
}

# Минимальные ограничения на активы
min_cb1 = 30
min_cb2 = 150
min_dep = 100

# Начальное состояние s0
start_state = {
    "cb1": 100.0,
    "cb2": 800.0,
    "dep": 400.0,
    "cash": 600.0
}

# Сценарии для каждого этапа (вероятности и коэффициенты)
scenarios = {
    1: [
        {"name": "good", "p": 0.60, "cb1": 1.20, "cb2": 1.10, "dep": 1.07},
        {"name": "neutral", "p": 0.30, "cb1": 1.05, "cb2": 1.02, "dep": 1.03},
        {"name": "bad", "p": 0.10, "cb1": 0.80, "cb2": 0.95, "dep": 1.00}
    ],
    2: [
        {"name": "good", "p": 0.30, "cb1": 1.40, "cb2": 1.15, "dep": 1.01},
        {"name": "neutral", "p": 0.20, "cb1": 1.05, "cb2": 1.00, "dep": 1.00},
        {"name": "bad", "p": 0.50, "cb1": 0.60, "cb2": 0.90, "dep": 1.00}
    ],
    3: [
        {"name": "good", "p": 0.40, "cb1": 1.15, "cb2": 1.12, "dep": 1.05},
        {"name": "neutral", "p": 0.40, "cb1": 1.05, "cb2": 1.01, "dep": 1.01},
        {"name": "bad", "p": 0.20, "cb1": 0.70, "cb2": 0.94, "dep": 1.00}
    ]
}




def calc_trade_cost(delta_cb1, delta_cb2, delta_dep):
    """
    Расчёт итогового изменения cash при покупке/продаже с учётом комиссий
    Возвращает значение, которое нужно ВЫЧЕСТЬ из cash
    (отрицательное значение = поступление средств)
    """
    cost = 0.0

    # ЦБ1
    s1 = delta_cb1 * paket_cb1
    if delta_cb1 > 0:
        cost += s1 * (1 + komissiya_buy["cb1"])
    elif delta_cb1 < 0:
        cost += s1 * (1 - komissiya_sell["cb1"])  # s1 < 0

    # ЦБ2
    s2 = delta_cb2 * paket_cb2
    if delta_cb2 > 0:
        cost += s2 * (1 + komissiya_buy["cb2"])
    elif delta_cb2 < 0:
        cost += s2 * (1 - komissiya_sell["cb2"])

    # Депозиты
    s3 = delta_dep * paket_dep
    if delta_dep > 0:
        cost += s3 * (1 + komissiya_buy["dep"])
    elif delta_dep < 0:
        cost += s3 * (1 - komissiya_sell["dep"])

    return cost


def is_valid_action(cb1, cb2, dep, cash, delta_cb1, delta_cb2, delta_dep):
    """
    Проверка ограничений для операции
    """

    # новые значения активов после операции
    new_cb1 = cb1 + delta_cb1 * paket_cb1
    new_cb2 = cb2 + delta_cb2 * paket_cb2
    new_dep = dep + delta_dep * paket_dep

    # Проверка минимальных уровней
    if new_cb1 < min_cb1 or new_cb2 < min_cb2 or new_dep < min_dep:
        return False

    # Проверка что нельзя продать больше чем есть
    if new_cb1 < 0 or new_cb2 < 0 or new_dep < 0:
        return False

    # Проверка cash с учётом комиссий
    cost = calc_trade_cost(delta_cb1, delta_cb2, delta_dep)
    new_cash = cash - cost

    if new_cash < 0:
        return False

    return True


def apply_action(cb1, cb2, dep, cash, delta_cb1, delta_cb2, delta_dep):
    """
    Применение допустимого управления u_k
    """
    new_cb1 = cb1 + delta_cb1 * paket_cb1
    new_cb2 = cb2 + delta_cb2 * paket_cb2
    new_dep = dep + delta_dep * paket_dep

    cost = calc_trade_cost(delta_cb1, delta_cb2, delta_dep)
    new_cash = cash - cost

    return new_cb1, new_cb2, new_dep, new_cash


def apply_scenario(cb1, cb2, dep, cash, scenario):
    """
    Применение случайной ситуации w_k
    """
    cb1_next = cb1 * scenario["cb1"]
    cb2_next = cb2 * scenario["cb2"]
    dep_next = dep * scenario["dep"]

    return cb1_next, cb2_next, dep_next, cash




from functools import lru_cache


@lru_cache(None)
def F(k, cb1, cb2, dep, cash):
    """
    Функция Беллмана F_k(s_k)
    """
    cb1 = round(cb1, 2)
    cb2 = round(cb2, 2)
    dep = round(dep, 2)
    cash = round(cash, 2)

    # Граничное условие
    if k == 3:
        return cb1 + cb2 + dep + cash

    best_value = -1e18

    # Перебор возможных действий (ограниченный диапазон пакетов)
    for d_cb1 in range(-2, 3):
        for d_cb2 in range(-2, 3):
            for d_dep in range(-2, 3):

                if not is_valid_action(cb1, cb2, dep, cash, d_cb1, d_cb2, d_dep):
                    continue

                # Применяем управление
                ncb1, ncb2, ndep, ncash = apply_action(
                    cb1, cb2, dep, cash, d_cb1, d_cb2, d_dep
                )

                # Считаем матожидание по сценариям
                expected_value = 0.0
                for sc in scenarios[k + 1]:
                    sc_cb1, sc_cb2, sc_dep, sc_cash = apply_scenario(
                        ncb1, ncb2, ndep, ncash, sc
                    )
                    expected_value += sc["p"] * F(
                        k + 1, sc_cb1, sc_cb2, sc_dep, sc_cash
                    )

                best_value = max(best_value, expected_value)

    return best_value



result = F(
    0,
    start_state["cb1"],
    start_state["cb2"],
    start_state["dep"],
    start_state["cash"]
)

print("Максимальный ожидаемый итоговый капитал:", round(result, 2))


print("\nИТОГИ:")
print("-" * 50)
print("Начальное состояние портфеля:")
print(f"  ЦБ1: {start_state['cb1']} д.е.")
print(f"  ЦБ2: {start_state['cb2']} д.е.")
print(f"  Депозиты: {start_state['dep']} д.е.")
print(f"  Свободные средства (cash): {start_state['cash']} д.е.\n")



print("\nИтог:")
print(f"  Ожидаемый максимальный капитал инвестора через 3 этапа: {round(result, 2)} д.е.")
print("  Полученная стратегия является оптимальной в смысле критерия Байеса.")
