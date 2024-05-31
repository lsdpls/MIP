import pybullet as p
import time
import pybullet_data
import numpy as np
import math

def main():
    # Выбор метода интерполяции
    # fifth_order_poly - для интерполяции полиномом пятого порядка 
    # trapezoidal - интерполяция траектории с трапецевидным профилем скорости
    method_option = "fifth_order_poly"

    # Использовать графический интерфейс?
    USE_GUI = True
    # Инициализация PyBullet
    physics_client = initialize_pybullet(USE_GUI)

    p.setGravity(0, 0, -9.8)
    body_id = p.loadURDF("./pendulum.urdf")

    # Начальная конфигурация
    initial_position = -1  # начальная позиция маятника
    simulation_step = 1 / 240  # шаг симуляции
    total_time = 4  # общее время движения
    max_time = total_time # максимальное время для логирования

    # Логирование данных
    log_data = initialize_log_data(initial_position, simulation_step, max_time)

    # Переход в начальную позицию
    move_to_start_position(body_id, initial_position)

    # Подготовка к симуляции
    p.changeDynamics(body_id, 1, linearDamping=0)  # отключаем демпфирование
    p.setJointMotorControl2(body_id, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=0)  # разрешаем свободное вращение маятника

    # Параметры управления
    control_params = {
        "kp": 5005,
        "kv": 100,
        "ki": 0,
        "qd": 0.8,  # конечная позиция маятника
        "g": 9.8,  # ускорение свободного падения
        "m": 1,  # масса маятника
        "L": 0.5,  # длина маятника
        "kf": 0.1  # коэффициент трения
    }

    # Запуск симуляции
    simulate_pendulum(body_id, method_option, initial_position, simulation_step, total_time, log_data, control_params, USE_GUI)

    p.disconnect()

    # Построение графиков
    plot_results(log_data, method_option)

def initialize_pybullet(gui):
    # Инициализация PyBullet
    if gui:
        client = p.connect(p.GUI)
        # Настройка камеры
        p.resetDebugVisualizerCamera(2, 90, -25, [0, 0, 1.5])
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # Добавление координатных осей
        p.addUserDebugLine([0, 0, 0], [1, 0, 0], [1, 0, 0], 4)
        p.addUserDebugLine([0, 0, 0], [0, 1, 0], [0, 1, 0], 4)
        p.addUserDebugLine([0, 0, 0], [0, 0, 1], [0, 0, 1], 4)
        # Загрузка плоскости
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF('plane.urdf')
    else:
        client = p.connect(p.DIRECT)


def initialize_log_data(start_pos, dt, max_time):
    # Инициализация структур для логирования данных
    log_time = np.arange(0.0, max_time, dt)
    data_size = log_time.size
    log_data = {
        "time": log_time,
        "position": np.zeros(data_size),
        "velocity": np.zeros(data_size),
        "acceleration": np.zeros(data_size),
        "jerk": np.zeros(data_size),
        "control": np.zeros(data_size - 1),
        "reference": np.zeros(data_size),
        "ref_velocity": np.zeros(data_size),
        "ref_acceleration": np.zeros(data_size),
        "ref_jerk": np.zeros(data_size)
    }
    log_data["position"][0] = start_pos
    log_data["reference"][0] = start_pos
    return log_data

def move_to_start_position(body_id, target_pos):
    # Перевод маятника в начальную позицию
    p.setJointMotorControl2(body_id, 1, p.POSITION_CONTROL, targetPosition=target_pos)
    for _ in range(1000):
        p.stepSimulation()

def simulate_pendulum(body_id, option, q0, dt, T, log_data, params, gui):
    # Симуляция движения маятника
    idx = 1
    pos = q0
    vel = 0
    prev_vel = 0
    prev_acc = 0
    prev_accd = 0
    for t in log_data["time"][1:]:
        # Выбор метода интерполяции
        if option == "fifth_order_poly":
            (s, posd, veld, accd) = fifth_order_poly_profile(q0, params["qd"], T, t)
        elif option == "trapezoidal":
            (s, posd, veld, accd) = trapezoidal_profile(q0, params["qd"], T, t)

        # Управление по линейной обратной связи
        ctrl = feedback_linearization(pos, vel, posd, veld, accd, params)
        update_log_data(log_data, idx, posd, veld, accd, prev_accd, ctrl)
        prev_accd = accd

        # Применение управляющего воздействия
        p.setJointMotorControl2(body_id, 1, p.TORQUE_CONTROL, force=ctrl)
        p.stepSimulation()
        pos = p.getJointState(body_id, 1)[0]
        vel = p.getJointState(body_id, 1)[1]

        # Обновление логов
        acc = (vel - prev_vel) / dt
        prev_vel = vel
        jerk = (acc - prev_acc) / dt
        prev_acc = acc

        log_data["position"][idx] = pos
        log_data["velocity"][idx] = vel
        log_data["acceleration"][idx] = acc
        log_data["jerk"][idx] = jerk
        log_data["control"][idx - 1] = ctrl
        idx += 1
        if gui:
            time.sleep(dt)

def fifth_order_poly_profile(Q_start, Q_end, T, t):
    # Интерполяция полиномом пятого порядка
    a3 = 10 / T**3
    a4 = -15 / T**4
    a5 = 6 / T**5
    s = a3 * t**3 + a4 * t**4 + a5 * t**5
    ds = 3 * a3 * t**2 + 4 * a4 * t**3 + 5 * a5 * t**4
    dds = 6 * a3 * t + 12 * a4 * t**2 + 20 * a5 * t**3
    Q = Q_start + s * (Q_end - Q_start)
    dQ = (Q_end - Q_start) * ds
    ddQ = (Q_end - Q_start) * dds

    return (s, Q, dQ, ddQ) if t <= T else (s, Q_end, 0, 0)

def trapezoidal_profile(Q_start, Q_end, T, t):
    # Трапециевидный профиль скорости
    v = 1.5 / T
    a = (v**2) / (v * T - 1)
    t_a = v / a
    s = ds = dds = 0
    if t <= t_a:
        s = 0.5 * a * t**2
        ds = a * t
        dds = a
    elif t > t_a and t <= T - t_a:
        s = v * t - (v**2) / (2 * a)
        ds = v
        dds = 0
    elif t > T - t_a and t <= T:
        s = (2 * a * v * T - 2 * v**2 - (a**2) * (t - T)**2) / (2 * a)
        ds = a * (T - t)
        dds = -a

    Q = Q_start + s * (Q_end - Q_start)
    dQ = (Q_end - Q_start) * ds
    ddQ = (Q_end - Q_start) * dds

    return (s, Q, dQ, ddQ) if t <= T else (s, Q_end, 0, 0)

def feedback_linearization(pos, vel, posd, veld, accd, params):
    # Линейная обратная связь
    u = -params["kp"] * (pos - posd) - params["kv"] * vel
    ctrl = params["m"] * params["L"]**2 * ((params["g"] / params["L"]) * math.sin(pos) + params["kf"] / (params["m"] * params["L"]**2) * vel + u)
    return ctrl

def update_log_data(log_data, idx, posd, veld, accd, prev_accd, ctrl):
    # Обновление данных логов
    log_data["reference"][idx] = posd
    log_data["ref_velocity"][idx] = veld
    log_data["ref_acceleration"][idx] = accd
    log_data["ref_jerk"][idx] = (accd - prev_accd) / (log_data["time"][1] - log_data["time"][0])

def plot_results(log_data, option):
    # Построение графиков
    import matplotlib.pyplot as plt

    plt.subplot(4, 1, 1)
    plt.grid(True)
    plt.plot(log_data["time"], log_data["position"], label="simPos")
    plt.plot(log_data["time"], log_data["reference"], label="simRef")

    if option == "fifth_order_poly":
        plt.title("Fifth-order polynomial time scaling")
    elif option == "trapezoidal":
        plt.title("Trapezoidal motion profile")

    plt.legend()

    plt.subplot(4, 1, 2)
    plt.grid(True)
    plt.plot(log_data["time"], log_data["velocity"], label="simVel")
    plt.plot(log_data["time"], log_data["ref_velocity"], label="simRefd")
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.grid(True)
    plt.plot(log_data["time"], log_data["acceleration"], label="simAcc")
    plt.plot(log_data["time"], log_data["ref_acceleration"], label="simRefdd")
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.grid(True)
    plt.plot(log_data["time"], log_data["jerk"], label="simJerk")
    plt.plot(log_data["time"], log_data["ref_jerk"], label="simRefddd")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
