"""
Simulación gravitatoria de un sistema planetario (1 estrella + 2 planetas)
Proyecto: Computación - Facultad de Ciencias, UdelaR
Autora: Valentina Oundjian (2026)

Implementación estructurada de acuerdo a la Sección 6.1 de la propuesta:
- condiciones_iniciales: Configuración inicial y transformación al marco del baricentro.
- aceleraciones: Cálculo vectorial de interacciones gravitatorias newtonianas (Ecs. 3 y 4).
- paso_euler: Actualización explícita de Euler (Ecs. 23 y 24).
- paso_leapfrog: Integrador simpléctico de medio impulso (Ecs. 25 a 27).
- magnitudes: Cálculo de energía, momento lineal, momento angular y centro de masa (Ecs. 11 a 17).
- simular: Bucle temporal que almacena el estado completo del sistema.
- graficar: Funciones de visualización con Matplotlib manteniendo escala 1:1.
"""

from typing import Literal
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# 1. CONDICIONES INICIALES (Sección 3.2, Ecuaciones 18 a 20)
# ==============================================================================

def condiciones_iniciales(
    tipo: Literal["tres_cuerpos", "dos_cuerpos"] = "tres_cuerpos",
    G: float = 1.0,
    M_estrella: float = 1.0,
    m1: float = 1e-3,
    m2: float = 1e-3,
    r1_0: float = 1.0,
    r2_0: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Define masas, posiciones y velocidades iniciales en unidades normalizadas (M0=M, L0=1, G=1).
    Aplica la transformación al marco del centro de masa (baricentro) para que R_CM(0)=0 y P(0)=0.

    Retorna:
        masas: np.ndarray shape (N,)
        posiciones: np.ndarray shape (N, 2)
        velocidades: np.ndarray shape (N, 2)
        T_ref: Período de referencia del planeta interior (Ec. 22)
        G: Constante de gravitación
    """
    if tipo == "tres_cuerpos":
        # Masas: Estrella (M), Planeta 1 (m1), Planeta 2 (m2)
        masas = np.array([M_estrella, m1, m2], dtype=np.float64)

        # Posiciones antes de centrar
        pos = np.array([
            [0.0, 0.0],   # Estrella
            [r1_0, 0.0],  # Planeta 1
            [r2_0, 0.0],  # Planeta 2
        ], dtype=np.float64)

        # Velocidades relativas iniciales (Ec. 19): órbitas circulares del par aislado
        u1 = np.sqrt(G * (M_estrella + m1) / r1_0)
        u2 = np.sqrt(G * (M_estrella + m2) / r2_0)

        vel = np.array([
            [0.0, 0.0],   # Estrella
            [0.0, u1],    # Planeta 1
            [0.0, u2],    # Planeta 2
        ], dtype=np.float64)

        # Período orbital de referencia del planeta interior (Ec. 22)
        T_ref = 2.0 * np.pi * np.sqrt((r1_0**3) / (G * (M_estrella + m1)))

    elif tipo == "dos_cuerpos":
        # Sistema de prueba: 1 estrella y 1 planeta
        masas = np.array([M_estrella, m1], dtype=np.float64)
        pos = np.array([
            [0.0, 0.0],
            [r1_0, 0.0],
        ], dtype=np.float64)
        u1 = np.sqrt(G * (M_estrella + m1) / r1_0)
        vel = np.array([
            [0.0, 0.0],
            [0.0, u1],
        ], dtype=np.float64)
        T_ref = 2.0 * np.pi * np.sqrt((r1_0**3) / (G * (M_estrella + m1)))

    else:
        raise ValueError(f"Tipo de sistema desconocido: {tipo}")

    # Transformación al centro de masa (Ec. 20)
    M_tot = np.sum(masas)
    R_cm = np.sum(pos * masas[:, np.newaxis], axis=0) / M_tot
    V_cm = np.sum(vel * masas[:, np.newaxis], axis=0) / M_tot

    pos_cm = pos - R_cm
    vel_cm = vel - V_cm

    return masas, pos_cm, vel_cm, float(T_ref), float(G)


# ==============================================================================
# 2. ACELERACIONES GRAVITATORIAS (Secciones 2.2 y 2.3, Ecuaciones 2 a 4)
# ==============================================================================

def aceleraciones(
    posiciones: np.ndarray,
    masas: np.ndarray,
    G: float = 1.0,
) -> np.ndarray:
    """
    Calcula la aceleración de cada cuerpo debida a la atracción gravitatoria de todos los demás.
    
    a_i = G * sum_{j != i} m_j * (r_j - r_i) / |r_j - r_i|^3  (Ec. 3)

    Detecta separaciones nulas antes de dividir para evitar indefiniciones.
    """
    N = len(masas)
    a = np.zeros_like(posiciones, dtype=np.float64)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            
            # Vector separación r_ij = r_j - r_i (apunta de i hacia j)
            d_ij = posiciones[j] - posiciones[i]
            r_ij = np.sqrt(d_ij[0]**2 + d_ij[1]**2)

            if r_ij == 0.0:
                raise ZeroDivisionError(f"Colisión detectada: separación nula entre cuerpos {i} y {j}")

            # Contribución a la aceleración de i producida por j
            a[i] += G * masas[j] * d_ij / (r_ij**3)

    return a


# ==============================================================================
# 3. MÉTODOS DE INTEGRACIÓN NUMÉRICA (Sección 4, Ecuaciones 23 a 27)
# ==============================================================================

def paso_euler(
    posiciones: np.ndarray,
    velocidades: np.ndarray,
    aceleraciones_actuales: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Avanza un paso de tiempo con el método de Euler explícito (Ecs. 23 y 24):
    r^{n+1}_i = r^n_i + v^n_i * dt
    v^{n+1}_i = v^n_i + a^n_i * dt

    Ambas cantidades se actualizan utilizando estrictamente el estado inicial del paso (t_n).
    """
    nuevas_pos = posiciones + velocidades * dt
    nuevas_vel = velocidades + aceleraciones_actuales * dt
    return nuevas_pos, nuevas_vel


def paso_leapfrog(
    posiciones: np.ndarray,
    velocidades: np.ndarray,
    aceleraciones_actuales: np.ndarray,
    dt: float,
    masas: np.ndarray,
    G: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Avanza un paso de tiempo con el método simpléctico Leapfrog (medio impulso,
    desplazamiento y medio impulso, Ecs. 25 a 27):
    v^{n+1/2} = v^n + (dt/2) * a^n
    r^{n+1}   = r^n + dt * v^{n+1/2}
    a^{n+1}   = aceleraciones(r^{n+1})
    v^{n+1}   = v^{n+1/2} + (dt/2) * a^{n+1}
    """
    v_medio = velocidades + 0.5 * dt * aceleraciones_actuales
    nuevas_pos = posiciones + dt * v_medio
    nuevas_acc = aceleraciones(nuevas_pos, masas, G)
    nuevas_vel = v_medio + 0.5 * dt * nuevas_acc
    return nuevas_pos, nuevas_vel, nuevas_acc


# ==============================================================================
# 4. MAGNITUDES FÍSICAS Y LEYES DE CONSERVACIÓN (Sección 2.6 y 2.7, Ecs. 11 a 17)
# ==============================================================================

def magnitudes(
    posiciones: np.ndarray,
    velocidades: np.ndarray,
    masas: np.ndarray,
    G: float = 1.0,
) -> dict[str, float | np.ndarray]:
    """
    Calcula las magnitudes físicas del sistema en un instante dado:
    - Energía cinética K (Ec. 13)
    - Energía potencial gravitatoria U (Ec. 13 y 14)
    - Energía mecánica total E = K + U
    - Momento lineal total P = (P_x, P_y) (Ec. 12)
    - Momento angular total L_z (Ec. 17)
    - Posición del centro de masa R_CM (Ec. 11)
    """
    N = len(masas)
    
    # Energía cinética K = 0.5 * sum(m_i * |v_i|^2)
    rapidez_sq = np.sum(velocidades**2, axis=1)
    K = 0.5 * np.sum(masas * rapidez_sq)

    # Energía potencial U = - sum_{i < j} G * m_i * m_j / r_ij
    U = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d_ij = posiciones[j] - posiciones[i]
            r_ij = np.sqrt(d_ij[0]**2 + d_ij[1]**2)
            if r_ij == 0.0:
                raise ZeroDivisionError(f"Separación nula al calcular potencial entre {i} y {j}")
            U -= G * masas[i] * masas[j] / r_ij

    E = K + U

    # Momento lineal P = sum(m_i * v_i)
    P = np.sum(masas[:, np.newaxis] * velocidades, axis=0)

    # Momento angular L_z = sum(m_i * (x_i * v_{y,i} - y_i * v_{x,i}))
    L_z = np.sum(masas * (posiciones[:, 0] * velocidades[:, 1] - posiciones[:, 1] * velocidades[:, 0]))

    # Centro de masa R_CM
    M_tot = np.sum(masas)
    R_cm = np.sum(posiciones * masas[:, np.newaxis], axis=0) / M_tot

    return {
        "K": float(K),
        "U": float(U),
        "E": float(E),
        "P": P,
        "L_z": float(L_z),
        "R_cm": R_cm,
    }


# ==============================================================================
# 5. BUCLE PRINCIPAL DE SIMULACIÓN (Secciones 3.4 y 6.1)
# ==============================================================================

def simular(
    pos_0: np.ndarray,
    vel_0: np.ndarray,
    masas: np.ndarray,
    dt: float,
    n_pasos: int,
    metodo: Literal["euler", "leapfrog"] = "euler",
    G: float = 1.0,
) -> dict:
    """
    Evoluciona el sistema en el tiempo durante n_pasos a paso constante dt.
    Almacena en cada paso el estado completo y evalúa las magnitudes de control.
    """
    N = len(masas)

    # Arreglos para almacenar la evolución histórica
    tiempos = np.zeros(n_pasos + 1, dtype=np.float64)
    posiciones = np.zeros((n_pasos + 1, N, 2), dtype=np.float64)
    velocidades = np.zeros((n_pasos + 1, N, 2), dtype=np.float64)
    energias = np.zeros(n_pasos + 1, dtype=np.float64)
    delta_E = np.zeros(n_pasos + 1, dtype=np.float64)
    momentos_P = np.zeros((n_pasos + 1, 2), dtype=np.float64)
    momentos_Lz = np.zeros(n_pasos + 1, dtype=np.float64)
    centros_cm = np.zeros((n_pasos + 1, 2), dtype=np.float64)

    # Estado inicial
    pos = pos_0.copy()
    vel = vel_0.copy()
    acc = aceleraciones(pos, masas, G)
    mags_0 = magnitudes(pos, vel, masas, G)

    tiempos[0] = 0.0
    posiciones[0] = pos
    velocidades[0] = vel
    energias[0] = mags_0["E"]
    delta_E[0] = 0.0
    momentos_P[0] = mags_0["P"]
    momentos_Lz[0] = mags_0["L_z"]
    centros_cm[0] = mags_0["R_cm"]

    E_0 = mags_0["E"]
    abs_E0 = abs(E_0) if E_0 != 0.0 else 1.0

    # Bucle de integración
    for step in range(1, n_pasos + 1):
        if metodo == "euler":
            pos, vel = paso_euler(pos, vel, acc, dt)
            acc = aceleraciones(pos, masas, G)
        elif metodo == "leapfrog":
            pos, vel, acc = paso_leapfrog(pos, vel, acc, dt, masas, G)
        else:
            raise ValueError(f"Método desconocido: {metodo}")

        mags = magnitudes(pos, vel, masas, G)

        tiempos[step] = step * dt
        posiciones[step] = pos
        velocidades[step] = vel
        energias[step] = mags["E"]
        delta_E[step] = (mags["E"] - E_0) / abs_E0  # Desviación relativa (Ec. 29)
        momentos_P[step] = mags["P"]
        momentos_Lz[step] = mags["L_z"]
        centros_cm[step] = mags["R_cm"]

    return {
        "tiempos": tiempos,
        "posiciones": posiciones,
        "velocidades": velocidades,
        "energias": energias,
        "delta_E": delta_E,
        "momentos_P": momentos_P,
        "momentos_Lz": momentos_Lz,
        "centros_cm": centros_cm,
        "masas": masas,
        "dt": dt,
        "metodo": metodo,
    }


# ==============================================================================
# 6. GRAFICACIÓN Y VISUALIZACIÓN CIENTÍFICA (Sección 6.1 y 6.2)
# ==============================================================================

def graficar_trayectorias(
    sim_res: dict,
    titulo: str = "Trayectorias orbitales (Escala 1:1)",
    guardar_en: str | None = None,
) -> None:
    """
    Grafica las órbitas en el plano (x, y) asegurando escala idéntica en ambos ejes
    (Sección 3.4: 'Las trayectorias se mostrarán con igual escala en ambos ejes').
    """
    pos = sim_res["posiciones"]
    N = pos.shape[1]

    nombres = ["Estrella", "Planeta 1", "Planeta 2"] if N == 3 else [f"Cuerpo {i}" for i in range(N)]
    colores = ["#d97706", "#2563eb", "#dc2626"] if N == 3 else plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 8))

    for i in range(N):
        color = colores[i]
        nombre = nombres[i]
        # Trayectoria completa
        ax.plot(pos[:, i, 0], pos[:, i, 1], label=f"{nombre}", color=color, lw=1.2, alpha=0.85)
        # Punto inicial
        ax.scatter(pos[0, i, 0], pos[0, i, 1], color=color, marker="o", s=40, zorder=5)
        # Punto final
        ax.scatter(pos[-1, i, 0], pos[-1, i, 1], color=color, marker="x", s=50, zorder=5)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"$x / L_0$", fontsize=12)
    ax.set_ylabel(r"$y / L_0$", fontsize=12)
    ax.set_title(f"{titulo}\n(Método: {sim_res['metodo'].capitalize()}, $\\Delta t = {sim_res['dt']:.5f}$)", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    if guardar_en:
        plt.savefig(guardar_en, dpi=200)
        print(f"[Graficado] Trayectorias guardadas en: {guardar_en}")
    else:
        plt.show()
    plt.close(fig)


def graficar_magnitudes_conservadas(
    sim_res: dict,
    guardar_en: str | None = None,
) -> None:
    """
    Grafica la evolución temporal de las magnitudes de control (Ecs. 29 y 30):
    1. Desviación relativa de la energía delta E(t)
    2. Componentes del momento lineal P_x(t) y P_y(t)
    3. Variación del momento angular L_z(t) - L_z(0)
    4. Desplazamiento del baricentro |R_cm(t)|
    """
    t = sim_res["tiempos"]
    delta_E = sim_res["delta_E"]
    P = sim_res["momentos_P"]
    Lz = sim_res["momentos_Lz"]
    R_cm = sim_res["centros_cm"]
    dt = sim_res["dt"]
    metodo = sim_res["metodo"].capitalize()

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Energía
    axs[0, 0].plot(t, delta_E, color="#b91c1c", lw=1.5)
    axs[0, 0].set_xlabel(r"Tiempo $t / t_0$")
    axs[0, 0].set_ylabel(r"$\delta E(t) = \frac{E(t) - E(0)}{|E(0)|}$")
    axs[0, 0].set_title(f"Desviación relativa de energía ({metodo})")
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)

    # 2. Momento lineal (Px - Px0, Py - Py0)
    axs[0, 1].plot(t, P[:, 0] - P[0, 0], label=r"$P_x(t) - P_x(0)$", color="#1d4ed8", lw=1.2)
    axs[0, 1].plot(t, P[:, 1] - P[0, 1], label=r"$P_y(t) - P_y(0)$", color="#059669", lw=1.2, linestyle="--")
    axs[0, 1].set_xlabel(r"Tiempo $t / t_0$")
    axs[0, 1].set_ylabel(r"$\Delta P$")
    axs[0, 1].set_title("Conservación del momento lineal")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    axs[0, 1].legend()

    # 3. Momento angular
    axs[1, 0].plot(t, Lz - Lz[0], color="#7c3aed", lw=1.5)
    axs[1, 0].set_xlabel(r"Tiempo $t / t_0$")
    axs[1, 0].set_ylabel(r"$L_z(t) - L_z(0)$")
    axs[1, 0].set_title("Variación del momento angular total")
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 4. Deriva del centro de masa
    r_cm_norm = np.sqrt(R_cm[:, 0]**2 + R_cm[:, 1]**2)
    axs[1, 1].plot(t, r_cm_norm, color="#d97706", lw=1.5)
    axs[1, 1].set_xlabel(r"Tiempo $t / t_0$")
    axs[1, 1].set_ylabel(r"$|R_{CM}(t)|$")
    axs[1, 1].set_title("Posición del baricentro (reposo en origen)")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"Evaluación de magnitudes conservadas ({metodo}, $\\Delta t = {dt:.5f}$)", fontsize=14, y=0.99)
    plt.tight_layout()

    if guardar_en:
        plt.savefig(guardar_en, dpi=200)
        print(f"[Graficado] Magnitudes guardadas en: {guardar_en}")
    else:
        plt.show()
    plt.close(fig)
