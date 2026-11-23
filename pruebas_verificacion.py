"""
Pruebas de verificación física y numérica.
Cumple con las Secciones 5 y 6.2 de la propuesta de Valentina Oundjian:
1. Comprobación analítica a mano del cálculo de aceleración (Sección 6.2).
2. Prueba analítica del problema de dos cuerpos y convergencia de Euler O(dt) (Sección 5.1 y 5.4).
3. Conservación del momento lineal a nivel de máquina (Sección 5.3, Ec. 30).
4. Verificación de conservación de energía y apertura orbital (Sección 5.2).
5. Comparación con Leapfrog (Sección 4.3).
"""

import numpy as np
import pytest

from simulacion_planetaria import (
    condiciones_iniciales,
    aceleraciones,
    paso_euler,
    paso_leapfrog,
    magnitudes,
    simular,
)


def test_aceleracion_calculo_a_mano():
    """
    Sección 6.2 de la propuesta:
    'Por ejemplo, con G = 1, una masa fuente igual a 1 en (0, 0) y otro cuerpo en (1, 0),
    la aceleración del segundo debida al primero debe ser (-1, 0).
    Esta prueba comprobará el signo, la dirección y el módulo sin depender de un resultado
    esperado inventado por el agente.'
    """
    G = 1.0
    masas = np.array([1.0, 1e-4], dtype=np.float64)
    posiciones = np.array([
        [0.0, 0.0],  # Masa fuente
        [1.0, 0.0],  # Segundo cuerpo
    ], dtype=np.float64)

    a = aceleraciones(posiciones, masas, G)

    # Aceleración del segundo cuerpo (índice 1) debida al primero (índice 0):
    # vector r_0 - r_1 = (-1, 0), distancia = 1.0
    # a_1 = G * m_0 * (r_0 - r_1) / r^3 = 1 * 1 * (-1, 0) / 1^3 = (-1, 0)
    assert np.isclose(a[1, 0], -1.0, atol=1e-12), f"Se esperaba ax = -1.0, se obtuvo {a[1, 0]}"
    assert np.isclose(a[1, 1], 0.0, atol=1e-12), f"Se esperaba ay = 0.0, se obtuvo {a[1, 1]}"

    # Aceleración del primer cuerpo debida al segundo:
    # a_0 = G * m_1 * (r_1 - r_0) / r^3 = (+1e-4, 0)
    assert np.isclose(a[0, 0], 1e-4, atol=1e-12)
    assert np.isclose(a[0, 1], 0.0, atol=1e-12)


def test_deteccion_separacion_nula():
    """Verifica que se detecte una división por cero si dos cuerpos coinciden en posición."""
    masas = np.array([1.0, 1.0])
    posiciones = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ZeroDivisionError):
        aceleraciones(posiciones, masas, G=1.0)


def test_dos_cuerpos_convergencia_euler():
    """
    Sección 5.1 y 5.4 de la propuesta:
    Simula 1 período completo T de órbita circular con paso dt, dt/2 y dt/4.
    Comprueba que el error en la posición tras un período disminuya aproximadamente a la mitad
    al reducir el paso a la mitad (convergencia de orden O(dt) de Euler).
    """
    masas, pos_0, vel_0, T_ref, G = condiciones_iniciales(tipo="dos_cuerpos", r1_0=1.0)

    # Paso de referencia
    dt_base = T_ref / 500.0
    pasos_base = 500

    errores_posicion = []
    pasos_dt = [dt_base, dt_base / 2.0, dt_base / 4.0]

    for factor, dt in enumerate(pasos_dt):
        n_pasos = pasos_base * (2**factor)
        res = simular(pos_0, vel_0, masas, dt=dt, n_pasos=n_pasos, metodo="euler", G=G)

        # Posición relativa final del planeta respecto a la estrella
        pos_rel_final = res["posiciones"][-1, 1] - res["posiciones"][-1, 0]
        pos_rel_inicial = pos_0[1] - pos_0[0]

        # Error tras 1 período: distancia euclídea entre la posición final y la inicial esperada
        err = np.linalg.norm(pos_rel_final - pos_rel_inicial)
        errores_posicion.append(err)

    # Razón de convergencia: err(dt) / err(dt/2) debe ser aproximadamente 2 para un método de orden 1
    razon_1 = errores_posicion[0] / errores_posicion[1]
    razon_2 = errores_posicion[1] / errores_posicion[2]

    print(f"\n[Convergencia 2 cuerpos] Errores en r(T): {errores_posicion}")
    print(f"[Convergencia 2 cuerpos] Razón dt / (dt/2): {razon_1:.2f} (Teórico O(dt): ~2.0)")
    print(f"[Convergencia 2 cuerpos] Razón (dt/2) / (dt/4): {razon_2:.2f} (Teórico O(dt): ~2.0)")

    # Se comprueba que la razón esté cerca de 2 (dentro del régimen asintótico de orden 1)
    assert 1.7 < razon_1 < 2.3
    assert 1.7 < razon_2 < 2.3


def test_conservacion_momento_lineal_exacta():
    """
    Sección 5.3 y Ec. 30:
    'Si las fuerzas se calculan consistentemente por parejas, Euler explícito conserva
    el momento lineal en aritmética exacta: P^{n+1} - P^n = dt * sum m_i a_i = 0.'
    """
    masas, pos_0, vel_0, T_ref, G = condiciones_iniciales(tipo="tres_cuerpos")
    dt = T_ref / 1000.0
    res = simular(pos_0, vel_0, masas, dt=dt, n_pasos=2000, metodo="euler", G=G)

    P = res["momentos_P"]
    delta_Px = np.abs(P[:, 0] - P[0, 0])
    delta_Py = np.abs(P[:, 1] - P[0, 1])

    # El momento lineal debe conservarse a nivel de precisión de máquina (~1e-15 o menor)
    max_err_Px = np.max(delta_Px)
    max_err_Py = np.max(delta_Py)

    print(f"\n[Momento Lineal] Max |Px(t) - Px(0)|: {max_err_Px:.2e}")
    print(f"[Momento Lineal] Max |Py(t) - Py(0)|: {max_err_Py:.2e}")

    assert max_err_Px < 1e-14
    assert max_err_Py < 1e-14


def test_euler_apertura_orbital_y_ganancia_energia():
    """
    Sección 4.2 y 5.2:
    'En general, Euler explícito no conserva exactamente la energía ni el momento angular...
    puede generar una ganancia artificial de energía y una trayectoria que se abre.'
    """
    masas, pos_0, vel_0, T_ref, G = condiciones_iniciales(tipo="tres_cuerpos")
    dt = T_ref / 1000.0
    res = simular(pos_0, vel_0, masas, dt=dt, n_pasos=2000, metodo="euler", G=G)

    delta_E = res["delta_E"]
    # En Euler explícito con órbitas gravitatorias, la energía siempre crece artificialmente (delta_E > 0)
    assert delta_E[-1] > 0.0
    # Y la distancia de los planetas a la estrella en promedio crece (se abre)
    r_planeta1_final = np.linalg.norm(res["posiciones"][-1, 1] - res["posiciones"][-1, 0])
    assert r_planeta1_final > 1.0


def test_leapfrog_vs_euler():
    """
    Sección 4.3:
    'Leapfrog con paso fijo para el sistema considerado es simpléctico... suele producir
    errores energéticos oscilatorios y acotados durante intervalos largos.'
    """
    masas, pos_0, vel_0, T_ref, G = condiciones_iniciales(tipo="tres_cuerpos")
    dt = T_ref / 1000.0
    n_pasos = 2000

    res_euler = simular(pos_0, vel_0, masas, dt=dt, n_pasos=n_pasos, metodo="euler", G=G)
    res_leap = simular(pos_0, vel_0, masas, dt=dt, n_pasos=n_pasos, metodo="leapfrog", G=G)

    max_err_euler = np.max(np.abs(res_euler["delta_E"]))
    max_err_leap = np.max(np.abs(res_leap["delta_E"]))

    print(f"\n[Comparación] Max |delta_E| Euler:    {max_err_euler:.3e}")
    print(f"[Comparación] Max |delta_E| Leapfrog: {max_err_leap:.3e}")

    # Leapfrog debe ser varios órdenes de magnitud más preciso que Euler para el mismo dt
    assert max_err_leap < max_err_euler * 1e-2


if __name__ == "__main__":
    test_aceleracion_calculo_a_mano()
    test_deteccion_separacion_nula()
    test_dos_cuerpos_convergencia_euler()
    test_conservacion_momento_lineal_exacta()
    test_euler_apertura_orbital_y_ganancia_energia()
    test_leapfrog_vs_euler()
    print("\n[OK] ¡Todas las pruebas de verificación física y numérica pasaron exitosamente!")
