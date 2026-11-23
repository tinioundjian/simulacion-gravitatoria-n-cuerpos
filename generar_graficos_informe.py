"""
Generador de gráficos de alta calidad para el informe y propuesta académica.
Proyecto: Simulación gravitatoria de un sistema planetario (1 estrella + 2 planetas)
Autora: Valentina Oundjian (Facultad de Ciencias, UdelaR)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from simulacion_planetaria import (
    condiciones_iniciales,
    simular,
    graficar_trayectorias,
    graficar_magnitudes_conservadas,
)

# Estilo de figuras
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

OUTPUT_DIR = "graficos"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generar_figura_1_trayectorias_euler():
    print("-> Generando Figura 1: Trayectorias con Euler explícito...")
    masas, pos_0, vel_0, T_ref, G = condiciones_iniciales("tres_cuerpos")
    
    # Simulación de 10 T_ref con dt = T_ref / 1000 (Sección 3.4)
    dt = T_ref / 1000.0
    n_pasos = int(10 * T_ref / dt)  # 10,000 pasos

    res = simular(pos_0, vel_0, masas, dt=dt, n_pasos=n_pasos, metodo="euler", G=G)

    out_path = os.path.join(OUTPUT_DIR, "figura1_trayectorias_euler.png")
    graficar_trayectorias(
        res,
        titulo=f"Trayectorias del sistema planetario (10 períodos de referencia)",
        guardar_en=out_path,
    )


def generar_figura_2_magnitudes_euler():
    print("-> Generando Figura 2: Magnitudes de control con Euler...")
    masas, pos_0, vel_0, T_ref, G = condiciones_iniciales("tres_cuerpos")
    dt = T_ref / 1000.0
    n_pasos = int(10 * T_ref / dt)

    res = simular(pos_0, vel_0, masas, dt=dt, n_pasos=n_pasos, metodo="euler", G=G)

    out_path = os.path.join(OUTPUT_DIR, "figura2_magnitudes_euler.png")
    graficar_magnitudes_conservadas(res, guardar_en=out_path)


def generar_figura_3_convergencia_paso_temporal():
    print("-> Generando Figura 3: Efecto de reducir el paso temporal (dt, dt/2, dt/4)...")
    masas, pos_0, vel_0, T_ref, G = condiciones_iniciales("tres_cuerpos")
    
    dt_base = T_ref / 1000.0
    pasos_base = int(5 * T_ref / dt_base)  # 5 períodos de referencia

    casos = [
        (dt_base, pasos_base, r"$\Delta t$", "#dc2626"),
        (dt_base / 2.0, pasos_base * 2, r"$\Delta t / 2$", "#2563eb"),
        (dt_base / 4.0, pasos_base * 4, r"$\Delta t / 4$", "#16a34a"),
    ]

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    for dt, n_pasos, label, color in casos:
        res = simular(pos_0, vel_0, masas, dt=dt, n_pasos=n_pasos, metodo="euler", G=G)
        pos = res["posiciones"]
        t = res["tiempos"]
        delta_E = res["delta_E"]

        # 1. Trayectoria del planeta interior (índice 1)
        axs[0].plot(pos[:, 1, 0], pos[:, 1, 1], label=f"Planeta 1 ({label})", color=color, lw=1.2)

        # 2. Desviación de energía
        axs[1].plot(t, delta_E, label=f"Error {label}", color=color, lw=1.4)

    # Gráfica de trayectorias
    axs[0].scatter([0], [0], color="#d97706", s=60, label="Estrella (aprox.)", zorder=5)
    axs[0].set_aspect("equal", adjustable="datalim")
    axs[0].set_xlabel(r"$x / L_0$")
    axs[0].set_ylabel(r"$y / L_0$")
    axs[0].set_title("Efecto del paso temporal en la trayectoria del Planeta 1\n(Menor apertura orbital al reducir $\\Delta t$)")
    axs[0].grid(True, linestyle="--", alpha=0.5)
    axs[0].legend(loc="upper right")

    # Gráfica de energía
    axs[1].set_xlabel(r"Tiempo $t / t_0$")
    axs[1].set_ylabel(r"$\delta E(t) = \frac{E(t) - E(0)}{|E(0)|}$")
    axs[1].set_title(r"Convergencia de la desviación de energía $\delta E(t)$")
    axs[1].grid(True, linestyle="--", alpha=0.5)
    axs[1].legend()

    plt.suptitle("Análisis de convergencia al reducir el paso temporal (Sección 5.4)", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "figura3_convergencia_dt.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Graficado] Figura 3 guardada en: {out_path}")


def generar_figura_4_comparacion_euler_leapfrog():
    print("-> Generando Figura 4: Comparación Euler explícito vs Leapfrog (Ampliación)...")
    masas, pos_0, vel_0, T_ref, G = condiciones_iniciales("tres_cuerpos")
    dt = T_ref / 1000.0
    n_pasos = int(10 * T_ref / dt)

    res_euler = simular(pos_0, vel_0, masas, dt=dt, n_pasos=n_pasos, metodo="euler", G=G)
    res_leap = simular(pos_0, vel_0, masas, dt=dt, n_pasos=n_pasos, metodo="leapfrog", G=G)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    ax_eul_tray = fig.add_subplot(gs[0, 0])
    ax_leap_tray = fig.add_subplot(gs[0, 1])
    ax_energia = fig.add_subplot(gs[1, :])

    # Trayectorias Euler
    pos_e = res_euler["posiciones"]
    ax_eul_tray.plot(pos_e[:, 0, 0], pos_e[:, 0, 1], label="Estrella", color="#d97706", lw=1.2)
    ax_eul_tray.plot(pos_e[:, 1, 0], pos_e[:, 1, 1], label="Planeta 1", color="#2563eb", lw=1.2)
    ax_eul_tray.plot(pos_e[:, 2, 0], pos_e[:, 2, 1], label="Planeta 2", color="#dc2626", lw=1.2)
    ax_eul_tray.set_aspect("equal", adjustable="datalim")
    ax_eul_tray.set_xlabel(r"$x / L_0$")
    ax_eul_tray.set_ylabel(r"$y / L_0$")
    ax_eul_tray.set_title("Euler Explícito: Deriva y apertura orbital espuria")
    ax_eul_tray.grid(True, linestyle="--", alpha=0.5)
    ax_eul_tray.legend(loc="upper right")

    # Trayectorias Leapfrog
    pos_l = res_leap["posiciones"]
    ax_leap_tray.plot(pos_l[:, 0, 0], pos_l[:, 0, 1], label="Estrella", color="#d97706", lw=1.2)
    ax_leap_tray.plot(pos_l[:, 1, 0], pos_l[:, 1, 1], label="Planeta 1", color="#2563eb", lw=1.2)
    ax_leap_tray.plot(pos_l[:, 2, 0], pos_l[:, 2, 1], label="Planeta 2", color="#dc2626", lw=1.2)
    ax_leap_tray.set_aspect("equal", adjustable="datalim")
    ax_leap_tray.set_xlabel(r"$x / L_0$")
    ax_leap_tray.set_ylabel(r"$y / L_0$")
    ax_leap_tray.set_title("Leapfrog (Simpléctico): Órbitas cerradas y estables")
    ax_leap_tray.grid(True, linestyle="--", alpha=0.5)
    ax_leap_tray.legend(loc="upper right")

    # Comparación de energía
    t = res_euler["tiempos"]
    err_e = np.abs(res_euler["delta_E"])
    err_l = np.abs(res_leap["delta_E"])

    ax_energia.semilogy(t, err_e, label=r"Euler Explícito: Error monótono creciente", color="#dc2626", lw=1.8)
    ax_energia.semilogy(t, np.clip(err_l, 1e-12, 1.0), label=r"Leapfrog: Error oscilatorio acotado ($\sim 10^{-7}$)", color="#16a34a", lw=1.8)
    ax_energia.set_xlabel(r"Tiempo $t / t_0$ (Períodos acumulados)", fontsize=12)
    ax_energia.set_ylabel(r"Error relativo $|\delta E(t)| = \frac{|E(t) - E(0)|}{|E(0)|}$", fontsize=12)
    ax_energia.set_title("Comparación de conservación de energía (Escala logarítmica)", fontsize=13)
    ax_energia.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_energia.legend(fontsize=11)

    plt.suptitle("Comparativa: Euler Explícito vs Integrador Simpléctico Leapfrog (Sección 4.3)", fontsize=15)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "figura4_comparacion_euler_leapfrog.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Graficado] Figura 4 guardada en: {out_path}")


def main():
    print("================================================================")
    print(" Generando todas las figuras para el informe de Valentina Oundjian")
    print("================================================================")
    generar_figura_1_trayectorias_euler()
    generar_figura_2_magnitudes_euler()
    generar_figura_3_convergencia_paso_temporal()
    generar_figura_4_comparacion_euler_leapfrog()
    print("\n[Éxito] Todas las figuras fueron generadas en la carpeta 'graficos/'.")


if __name__ == "__main__":
    main()
