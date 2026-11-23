# 🌌 Simulador Orbital de N-Cuerpos con Integradores Simplécticos

Un simulador de mecánica celeste y dinámica orbital de alta precisión para sistemas de $N$ cuerpos en Python.

Diseñado tanto para la **exploración visual interactiva en tiempo real** (con Pygame) como para el **análisis científico y benchmarking de precisión numérica** (conservación de energía y momento angular).

---

## 🚀 Características Principales

1. **Integradores Numéricos Comparables en Tiempo Real**:
   - **Velocity Verlet (2do orden simpléctico)**: Conservación geométrica del espacio de fases sin deriva secular de energía.
   - **Yoshida 4th (4to orden simpléctico)**: Integrador simpléctico de orden superior (Haruo Yoshida, 1990) con error $O(\Delta t^4)$.
   - **Runge-Kutta 4 (RK4 clásico)**: Gran precisión local, pero disipativo/con deriva secular a muy largo plazo.
   - **Euler hacia adelante (1er orden)**: Para fines pedagógicos (demuestra la divergencia y expulsión orbital rápida).

2. **Monitoreo de Invariantes Físicos**:
   - Energía cinética $K$, energía potencial gravitatoria $U$, y energía mecánica total $E = K + U$.
   - Error relativo de energía en tiempo real $\Delta E / |E_0|$.
   - Momento angular total $\vec{L}_z$.
   - Rastreo del baricentro (centro de masa) $\vec{R}_{\text{cm}}$.

3. **Presets Clásicos y Caóticos**:
   - **1. Sistema Solar Interior**: Sol, Mercurio, Venus, Tierra con Luna y Marte en unidades astronómicas reales ($\text{AU}, M_\odot, \text{años}$).
   - **2. Problema de 3 Cuerpos (Figura 8)**: La célebre solución coreográfica periódica descubierta por C. Moore (1993) y probada por Chenciner & Montgomery (2000).
   - **3. Puntos de Lagrange y Troyanos ($L_4$ y $L_5$)**: Estrella masiva, planeta gigante y enjambre de asteroides troyanos librando en equilibrio.
   - **4. Problema Pitagórico de Burrau**: Masas 3, 4 y 5 colocadas en los vértices de un triángulo 3-4-5 en reposo total. Evolución sumamente caótica con asistencias gravitatorias y expulsión.
   - **5. Sistema Binario Circumbinario (tipo Tatooine)**: Dos estrellas en órbita mutua y un planeta orbitando el centro de masa exterior.

4. **Visualizador Interactivo y Sandbox**:
   - Zoom suave centrado en el cursor del ratón.
   - Paneo libre y seguimiento focal de cuerpos celestes o del centro de masa.
   - Creación interactiva de nuevos cuerpos: haz clic y arrastra con el botón izquierdo para fijar posición y vector de velocidad inicial.
   - Rastros orbitales (trails) con desvanecimiento alfa gradual.

---

## 🛠️ Instalación y Requisitos

Requiere Python 3.10+:

```bash
pip install -r requirements.txt
```

Dependencias principales:
- `numpy`
- `pygame-ce`
- `matplotlib`
- `pytest`

---

## 🎮 Uso del Simulador Interactivo

Para abrir la ventana gráfica interactiva:

```bash
python main.py
```

Opciones opcionales por línea de comando:
```bash
# Iniciar directamente en el preset de Figura 8 con integrador Yoshida:
python main.py --preset 2 --integrator yoshida
```

### Controles en Pantalla

| Tecla / Acción | Función |
| :--- | :--- |
| **Clic Derecho + Arrastrar** | Mover la cámara (pan) libremente |
| **Rueda del Ratón** | Zoom in / Zoom out centrado en el puntero |
| **Clic Izquierdo + Arrastrar** | Crear y lanzar un nuevo cuerpo espacial con vector de velocidad |
| **`[1]` a `[5]`** | Cambiar instantáneamente de Preset |
| **`[I]`** | Cambiar integrador (`Verlet` $\to$ `Yoshida` $\to$ `RK4` $\to$ `Euler`) |
| **`[F]`** | Alternar foco de cámara (Cámara Libre $\to$ Baricentro $\to$ Cuerpos individuales) |
| **`[ESPACIO]`** | Pausar / Reanudar la simulación |
| **`[S]` o `[.]`** | Avanzar un único paso de tiempo (cuando está pausado) |
| **`[+]` / `[-]`** | Acelerar o ralentizar la velocidad de simulación temporal |
| **`[T]`** | Mostrar / ocultar rastros de órbitas |
| **`[C]`** | Limpiar los rastros acumulados |
| **`[H]`** | Mostrar / ocultar el panel HUD |
| **`[R]`** | Reiniciar el preset actual a sus condiciones iniciales |

---

## 📊 Benchmark Científico Comparativo

Puedes ejecutar una prueba automatizada que integra miles de pasos con los cuatro integradores y grafica la deriva del error de energía:

```bash
python main.py --benchmark --preset 2 --steps 3000
```

Esto generará el archivo `benchmark_integrators.png` mostrando la curva semi-logarítmica del error relativo $|E(t) - E_0| / |E_0|$.

---

## 🧪 Tests Automatizados

Para verificar el correcto funcionamiento de las ecuaciones de gravitación, la conservación del momento angular y la estabilidad simpléctica:

```bash
pytest tests/
```
