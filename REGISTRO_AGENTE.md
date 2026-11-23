# 📝 Registro de Trabajo con el Agente de IA: Simulación Gravitatoria de un Sistema Planetario

**Estudiante:** Valentina Oundjian  
**Asignatura:** Computación — Facultad de Ciencias, Universidad de la República (UdelaR)  
**Fecha:** 6 de noviembre de 2026  
**Entorno:** Python 3.14 + NumPy + Matplotlib en Antigravity  

---

## 1. Introducción y Propósito del Registro

Este documento registra el proceso de desarrollo colaborativo con el agente de IA de Antigravity para la construcción y verificación de una simulación gravitatoria de un sistema idealizado compuesto por **una estrella y dos planetas** en dos dimensiones ($N = 3$), correspondiente a la propuesta de trabajo final del curso de Computación.

A continuación se detallan las etapas de desarrollo, la formulación física, las funciones implementadas, las comprobaciones numéricas efectuadas a mano y en código, y la discusión sobre los resultados y las limitaciones de los métodos numéricos (Euler explícito vs. Leapfrog).

---

## 2. Descripción Física y Adimensionalización

### 2.1. Sistema de Unidades Normalizadas (Sección 3.2 de la propuesta)
Para evitar errores de redondeo asociados a números con exponentes extremos (como $G \approx 6.674 \times 10^{-11} \text{ m}^3\text{kg}^{-1}\text{s}^{-2}$ o $M_\odot \approx 2 \times 10^{30} \text{ kg}$), se adoptó el sistema de unidades normalizadas propuesto:

* **Unidad de masa:** $M_0 = M_{\text{estrella}} = 1$.
* **Unidad de longitud:** $L_0 = 1$ (separación inicial del planeta interior).
* **Unidad de tiempo:** $t_0 = \sqrt{\frac{L_0^3}{G M_0}} = 1$.
* **Constante gravitatoria:** $G = 1$.

En estas unidades, la velocidad se mide en $L_0 / t_0$, y las masas de los planetas son $m_1 = 10^{-3}$ y $m_2 = 10^{-3}$.

### 2.2. Condiciones Iniciales y Transformación al Baricentro
Antes del centrado, los cuerpos se ubican sobre el semieje $x > 0$:
* **Estrella:** $\vec{r}_\star = (0, 0)$, $\vec{v}_\star = (0, 0)$.
* **Planeta 1:** $\vec{r}_1 = (1, 0)$, $\vec{v}_1 = (0, u_1)$ con $u_1 = \sqrt{\frac{G(M + m_1)}{1}} = \sqrt{1.001} \approx 1.00049987$.
* **Planeta 2:** $\vec{r}_2 = (2, 0)$, $\vec{v}_2 = (0, u_2)$ con $u_2 = \sqrt{\frac{G(M + m_2)}{2}} = \sqrt{\frac{1.001}{2}} \approx 0.70746025$.

Para que el sistema sea aislado y el momento lineal total inicial sea nulo ($\vec{P}(0) = \vec{0}$), se calcularon el centro de masa $\vec{R}_{\text{cm}}(0)$ y la velocidad del centro de masa $\vec{V}_{\text{cm}}(0)$:
$$\vec{R}_{\text{cm}} = \frac{\sum m_i \vec{r}_i}{M_{\text{tot}}}, \quad \vec{V}_{\text{cm}} = \frac{\sum m_i \vec{v}_i}{M_{\text{tot}}}$$
donde $M_{\text{tot}} = 1 + 10^{-3} + 10^{-3} = 1.002$. Las condiciones transformadas son:
$$\vec{r}_i\,' = \vec{r}_i - \vec{R}_{\text{cm}}(0), \quad \vec{v}_i\,' = \vec{v}_i - \vec{V}_{\text{cm}}(0)$$

---

## 3. Estructura del Código y Explicación de Funciones

Tal como se definió en la **Sección 6.1** de la propuesta, el programa en [`simulacion_planetaria.py`](simulacion_planetaria.py) está modularizado en funciones breves e independientes:

### 3.1. `condiciones_iniciales(tipo, ...)`
* **Entradas:** Selección de sistema (`"tres_cuerpos"` o `"dos_cuerpos"`), masas y distancias iniciales.
* **Salidas:** Tupla `(masas, posiciones, velocidades, T_ref, G)`.
* **Física:** Configura los vectores en unidades normalizadas y aplica la traslación al sistema baricéntrico. Calcula el período kepleriano de referencia del planeta 1: $T_{\text{ref}} = 2\pi \sqrt{\frac{1^3}{G(M + m_1)}} \approx 6.2798$.

### 3.2. `aceleraciones(posiciones, masas, G)`
* **Entradas:** Arreglo de posiciones de dimensión $(N, 2)$, masas de dimensión $(N,)$, constante $G$.
* **Salidas:** Arreglo de aceleraciones de dimensión $(N, 2)$.
* **Física:** Evalúa la segunda ley de Newton combinada con la ley de gravitación universal (Ecuaciones 3 y 4):
  $$\vec{a}_i = G \sum_{j \neq i} m_j \frac{\vec{r}_j - \vec{r}_i}{|\vec{r}_j - \vec{r}_i|^3}$$
  Excluye las autointeracciones ($j = i$) y verifica que $|\vec{r}_j - \vec{r}_i| > 0$, lanzando una excepción `ZeroDivisionError` si se detecta una colisión o separación nula.

### 3.3. `paso_euler(posiciones, velocidades, aceleraciones, dt)`
* **Entradas:** Estado en $t_n$ y paso temporal $\Delta t$.
* **Salidas:** Nuevas posiciones y velocidades en $t_{n+1}$.
* **Física:** Implementa Euler explícito (Ecs. 23 y 24):
  $$\vec{r}_i^{n+1} = \vec{r}_i^n + \vec{v}_i^n \Delta t, \quad \vec{v}_i^{n+1} = \vec{v}_i^n + \vec{a}_i^n \Delta t$$
  Ambas actualizaciones usan simultáneamente el estado anterior en $t_n$.

### 3.4. `paso_leapfrog(posiciones, velocidades, aceleraciones, dt, masas, G)` (Ampliación)
* **Entradas:** Estado en $t_n$, aceleración en $t_n$, $\Delta t$, masas y $G$.
* **Salidas:** Nuevas posiciones, velocidades y aceleraciones en $t_{n+1}$.
* **Física:** Integrador simpléctico de 2º orden en forma *Kick-Drift-Kick* (Ecs. 25 a 27):
  $$\vec{v}_i^{n+1/2} = \vec{v}_i^n + \frac{\Delta t}{2} \vec{a}_i^n$$
  $$\vec{r}_i^{n+1} = \vec{r}_i^n + \Delta t \vec{v}_i^{n+1/2}$$
  $$\vec{a}_i^{n+1} = \text{aceleraciones}(\vec{r}^{n+1})$$
  $$\vec{v}_i^{n+1} = \vec{v}_i^{n+1/2} + \frac{\Delta t}{2} \vec{a}_i^{n+1}$$

### 3.5. `magnitudes(posiciones, velocidades, masas, G)`
* **Física:** Evalúa los invariantes del sistema en cada instante:
  * Energía cinética: $K = \frac{1}{2} \sum m_i |\vec{v}_i|^2$
  * Energía potencial: $U = - G \sum_{i < j} \frac{m_i m_j}{r_{ij}}$ (incluyendo pares estrella-planeta 1, estrella-planeta 2 y planeta 1-planeta 2)
  * Energía total: $E = K + U$
  * Momento lineal total: $\vec{P} = \sum m_i \vec{v}_i$
  * Momento angular total: $L_z = \sum m_i (x_i v_{y,i} - y_i v_{x,i})$
  * Baricentro: $\vec{R}_{\text{cm}} = \frac{\sum m_i \vec{r}_i}{\sum m_i}$

### 3.6. `simular(...)` y `graficar(...)`
* Controlan el bucle temporal, registran el historial de trayectorias e invariantes, y generan figuras científicas con igual escala en ambos ejes (`ax.set_aspect('equal')`).

---

## 4. Registro de Pruebas de Verificación y Respuestas a Preguntas

Todas las pruebas se implementaron en [`pruebas_verificacion.py`](pruebas_verificacion.py) y se ejecutaron mediante `pytest`:

### Prueba 1: Comprobación analítica a mano (Sección 6.2 de la propuesta)
* **Planteo:** $G = 1$, cuerpo 1 con $m_1 = 1$ en $(0, 0)$ y cuerpo 2 en $(1, 0)$.
* **Cálculo teórico:** El vector posición relativa desde 2 hacia 1 es $\vec{r}_1 - \vec{r}_2 = (-1, 0)$, con distancia $r = 1.0$.
  $$\vec{a}_2 = G m_1 \frac{\vec{r}_1 - \vec{r}_2}{r^3} = 1 \cdot 1 \cdot \frac{(-1, 0)}{1^3} = (-1, 0)$$
* **Resultado del código:** $a_{2,x} = -1.000000000000$, $a_{2,y} = 0.000000000000$. Coincidencia exacta con tolerancia $10^{-12}$.

### Prueba 2: Problema analítico de dos cuerpos y orden de convergencia (Sección 5.1 y 5.4)
* **Planteo:** Estrella $M = 1$ y planeta $m = 10^{-3}$ en órbita circular a $r_0 = 1$. Se integró durante 1 período teórico completo $T_{\text{ref}}$ con pasos $\Delta t$, $\Delta t/2$ y $\Delta t/4$.
* **Resultados numéricos del error en posición tras 1 período:**
  * Para $\Delta t = T_{\text{ref}} / 500$: Error $\Delta r = 0.675900$
  * Para $\Delta t / 2$: Error $\Delta r = 0.358532$
  * Para $\Delta t / 4$: Error $\Delta r = 0.184664$
* **Razón de reducción:**
  $$\frac{\text{Error}(\Delta t)}{\text{Error}(\Delta t/2)} = \frac{0.675900}{0.358532} \approx 1.89$$
  $$\frac{\text{Error}(\Delta t/2)}{\text{Error}(\Delta t/4)} = \frac{0.358532}{0.184664} \approx 1.94$$
* **Conclusión física:** Los cocientes se aproximan a $2.0$, lo que confirma experimentalmente que el error global acumulado de Euler explícito es de orden $O(\Delta t)$ (orden 1).

### Prueba 3: Conservación del momento lineal (Sección 5.3, Ec. 30)
* **Planteo:** Por la tercera ley de Newton ($\vec{F}_{ij} = -\vec{F}_{ji}$), en aritmética exacta se debe cumplir:
  $$\vec{P}^{n+1} - \vec{P}^n = \Delta t \sum_i m_i \vec{a}_i^n = \vec{0}$$
* **Resultado numérico:**
  $$\max_t |P_x(t) - P_x(0)| = 1.84 \times 10^{-18}$$
  $$\max_t |P_y(t) - P_y(0)| = 2.60 \times 10^{-18}$$
* **Conclusión:** La conservación del momento lineal se cumple de forma estricta hasta la precisión de redondeo de máquina de coma flotante de doble precisión (IEEE 754 float64).

### Prueba 4: Apertura orbital espuria y ganancia de energía en Euler (Sección 4.2 y 5.2)
* **Observación:** En la simulación de 10 períodos de referencia con Euler explícito ($\Delta t = T_{\text{ref}}/1000$), la energía relativa $\delta E(t) = \frac{E(t)-E(0)}{|E(0)|}$ crece monótonamente hasta alcanzar $+8.745 \times 10^{-2}$ ($+8.7\%$).
* **Interpretación física vs numérica:** Esta ganancia de energía **no es física**, sino un artefacto numérico de Euler explícito. Al proyectar la trayectoria tangencialmente en cada paso ($r^{n+1} = r^n + v^n \Delta t$), el planeta se desplaza sistemáticamente hacia un radio ligeramente mayor que el real, "escalando" el pozo de potencial gravitatorio y abriendo la órbita en espiral.

### Prueba 5: Comparación con el integrador simpléctico Leapfrog (Sección 4.3)
* **Comparación bajo idénticas condiciones ($\Delta t = T_{\text{ref}}/1000$ durante $10 T_{\text{ref}}$):**
  * **Euler explícito:** $\max |\delta E| = 8.745 \times 10^{-2}$ (crecimiento secular continuo; órbitas desintegrándose).
  * **Leapfrog:** $\max |\delta E| = 3.917 \times 10^{-8}$ (error acotado y oscilatorio; órbitas perfectamente estables).
* **Conclusión:** Leapfrog preserva la estructura simpléctica del espacio de fases (teorema de Liouville), eliminando la deriva secular de energía y permitiendo simulaciones estables a largo plazo con el mismo costo computacional por paso.

---

## 5. Figuras Generadas para el Trabajo

Las figuras se encuentran almacenadas en alta resolución (200 DPI) dentro de la carpeta [`graficos/`](graficos/):

1. [**`figura1_trayectorias_euler.png`**](graficos/figura1_trayectorias_euler.png): Trayectorias de la estrella y los dos planetas en el plano $(x, y)$ a escala 1:1, mostrando la apertura progresiva de las órbitas bajo Euler explícito.
2. [**`figura2_magnitudes_euler.png`**](graficos/figura2_magnitudes_euler.png): Panel de cuatro gráficos con la evolución temporal de la energía relativa $\delta E(t)$, momento lineal $\vec{P}(t)$, momento angular $L_z(t)$ y desplazamiento del baricentro.
3. [**`figura3_convergencia_dt.png`**](graficos/figura3_convergencia_dt.png): Comparativa de la reducción del paso temporal ($\Delta t, \Delta t/2, \Delta t/4$), demostrando la convergencia lineal del método.
4. [**`figura4_comparacion_euler_leapfrog.png`**](graficos/figura4_comparacion_euler_leapfrog.png): Contraste directo entre la inestabilidad de Euler y la conservación de Leapfrog en escala logarítmica.
