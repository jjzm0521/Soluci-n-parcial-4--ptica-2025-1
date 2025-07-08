# Simulador de Difracción de Fraunhofer 2D

Este proyecto proporciona herramientas para calcular y visualizar el patrón de difracción de Fraunhofer para varias aberturas 2D utilizando la Transformada Rápida de Fourier (FFT). Se incluyen dos implementaciones de interfaz gráfica de usuario (GUI): una basada en Tkinter y otra directamente en Matplotlib con sus widgets.

## Principios Teóricos

La difracción de Fraunhofer (o difracción de campo lejano) ocurre cuando tanto la fuente de luz como la pantalla de observación están efectivamente a una distancia infinita de la abertura difractora. En esta aproximación, el patrón de amplitud de la onda difractada en el plano de observación es la Transformada de Fourier 2D de la función que describe la abertura.

Matemáticamente, si `A(x, y)` es la función de la abertura (donde `x` e `y` son coordenadas en el plano de la abertura), el campo complejo `U(fx, fy)` en el plano de difracción (donde `fx` y `fy` son frecuencias espaciales) es:

`U(fx, fy) = FFT2D(A(x, y))`

La intensidad del patrón de difracción que se observa es el módulo al cuadrado de este campo complejo:

`I(fx, fy) = |U(fx, fy)|^2`

Este programa utiliza bibliotecas como NumPy y SciPy para calcular la FFT 2D y generar la intensidad del patrón. Para mejorar la visualización de los detalles más débiles, a menudo se aplica una escala logarítmica a la intensidad.

## Requisitos Generales

*   Python 3.x
*   NumPy: `pip install numpy`
*   Matplotlib: `pip install matplotlib`
*   SciPy: `pip install scipy` (usado por la GUI de Matplotlib para `scipy.ndimage.rotate` y `scipy.fft`)

---

## Interfaz Gráfica 1: Basada en Tkinter

Esta interfaz utiliza Tkinter para los controles y Matplotlib para la visualización incrustada.

### Requisitos Adicionales para GUI Tkinter:
*   Tkinter: Generalmente incluido con la instalación estándar de Python. Si no, puede requerir una instalación separada (ej. `sudo apt-get install python3-tk` en sistemas Debian/Ubuntu).

### Instrucciones de Uso (GUI Tkinter)

1.  Asegúrese de tener todos los requisitos generales y Tkinter instalados.
2.  Descargue los archivos `diffraction_calculator.py` (contiene la lógica de cálculo base, aunque la GUI de Matplotlib tiene su propia lógica interna) y `diffraction_gui.py` en el mismo directorio.
3.  Ejecute el programa desde la línea de comandos:
    ```bash
    python diffraction_gui.py
    ```
4.  Se abrirá la ventana de la aplicación "Simulador de Difracción de Fraunhofer".
    *   **Panel de Control (Izquierda):**
        *   **Tipo de Abertura:** Seleccione una forma de abertura (Círculo, Cuadrado, Rendija Simple, Doble Rendija) del menú desplegable.
        *   **Parámetros para [Tipo]:** Ajuste las dimensiones específicas de la abertura seleccionada (ej. radio, lado, ancho/alto de rendija, separación). Los valores suelen estar en "píxeles" de la matriz de simulación.
        *   **Parámetros de Simulación:**
            *   **Resolución (N x N):** Define el tamaño de la matriz de la abertura y del patrón de difracción (ej. 256 para una matriz de 256x256).
            *   **Longitud de Onda (nm):** Parámetro físico conceptual en esta GUI, ya que la escala se basa en píxeles.
        *   **Botón "Calcular y Mostrar Patrón":** Haga clic para actualizar las visualizaciones.
    *   **Panel de Visualización (Derecha):**
        *   **Abertura:** Muestra la forma de la abertura configurada.
        *   **Patrón de Difracción (log):** Muestra el patrón de intensidad calculado.

---

## Interfaz Gráfica 2: Basada en Widgets de Matplotlib

Esta interfaz está construida íntegramente dentro de una figura de Matplotlib, utilizando sus propios widgets (Sliders, RadioButtons) para una experiencia interactiva integrada.

### Instrucciones de Uso (GUI Matplotlib)

1.  Asegúrese de tener todos los requisitos generales instalados (NumPy, Matplotlib, SciPy).
2.  Descargue el archivo `matplotlib_diffraction_gui.py`.
3.  Ejecute el programa desde la línea de comandos:
    ```bash
    python matplotlib_diffraction_gui.py
    ```
4.  Se abrirá una ventana de Matplotlib con la interfaz interactiva:
    *   **Selección de Abertura (Inferior Izquierda):**
        *   Use los `RadioButtons` para elegir el tipo de abertura: Rectangular, Circular, Doble Rendija, Cruz, Triángulo, Anular, o una forma Personalizada (L).
    *   **Sliders de Parámetros (Inferior Central/Derecha):**
        *   Aparecerán sliders relevantes según el tipo de abertura seleccionada. Estos permiten ajustar:
            *   Dimensiones como Ancho/Base, Alto, Radio, Radio Externo/Interno, Separación (en µm).
            *   Parámetros generales: Rotación del ángulo (grados), Longitud de onda λ (nm), Resolución N (puntos de muestreo).
        *   Los plots se actualizan automáticamente al cambiar los sliders.
    *   **Visualizaciones (Paneles Superiores y Centrales):**
        *   **Abertura:** Muestra la forma 2D de la abertura configurada, con ejes en mm.
        *   **Patrón de Difracción 2D:** Muestra el patrón de intensidad 2D (en escala logarítmica) en el plano de observación, con ejes en mm. Incluye un colorbar.
        *   **Corte Horizontal Patrón (y=0):** Perfil 1D del patrón de difracción a lo largo del eje x central.
        *   **Corte Vertical Patrón (x=0):** Perfil 1D del patrón de difracción a lo largo del eje y central.
        *   **Perfil Abertura (y=0):** Corte 1D de la función de transmitancia de la abertura a lo largo del eje x central, con ejes en µm.
        *   **Magnitud FFT Perfil Abertura:** Muestra la magnitud de la Transformada de Fourier del perfil 1D de la abertura, con ejes de frecuencia espacial en mm⁻¹.

### Guardar Resultados (GUI Matplotlib)
*   La clase `FraunhoferDiffractionFFTMatplotlib` en `matplotlib_diffraction_gui.py` tiene un método `save_results(filename='diffraction_results_mpl.png')`. Esto no está conectado a un botón en la GUI, pero puede llamarse desde el script si se desea guardar la figura actual.

---

## Estructura del Código

*   **`diffraction_calculator.py`:** (Principalmente para la GUI Tkinter)
    *   Contiene lógica de cálculo para generar aberturas como matrices de píxeles y calcular patrones de difracción. Usado por `diffraction_gui.py`.

*   **`diffraction_gui.py`:** (GUI Tkinter)
    *   Implementa la interfaz gráfica utilizando Tkinter y Matplotlib incrustado.
    *   Clase `DiffractionApp`.

*   **`matplotlib_diffraction_gui.py`:** (GUI Matplotlib)
    *   Implementa una interfaz gráfica completa utilizando los widgets interactivos de Matplotlib.
    *   Clase `FraunhoferDiffractionFFTMatplotlib`.
    *   Contiene su propia lógica para la creación de aberturas (usando coordenadas físicas) y cálculo de patrones de difracción.
    *   Usa `scipy.fft` para las transformadas y `scipy.ndimage.rotate` para la rotación.

## Posibles Mejoras Futuras (Generales)

*   **Aberturas Personalizadas Avanzadas:** Permitir cargar imágenes o definir formas vectoriales.
*   **Escalado Físico Detallado:** Mayor control sobre las distancias y tamaños de píxel para una correspondencia más precisa con configuraciones experimentales.
*   **Parámetros de Visualización Avanzados:** Controles para mapas de color, rangos dinámicos, etc.
*   **Más Tipos de Aberturas Predefinidas.**
*   **Botón para Guardar en GUI Matplotlib.**

Este programa sirve como una herramienta educativa y de demostración para los principios de la difracción de Fraunhofer.
