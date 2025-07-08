# Simulador de Difracción de Fraunhofer 2D

Este programa calcula y visualiza el patrón de difracción de Fraunhofer para varias aberturas 2D utilizando la Transformada Rápida de Fourier (FFT). Incluye una interfaz gráfica de usuario (GUI) para seleccionar tipos de aberturas, ajustar sus parámetros y ver los resultados en tiempo real.

## Principios Teóricos

La difracción de Fraunhofer (o difracción de campo lejano) ocurre cuando tanto la fuente de luz como la pantalla de observación están efectivamente a una distancia infinita de la abertura difractora. En esta aproximación, el patrón de amplitud de la onda difractada en el plano de observación es la Transformada de Fourier 2D de la función que describe la abertura.

Matemáticamente, si `A(x, y)` es la función de la abertura (donde `x` e `y` son coordenadas en el plano de la abertura), el campo complejo `U(fx, fy)` en el plano de difracción (donde `fx` y `fy` son frecuencias espaciales) es:

`U(fx, fy) = FFT2D(A(x, y))`

La intensidad del patrón de difracción que se observa es el módulo al cuadrado de este campo complejo:

`I(fx, fy) = |U(fx, fy)|^2`

Este programa utiliza la biblioteca NumPy para calcular la FFT 2D y generar la intensidad del patrón. Para mejorar la visualización de los detalles más débiles, a menudo se aplica una escala logarítmica a la intensidad.

## Requisitos

*   Python 3.x
*   NumPy: `pip install numpy`
*   Matplotlib: `pip install matplotlib`
*   Tkinter: Generalmente incluido con la instalación estándar de Python. Si no, puede requerir una instalación separada (ej. `sudo apt-get install python3-tk` en sistemas Debian/Ubuntu).

## Instrucciones de Uso

1.  Asegúrese de tener todos los requisitos instalados.
2.  Clone el repositorio o descargue los archivos `diffraction_calculator.py` y `diffraction_gui.py` en el mismo directorio.
3.  Ejecute el programa desde la línea de comandos:
    ```bash
    python diffraction_gui.py
    ```
4.  Se abrirá la ventana de la aplicación "Simulador de Difracción de Fraunhofer".
    *   **Panel de Control (Izquierda):**
        *   **Tipo de Abertura:** Seleccione una forma de abertura (Círculo, Cuadrado, Rendija Simple, Doble Rendija) del menú desplegable.
        *   **Parámetros para [Tipo]:** Ajuste las dimensiones específicas de la abertura seleccionada (ej. radio, lado, ancho/alto de rendija, separación). Los valores suelen estar en "píxeles" de la matriz de simulación.
        *   **Parámetros de Simulación:**
            *   **Resolución (N x N):** Define el tamaño de la matriz de la abertura y del patrón de difracción (ej. 256 para una matriz de 256x256). Valores más altos dan más detalle pero requieren más cálculo.
            *   **Longitud de Onda (nm):** Parámetro físico. En la implementación actual, no escala directamente el tamaño del patrón de difracción en la FFT (que opera en el espacio de píxeles/frecuencias espaciales relativas), pero se incluye por completitud conceptual.
        *   **Botón "Calcular y Mostrar Patrón":** Haga clic para actualizar las visualizaciones después de cambiar los parámetros.
    *   **Panel de Visualización (Derecha):**
        *   **Abertura:** Muestra la forma de la abertura configurada.
        *   **Patrón de Difracción (log):** Muestra el patrón de intensidad calculado, con una escala logarítmica para mejorar la visibilidad de los máximos secundarios.

## Estructura del Código

El proyecto consta de dos archivos Python principales:

*   **`diffraction_calculator.py`:**
    *   Contiene la lógica central para los cálculos.
    *   `create_circular_aperture(...)`, `create_square_aperture(...)`, `create_single_slit_aperture(...)`, `create_double_slit_aperture(...)`: Funciones para generar matrices 2D (NumPy arrays) que representan las diferentes formas de aberturas.
    *   `calculate_diffraction_pattern(...)`: Toma una matriz de abertura, calcula su FFT 2D, aplica `fftshift` para centrar el componente de frecuencia cero, calcula la intensidad (módulo al cuadrado), y opcionalmente aplica una escala logarítmica.
    *   Incluye un bloque `if __name__ == '__main__':` para pruebas directas de las funciones de cálculo con Matplotlib (sin GUI).

*   **`diffraction_gui.py`:**
    *   Implementa la interfaz gráfica de usuario utilizando Tkinter y Matplotlib.
    *   Clase `DiffractionApp`: Gestiona la ventana principal, los widgets de control (menús, campos de entrada, botones) y la incrustación de los gráficos de Matplotlib.
    *   Interactúa con `diffraction_calculator.py` para obtener los datos de la abertura y el patrón de difracción.
    *   Maneja la actualización dinámica de la interfaz y las visualizaciones en respuesta a las acciones del usuario.

## Posibles Mejoras Futuras

*   **Aberturas Personalizadas:** Permitir al usuario cargar una imagen en blanco y negro para usarla como abertura.
*   **Escalado Físico:** Incorporar la longitud de onda y la distancia a la pantalla para escalar los ejes del patrón de difracción a unidades físicas (ej. milímetros en la pantalla).
*   **Parámetros de Visualización:** Controles para el mapa de colores, rango dinámico del patrón de difracción, etc.
*   **Perfiles de Intensidad:** Graficar cortes transversales del patrón de difracción para un análisis cuantitativo.
*   **Optimización:** Para resoluciones muy altas, explorar optimizaciones o el uso de bibliotecas FFT más rápidas si es necesario (aunque NumPy FFT es generalmente muy eficiente).
*   **Más Tipos de Aberturas:** Añadir aberturas como rejillas de difracción, triángulos, etc.
*   **Guardar Resultados:** Opción para guardar las imágenes de la abertura y el patrón de difracción.

Este programa sirve como una herramienta educativa y de demostración para los principios de la difracción de Fraunhofer.
