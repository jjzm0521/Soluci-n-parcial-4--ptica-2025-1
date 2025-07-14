# Punto 2: Patrón de difracción de Fraunhofer mediante FFT

Esta carpeta contiene el código para calcular y mostrar el patrón de difracción 2D de cualquier abertura en la aproximación de Fraunhofer mediante la transformada de Fourier espacial.

## Archivos

- `diffraction_calculator.py`: Contiene las funciones principales para crear diferentes tipos de aberturas y para calcular el patrón de difracción de Fraunhofer usando la Transformada Rápida de Fourier (FFT).
- `fraunhofer_fft_gui.py`: Una interfaz gráfica de usuario (GUI) construida con `matplotlib` para visualizar los patrones de difracción.
- `fraunhofer_fft_gui_tk.py`: Una GUI alternativa construida con `tkinter`.

## Uso

Para ejecutar la simulación, puede correr cualquiera de los siguientes archivos:

```bash
python punto2/fraunhofer_fft_gui.py
```

o

```bash
python punto2/fraunhofer_fft_gui_tk.py
```
