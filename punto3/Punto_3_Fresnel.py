import numpy as np
import matplotlib.pyplot as plt
from scipy.special import fresnel
from matplotlib.widgets import Slider
import warnings
warnings.filterwarnings('ignore')

def calcular_patron_fresnel(u, N):
    """
    Calcula la intensidad del patrón de difracción de Fresnel para una rendija.

    Args:
        u (np.ndarray): Arreglo de coordenadas normalizadas en la pantalla (u = a*y / (lambda*z)).
        N (float): El número de Fresnel (N = a^2 / (lambda*z)).

    Returns:
        np.ndarray: La intensidad normalizada del patrón de difracción.
    """
    # Para N muy pequeño, usar directamente el patrón de Fraunhofer
    if N < 1e-6:
        return np.sinc(u)**2

    # Argumentos para las integrales de Fresnel
    # Usar mayor precisión en los cálculos
    sqrt_N_2 = np.sqrt(N / 2.0)
    sqrt_2_N = np.sqrt(2.0 / N)
    
    arg1 = sqrt_N_2 - u * sqrt_2_N
    arg2 = -sqrt_N_2 - u * sqrt_2_N

    # Calcular las integrales de Fresnel
    S1, C1 = fresnel(arg1)
    S2, C2 = fresnel(arg2)

    # La intensidad es proporcional al módulo al cuadrado de la amplitud compleja
    intensidad = (C1 - C2)**2 + (S1 - S2)**2
    
    # Normalización más robusta
    S0_pos, C0_pos = fresnel(sqrt_N_2)
    S0_neg, C0_neg = fresnel(-sqrt_N_2)
    intensidad_central = (C0_pos - C0_neg)**2 + (S0_pos - S0_neg)**2

    # Evitar división por cero y asegurar normalización correcta
    if intensidad_central < 1e-12:
        return intensidad
    
    return intensidad / intensidad_central

def calcular_parametros_opticos(N, lambda_luz=632.8e-9, z=1.0):
    """
    Calcula los parámetros ópticos a partir del número de Fresnel.
    
    Args:
        N (float): Número de Fresnel
        lambda_luz (float): Longitud de onda en metros (por defecto He-Ne: 632.8 nm)
        z (float): Distancia a la pantalla en metros
    
    Returns:
        dict: Diccionario con los parámetros calculados
    """
    a = np.sqrt(N * lambda_luz * z)  # Ancho de la rendija
    return {
        'ancho_rendija_mm': a * 1000,
        'longitud_onda_nm': lambda_luz * 1e9,
        'distancia_pantalla_m': z
    }

def graficar_difraccion_interactiva():
    """
    Genera una gráfica interactiva con un deslizador para controlar
    el número de Fresnel y observar la evolución del patrón de difracción.
    """
    # --- Configuración Inicial ---
    # Aumentar resolución para mayor precisión
    u = np.linspace(-6, 6, 2000)
    N_inicial = 10.0
    
    # Rango más amplio para explorar mejor la transición
    logN_min = np.log10(0.01)
    logN_max = np.log10(100.0)
    logN_inicial = np.log10(N_inicial)

    # --- Creación de la Figura y los Ejes ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.25, hspace=0.3)

    # --- Gráfica Principal (Patrón de Intensidad) ---
    # 1. Patrón de Fraunhofer (referencia estática)
    intensidad_fraunhofer = np.sinc(u)**2
    ax1.plot(u, intensidad_fraunhofer, 'k--', linewidth=2, label='Fraunhofer (N → 0)')

    # 2. Patrón de Fresnel inicial
    intensidad_fresnel_inicial = calcular_patron_fresnel(u, N_inicial)
    linea_fresnel, = ax1.plot(u, intensidad_fresnel_inicial, 'r-', lw=2, label='Fresnel')
    
    # Configuración de la gráfica principal
    ax1.set_xlabel(r'Coordenada Normalizada, $u = \frac{ay}{\lambda z}$', fontsize=12)
    ax1.set_ylabel('Intensidad Normalizada', fontsize=12)
    ax1.set_ylim(-0.05, 3.5)
    ax1.set_xlim(u.min(), u.max())
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    title1 = ax1.set_title(f'Patrón de Difracción para N = {N_inicial:.2f}', fontsize=14)

    # --- Gráfica Secundaria (Parámetros Físicos) ---
    params = calcular_parametros_opticos(N_inicial)
    ax2.text(0.05, 0.8, f"Número de Fresnel: {N_inicial:.3f}", transform=ax2.transAxes, fontsize=12)
    ax2.text(0.05, 0.6, f"Ancho de rendija: {params['ancho_rendija_mm']:.3f} mm", transform=ax2.transAxes, fontsize=12)
    ax2.text(0.05, 0.4, f"Longitud de onda: {params['longitud_onda_nm']:.1f} nm", transform=ax2.transAxes, fontsize=12)
    ax2.text(0.05, 0.2, f"Distancia a pantalla: {params['distancia_pantalla_m']:.1f} m", transform=ax2.transAxes, fontsize=12)
    
    # Añadir información sobre el régimen de difracción
    regime_text = ax2.text(0.55, 0.8, "", transform=ax2.transAxes, fontsize=12, 
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Parámetros Físicos del Sistema', fontsize=14)

    # --- Creación del Deslizador ---
    ax_slider = fig.add_axes([0.2, 0.1, 0.65, 0.03])
    
    slider_N = Slider(
        ax=ax_slider,
        label='log_10(N)',
        valmin=logN_min,
        valmax=logN_max,
        valinit=logN_inicial,
        color='darkred'
    )

    # --- Función de Actualización ---
    def update(val):
        logN = slider_N.val
        N = 10**logN
        
        # Recalcular el patrón de Fresnel
        intensidad_actualizada = calcular_patron_fresnel(u, N)
        linea_fresnel.set_ydata(intensidad_actualizada)
        
        # Actualizar título
        title1.set_text(f'Patrón de Difracción para N = {N:.3f}')
        
        # Actualizar parámetros físicos
        params = calcular_parametros_opticos(N)
        ax2.clear()
        ax2.text(0.05, 0.8, f"Número de Fresnel: {N:.3f}", transform=ax2.transAxes, fontsize=12)
        ax2.text(0.05, 0.6, f"Ancho de rendija: {params['ancho_rendija_mm']:.3f} mm", transform=ax2.transAxes, fontsize=12)
        ax2.text(0.05, 0.4, f"Longitud de onda: {params['longitud_onda_nm']:.1f} nm", transform=ax2.transAxes, fontsize=12)
        ax2.text(0.05, 0.2, f"Distancia a pantalla: {params['distancia_pantalla_m']:.1f} m", transform=ax2.transAxes, fontsize=12)
        
        # Determinar régimen de difracción
        if N < 0.1:
            regime = "Régimen de Fraunhofer\n(Campo lejano)"
            color = "lightgreen"
        elif N < 1.0:
            regime = "Transición\nFresnel-Fraunhofer"
            color = "lightyellow"
        else:
            regime = "Régimen de Fresnel\n(Campo cercano)"
            color = "lightcoral"
        
        ax2.text(0.55, 0.7, regime, transform=ax2.transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('Parámetros Físicos del Sistema', fontsize=14)
        
        fig.canvas.draw_idle()

    # --- Conexión del Deslizador ---
    slider_N.on_changed(update)

    # Añadir información adicional
    fig.suptitle('Evolución del Patrón de Difracción: Fresnel → Fraunhofer', fontsize=16, y=0.95)
    
    # Instrucciones para el usuario
    fig.text(0.5, 0.02, 'Desliza para observar la transición del patrón de Fresnel (N grande) al patrón de Fraunhofer (N pequeño)', 
             ha='center', fontsize=10, style='italic')

    plt.show()

def generar_comparacion_estatica():
    """
    Genera una comparación estática de varios valores de N para documentación.
    """
    u = np.linspace(-4, 4, 1000)
    N_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, N in enumerate(N_values):
        ax = axes[i]
        
        # Patrón de Fraunhofer
        intensidad_fraunhofer = np.sinc(u)**2
        ax.plot(u, intensidad_fraunhofer, 'k--', alpha=0.5, label='Fraunhofer')
        
        # Patrón de Fresnel
        intensidad_fresnel = calcular_patron_fresnel(u, N)
        ax.plot(u, intensidad_fresnel, 'r-', linewidth=2, label='Fresnel')
        
        ax.set_title(f'N = {N}', fontsize=12)
        ax.set_ylim(0, 3)
        ax.grid(True, alpha=0.3)
        
        if i >= 4:
            ax.set_xlabel(r'$u = \frac{ay}{\lambda z}$')
        if i % 4 == 0:
            ax.set_ylabel('Intensidad')
        
        if i == 0:
            ax.legend()
    
    # Remover el último subplot vacío
    axes[-1].remove()
    
    plt.tight_layout()
    plt.suptitle('Comparación Estática: Evolución del Patrón de Difracción', fontsize=16, y=0.98)
    plt.show()

# --- Ejecutar el programa ---
if __name__ == '__main__':
    print("1. Gráfica interactiva")
    print("2. Comparación estática")
    opcion = input("Selecciona una opción (1 o 2): ")
    
    if opcion == "2":
        generar_comparacion_estatica()
    else:
        graficar_difraccion_interactiva()