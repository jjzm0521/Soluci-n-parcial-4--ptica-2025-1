import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.patches as patches # No se usa actualmente, pero puede ser útil
from scipy.fft import fft2, fftshift, fftfreq
from scipy.ndimage import rotate

class FraunhoferDiffractionFFTMatplotlib:
    def __init__(self):
        self.setup_parameters()
        self.setup_plot_layout() # Primero el layout general
        self.create_controls()   # Luego los controles específicos
        self.update_plot()       # Finalmente, el primer plot

    def setup_parameters(self):
        """Configuración inicial de parámetros"""
        # Parámetros físicos
        self.wavelength = 632.8e-9  # Longitud de onda (m) - láser He-Ne
        self.z = 1.0  # Distancia al plano de observación (m)

        # Parámetros de la malla
        self.N = 512  # Número de puntos de muestreo
        self.pixel_size_input = 1e-6  # Tamaño del pixel en el plano de entrada (m) -> Renombrado para claridad
        self.L_input = self.N * self.pixel_size_input  # Tamaño total del plano de entrada

        # Parámetros de la abertura (valores iniciales)
        self.aperture_type = 'rectangular'
        self.width_param = 100e-6  # Ancho de la abertura (m)
        self.height_param = 100e-6  # Alto de la abertura (m)
        self.radius_param = 50e-6  # Radio para abertura circular (m)
        self.r_outer_param = 50e-6 # Radio exterior anular
        self.r_inner_param = 25e-6 # Radio interior anular

        # Parámetros para abertura doble
        self.separation_param = 200e-6  # Separación entre aberturas (m)

        # Parámetros adicionales
        self.rotation_angle = 0.0  # Ángulo de rotación (grados)

        # Crear coordenadas del plano de entrada
        self.x_coords_input = np.linspace(-self.L_input/2, self.L_input/2, self.N)
        self.y_coords_input = np.linspace(-self.L_input/2, self.L_input/2, self.N)
        self.X_input, self.Y_input = np.meshgrid(self.x_coords_input, self.y_coords_input)

        # Coordenadas del plano de Fourier (frecuencias espaciales)
        # fftfreq necesita el espaciado de las muestras (pixel_size_input)
        self.fx_coords = fftshift(fftfreq(self.N, d=self.pixel_size_input))
        self.fy_coords = fftshift(fftfreq(self.N, d=self.pixel_size_input))
        # No es necesario hacer fftshift de nuevo aquí si ya se hizo en fftfreq y luego se aplica a la salida de fft2

        # Coordenadas del plano de observación (escalado por lambda*z)
        # Se recalculan en update_plot porque wavelength puede cambiar
        self.x_coords_obs = self.fx_coords * self.wavelength * self.z
        self.y_coords_obs = self.fy_coords * self.wavelength * self.z

    def create_aperture(self):
        """Crear diferentes tipos de aberturas"""
        aperture = np.zeros((self.N, self.N), dtype=float) # Usar float, no complex si es solo transmitancia 0 o 1

        # Usar los parámetros actuales de la clase
        current_width = self.width_param
        current_height = self.height_param
        current_radius = self.radius_param
        current_separation = self.separation_param
        current_r_outer = self.r_outer_param
        current_r_inner = self.r_inner_param

        if self.aperture_type == 'rectangular':
            mask_x = np.abs(self.X_input) < current_width / 2
            mask_y = np.abs(self.Y_input) < current_height / 2
            aperture[mask_x & mask_y] = 1.0

        elif self.aperture_type == 'circular':
            r = np.sqrt(self.X_input**2 + self.Y_input**2)
            aperture[r <= current_radius] = 1.0

        elif self.aperture_type == 'double_slit':
            # Rendijas verticales por defecto
            # Primera rendija
            mask_x1 = np.abs(self.X_input + current_separation / 2) < current_width / 2
            mask_y1 = np.abs(self.Y_input) < current_height / 2 # height_param aquí es la altura de cada rendija
            aperture[mask_x1 & mask_y1] = 1.0

            # Segunda rendija
            mask_x2 = np.abs(self.X_input - current_separation / 2) < current_width / 2
            mask_y2 = np.abs(self.Y_input) < current_height / 2
            aperture[mask_x2 & mask_y2] = 1.0

        elif self.aperture_type == 'cross':
            # Barra horizontal
            mask_h_x = np.abs(self.X_input) < current_width / 2  # Ancho total de la cruz
            mask_h_y = np.abs(self.Y_input) < current_height / 10 # Grosor de la barra
            aperture[mask_h_x & mask_h_y] = 1.0

            # Barra vertical
            mask_v_x = np.abs(self.X_input) < current_width / 10 # Grosor de la barra
            mask_v_y = np.abs(self.Y_input) < current_height / 2 # Alto total de la cruz
            aperture[mask_v_x & mask_v_y] = 1.0 # Usar OR lógico implícito al asignar 1.0 a regiones
                                            # que podrían solaparse. np.maximum podría ser más explícito si se construyen por separado.

        elif self.aperture_type == 'triangle':
            # Triángulo equilátero centrado, apuntando hacia arriba
            # height_param es la altura total del triángulo
            # width_param es la base del triángulo
            # Coordenadas relativas al centro del triángulo
            # Vértices: (0, H/2), (-B/2, -H/2), (B/2, -H/2)
            # H = current_height, B = current_width

            # Simplificación: Usar un triángulo isósceles donde width_param es la base
            # y height_param es la altura desde la base hasta el vértice opuesto.
            # Centrado en (0,0) significa que la base va de -width/2 a width/2 en x,
            # y el triángulo se extiende desde y=-height/2 hasta y=height/2.

            # Pendiente de los lados
            # Lado derecho: y = (2*H/B) * (x - B/2) -> y - H/2 = m (x - 0) -> y = m*x + H/2
            # Lado izquierdo: y = (-2*H/B) * (x + B/2)
            # Base inferior: y = -H/2

            # Ecuaciones de las líneas que forman los lados (para un triángulo apuntando hacia arriba, base abajo)
            # y_coord = self.Y_input
            # x_coord = self.X_input
            # H = current_height
            # B = current_width

            # Condición para estar dentro del triángulo:
            # 1. y_coord >= -H/2  (Encima de la base)
            # 2. y_coord <= H - (2*H/B) * abs(x_coord) (Debajo de los dos lados inclinados, si está centrado en (0,0))
            #    Esto es para un triángulo con base en y=-H/2 y vértice en y=H/2.
            #    Si H es la altura total y B la base total:
            #    y_coord <= (H/2) - (H/B) * 2 * np.abs(x_coord - 0) -> mal

            # Para un triángulo con y_min = -H/2, y_max = H/2
            # y <= H/2
            # y >= -H/2
            # abs(x) <= (B/2) * ( (H/2 - y) / H )
            # (H/2 - y)/H = 1/2 - y/H

            y = self.Y_input
            x = self.X_input
            H = current_height
            B = current_width

            # Máscara para el triángulo isósceles centrado
            # La base está en y = -H/2, el vértice en y = H/2
            # Los puntos (x,y) deben satisfacer:
            # 1. y >= -H/2
            # 2. y <= H/2
            # 3. Para un y dado, x debe estar entre -x_lim(y) y +x_lim(y)
            #    donde x_lim(y) = (B/2) * ( (H/2 - y) / H ) si H es la altura desde la base al ápice
            #    Si H es la altura total del bounding box:
            #    x_lim = (B/2) * ( (current_height/2 - y) / current_height )
            mask_y_base = y >= -H/2
            mask_y_top  = y <=  H/2

            # Interpolar el ancho permitido en x en función de y
            # Cuando y = -H/2 (base), x_boundary = B/2
            # Cuando y =  H/2 (pico), x_boundary = 0
            x_boundary_at_y = (B/2) * (1 - (y + H/2) / H) # Normalizar y de 0 a H

            mask_x_sides = np.abs(x) <= x_boundary_at_y

            aperture[mask_y_base & mask_y_top & mask_x_sides] = 1.0

        elif self.aperture_type == 'annular':
            r = np.sqrt(self.X_input**2 + self.Y_input**2)
            aperture[(r <= current_r_outer) & (r >= current_r_inner)] = 1.0

        elif self.aperture_type == 'custom': # Ejemplo simple de abertura personalizada
            # Una L simple
            mask_bar1_x = (self.X_input > -current_width/2) & (self.X_input < -current_width/2 + current_width/5)
            mask_bar1_y = (self.Y_input > -current_height/2) & (self.Y_input < current_height/2)
            aperture[mask_bar1_x & mask_bar1_y] = 1.0

            mask_bar2_x = (self.X_input > -current_width/2) & (self.X_input < current_width/2)
            mask_bar2_y = (self.Y_input > -current_height/2) & (self.Y_input < -current_height/2 + current_height/5)
            aperture[mask_bar2_x & mask_bar2_y] = 1.0

        # Aplicar rotación si es necesaria
        if self.rotation_angle != 0:
            aperture = rotate(aperture, self.rotation_angle, reshape=False, order=1, mode='constant', cval=0.0)

        return aperture

    def calculate_diffraction_pattern(self, aperture):
        """Calcular patrón de difracción usando FFT"""
        # Aplicar FFT 2D
        fft_aperture = fft2(aperture) # La entrada ya es solo la parte real (0 o 1)
        fft_aperture_shifted = fftshift(fft_aperture) # Centrar el componente DC

        # Calcular intensidad (|F{aperture}|²)
        intensity = np.abs(fft_aperture_shifted)**2

        # Normalizar para que el máximo sea 1 (opcional, pero bueno para comparación)
        max_intensity = np.max(intensity)
        if max_intensity > 0:
            intensity = intensity / max_intensity

        return intensity

    def setup_plot_layout(self):
        """Configurar la interfaz gráfica y los subplots"""
        self.fig = plt.figure(figsize=(14, 10)) # Ajustar tamaño

        # Usar gridspec para un layout más controlado
        # 3 filas principales para plots, la última más corta.
        # Los controles se posicionan manualmente en la parte inferior.
        gs = self.fig.add_gridspec(3, 3, height_ratios=[2, 2, 1.5], hspace=0.55, wspace=0.35)

        # Fila 1
        self.ax_aperture = self.fig.add_subplot(gs[0, 0])
        self.ax_diffraction2D = self.fig.add_subplot(gs[0, 1])
        self.ax_colorbar = self.fig.add_subplot(gs[0, 2]) # Eje para el colorbar

        # Fila 2
        self.ax_slice_x = self.fig.add_subplot(gs[1, 0])
        self.ax_slice_y = self.fig.add_subplot(gs[1, 1])
        # gs[1, 2] queda vacío o para uso futuro

        # Fila 3
        self.ax_aperture_profile_x = self.fig.add_subplot(gs[2, 0])
        self.ax_aperture_profile_x_fft = self.fig.add_subplot(gs[2, 1])
        # gs[2, 2] queda vacío o para uso futuro

        # Ajustar espacio para que los títulos no se solapen y para los controles inferiores
        plt.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.30)

    def create_controls(self):
        """Crear controles interactivos"""
        # --- RadioButtons para tipo de abertura ---
        # Posición: [left, bottom, width, height] relativo a la figura
        ax_radio = plt.axes([0.05, 0.15, 0.10, 0.12], frameon=True, aspect='equal')
        self.radio_buttons_aperture_type = RadioButtons(
            ax_radio,
            ('Rectangular', 'Circular', 'Doble Rendija', 'Cruz', 'Triángulo', 'Anular', 'Personalizada'),
            active=0 # Rectangular por defecto
        )
        self.radio_buttons_aperture_type.on_clicked(self._update_aperture_type)

        # --- Sliders ---
        slider_col_starts = [0.17, 0.34, 0.51, 0.68, 0.85] # Puntos de inicio para columnas de sliders
        slider_row_bottoms = [0.22, 0.17, 0.12, 0.07, 0.02] # Puntos inferiores para filas de sliders
        slider_width = 0.15 # Ancho común para sliders
        slider_height = 0.025 # Alto común para sliders

        # Columna 1 de Sliders
        self.ax_width_slider = plt.axes([slider_col_starts[0], slider_row_bottoms[0], slider_width, slider_height])
        self.slider_width = Slider(self.ax_width_slider, 'Ancho/Base (µm)', 1, 250, valinit=self.width_param*1e6, valstep=1)
        self.slider_width.on_changed(self._update_plot_from_event)

        self.ax_height_slider = plt.axes([slider_col_starts[0], slider_row_bottoms[1], slider_width, slider_height])
        self.slider_height = Slider(self.ax_height_slider, 'Alto (µm)', 1, 250, valinit=self.height_param*1e6, valstep=1)
        self.slider_height.on_changed(self._update_plot_from_event)

        self.ax_radius_slider = plt.axes([slider_col_starts[0], slider_row_bottoms[2], slider_width, slider_height])
        self.slider_radius = Slider(self.ax_radius_slider, 'Radio (µm)', 1, 150, valinit=self.radius_param*1e6, valstep=1)
        self.slider_radius.on_changed(self._update_plot_from_event)

        # Columna 2 de Sliders
        self.ax_r_outer_slider = plt.axes([slider_col_starts[1], slider_row_bottoms[0], slider_width, slider_height])
        self.slider_r_outer = Slider(self.ax_r_outer_slider, 'Radio Ext (µm)', 1, 150, valinit=self.r_outer_param*1e6, valstep=1)
        self.slider_r_outer.on_changed(self._update_plot_from_event)

        self.ax_r_inner_slider = plt.axes([slider_col_starts[1], slider_row_bottoms[1], slider_width, slider_height])
        self.slider_r_inner = Slider(self.ax_r_inner_slider, 'Radio Int (µm)', 1, 149, valinit=self.r_inner_param*1e6, valstep=1)
        self.slider_r_inner.on_changed(self._update_plot_from_event)

        self.ax_separation_slider = plt.axes([slider_col_starts[1], slider_row_bottoms[2], slider_width, slider_height])
        self.slider_separation = Slider(self.ax_separation_slider, 'Separación (µm)', 10, 500, valinit=self.separation_param*1e6, valstep=5)
        self.slider_separation.on_changed(self._update_plot_from_event)

        # Columna 3 de Sliders (Parámetros Generales)
        self.ax_rotation_slider = plt.axes([slider_col_starts[2], slider_row_bottoms[0], slider_width, slider_height])
        self.slider_rotation = Slider(self.ax_rotation_slider, 'Rotación (°)', 0, 360, valinit=self.rotation_angle, valstep=1)
        self.slider_rotation.on_changed(self._update_plot_from_event)

        self.ax_lambda_slider = plt.axes([slider_col_starts[2], slider_row_bottoms[1], slider_width, slider_height])
        self.slider_lambda = Slider(self.ax_lambda_slider, 'λ (nm)', 400, 800, valinit=self.wavelength*1e9, valstep=10)
        self.slider_lambda.on_changed(self._update_plot_from_event)

        self.ax_N_slider = plt.axes([slider_col_starts[2], slider_row_bottoms[2], slider_width, slider_height])
        self.slider_N = Slider(self.ax_N_slider, 'Resolución N', 64, 1024, valinit=self.N, valstep=64)
        self.slider_N.on_changed(self._update_resolution)

        self._toggle_slider_visibility() # Aplicar visibilidad inicial

    def _update_aperture_type(self, label):
        aperture_map = {
            'Rectangular': 'rectangular', 'Circular': 'circular', 'Doble Rendija': 'double_slit',
            'Cruz': 'cross', 'Triángulo': 'triangle', 'Anular': 'annular', 'Personalizada': 'custom'
        }
        self.aperture_type = aperture_map[label]
        self._toggle_slider_visibility()
        self.update_plot() # Actualizar el plot cuando cambia el tipo

    def _toggle_slider_visibility(self):
        """Gestiona la visibilidad de los sliders según el tipo de abertura."""
        sliders_config = {
            'rectangular': [self.ax_width_slider, self.ax_height_slider],
            'circular': [self.ax_radius_slider],
            'double_slit': [self.ax_width_slider, self.ax_height_slider, self.ax_separation_slider],
            'cross': [self.ax_width_slider, self.ax_height_slider],
            'triangle': [self.ax_width_slider, self.ax_height_slider],
            'annular': [self.ax_r_outer_slider, self.ax_r_inner_slider],
            'custom': [self.ax_width_slider, self.ax_height_slider] # Ejemplo: custom usa width/height
        }

        all_specific_slider_axes = [ # Sliders que dependen del tipo de abertura
            self.ax_width_slider, self.ax_height_slider, self.ax_radius_slider,
            self.ax_r_outer_slider, self.ax_r_inner_slider, self.ax_separation_slider
        ]

        for ax_s in all_specific_slider_axes:
            ax_s.set_visible(False)

        if self.aperture_type in sliders_config:
            for ax_s in sliders_config[self.aperture_type]:
                ax_s.set_visible(True)

        self.fig.canvas.draw_idle()

    def _update_resolution(self, val):
        # Actualizar N y recalcular todas las coordenadas dependientes
        self.N = int(self.slider_N.val)
        if self.N % 2 != 0: # Asegurar que N sea par para facilitar centros
            self.N = int(self.N // 2 * 2)
            self.slider_N.set_val(self.N) # Actualizar el slider si se ajustó N

        self.L_input = self.N * self.pixel_size_input
        self.x_coords_input = np.linspace(-self.L_input/2, self.L_input/2, self.N)
        self.y_coords_input = np.linspace(-self.L_input/2, self.L_input/2, self.N)
        self.X_input, self.Y_input = np.meshgrid(self.x_coords_input, self.y_coords_input)

        self.fx_coords = fftshift(fftfreq(self.N, d=self.pixel_size_input))
        self.fy_coords = fftshift(fftfreq(self.N, d=self.pixel_size_input))

        self.update_plot() # Re-plotear todo

    def _update_plot_from_event(self, val=None):
        """Wrapper para llamar a update_plot desde eventos de widgets"""
        self.update_plot()

    def update_plot(self):
        """Actualizar todos los gráficos"""
        # Actualizar parámetros desde sliders
        self.width_param = self.slider_width.val * 1e-6
        self.height_param = self.slider_height.val * 1e-6
        self.radius_param = self.slider_radius.val * 1e-6
        self.r_outer_param = self.slider_r_outer.val * 1e-6
        self.r_inner_param = self.slider_r_inner.val * 1e-6
        if self.r_inner_param >= self.r_outer_param and self.aperture_type == 'annular': # Evitar radio interior > exterior
            # Solo ajustar si el slider está visible y es relevante
            if self.ax_r_inner_slider.get_visible():
                 self.r_inner_param = max(1e-6, self.r_outer_param - 1e-6) # Asegurar que no sea negativo o cero
                 self.slider_r_inner.set_val(self.r_inner_param * 1e6)

        self.separation_param = self.slider_separation.val * 1e-6
        self.rotation_angle = self.slider_rotation.val
        self.wavelength = self.slider_lambda.val * 1e-9

        # Recalcular coordenadas del plano de observación (dependen de wavelength y N a través de fx, fy)
        self.x_coords_obs = self.fx_coords * self.wavelength * self.z
        self.y_coords_obs = self.fy_coords * self.wavelength * self.z

        # Crear abertura
        aperture_field = self.create_aperture()

        # Calcular patrón de difracción
        diffraction_intensity = self.calculate_diffraction_pattern(aperture_field)

        # Limpiar ejes antes de redibujar
        self.ax_aperture.clear()
        self.ax_diffraction2D.clear()
        self.ax_slice_x.clear()
        self.ax_slice_y.clear()
        self.ax_aperture_profile_x.clear()
        self.ax_aperture_profile_x_fft.clear()
        # No es necesario limpiar self.ax_colorbar si se va a redibujar con fig.colorbar

        # --- Plot Abertura ---
        extent_input_mm = [
            self.x_coords_input[0]*1e3, self.x_coords_input[-1]*1e3,
            self.y_coords_input[0]*1e3, self.y_coords_input[-1]*1e3
        ]
        self.ax_aperture.imshow(np.abs(aperture_field), extent=extent_input_mm,
                                cmap='gray', origin='lower', interpolation='nearest')
        self.ax_aperture.set_title(f'Abertura: {self.radio_buttons_aperture_type.value_selected}') # Usar valor del RadioButton
        self.ax_aperture.set_xlabel('x_input (mm)')
        self.ax_aperture.set_ylabel('y_input (mm)')

        # --- Plot Patrón de Difracción 2D ---
        extent_obs_mm = [
            self.x_coords_obs[0]*1e3, self.x_coords_obs[-1]*1e3,
            self.y_coords_obs[0]*1e3, self.y_coords_obs[-1]*1e3
        ]
        log_intensity = np.log10(diffraction_intensity + 1e-12) # Epsilon pequeño

        # Limpiar el eje del colorbar antes de redibujar el colorbar mismo
        if hasattr(self, 'colorbar_instance') and self.colorbar_instance:
             self.colorbar_instance.remove()

        im = self.ax_diffraction2D.imshow(log_intensity, extent=extent_obs_mm,
                                          cmap='hot', origin='lower', interpolation='bilinear')
        self.ax_diffraction2D.set_title('Patrón Difracción 2D')
        self.ax_diffraction2D.set_xlabel('x_obs (mm)')
        self.ax_diffraction2D.set_ylabel('y_obs (mm)')

        self.colorbar_instance = self.fig.colorbar(im, cax=self.ax_colorbar, orientation='vertical')
        self.ax_colorbar.set_ylabel('Log10(Intensidad Norm.)') # Título para el colorbar

        # --- Plot Cortes 1D del Patrón de Difracción ---
        center_idx = self.N // 2
        profile_diff_x = diffraction_intensity[center_idx, :]
        self.ax_slice_x.plot(self.x_coords_obs * 1e3, profile_diff_x, 'r-', linewidth=1.5)
        self.ax_slice_x.set_title('Corte Horizontal Patrón (y=0)')
        self.ax_slice_x.set_xlabel('x_obs (mm)')
        self.ax_slice_x.set_ylabel('Intensidad Norm.')
        self.ax_slice_x.grid(True, linestyle=':', alpha=0.7)
        max_val_x = np.max(profile_diff_x) if np.any(profile_diff_x) else 1.0
        self.ax_slice_x.set_ylim(0, max_val_x * 1.1 if max_val_x > 0 else 0.1)


        profile_diff_y = diffraction_intensity[:, center_idx]
        self.ax_slice_y.plot(self.y_coords_obs * 1e3, profile_diff_y, 'b-', linewidth=1.5)
        self.ax_slice_y.set_title('Corte Vertical Patrón (x=0)')
        self.ax_slice_y.set_xlabel('y_obs (mm)')
        self.ax_slice_y.set_ylabel('Intensidad Norm.')
        self.ax_slice_y.grid(True, linestyle=':', alpha=0.7)
        max_val_y = np.max(profile_diff_y) if np.any(profile_diff_y) else 1.0
        self.ax_slice_y.set_ylim(0, max_val_y * 1.1 if max_val_y > 0 else 0.1)

        # --- Plot Perfil de Abertura y su FFT ---
        aperture_profile_x_data = np.abs(aperture_field[center_idx, :])
        self.ax_aperture_profile_x.plot(self.x_coords_input * 1e6, aperture_profile_x_data, 'k-', linewidth=1.5)
        self.ax_aperture_profile_x.set_title('Perfil Abertura (y=0)')
        self.ax_aperture_profile_x.set_xlabel('x_input (µm)')
        self.ax_aperture_profile_x.set_ylabel('Transmitancia')
        self.ax_aperture_profile_x.grid(True, linestyle=':', alpha=0.7)
        self.ax_aperture_profile_x.set_ylim(-0.1, 1.1)

        # FFT de este perfil 1D
        # Para un array 1D real, fft es suficiente. fftshift para centrar.
        fft_abs_profile = np.abs(fftshift(np.fft.fft(aperture_profile_x_data)))
        fx_coords_1D = fftshift(fftfreq(self.N, d=self.pixel_size_input))

        self.ax_aperture_profile_x_fft.plot(fx_coords_1D * 1e-3, fft_abs_profile, 'g-', linewidth=1.5)
        self.ax_aperture_profile_x_fft.set_title('Magnitud FFT Perfil Abertura')
        self.ax_aperture_profile_x_fft.set_xlabel('fx (mm⁻¹)')
        self.ax_aperture_profile_x_fft.set_ylabel('|F{perfil}|')
        self.ax_aperture_profile_x_fft.grid(True, linestyle=':', alpha=0.7)
        if np.any(fft_abs_profile): self.ax_aperture_profile_x_fft.set_ylim(0, np.max(fft_abs_profile)*1.1)


        self.fig.canvas.draw_idle()

    def save_results(self, filename='diffraction_results_mpl.png'):
        """Guardar resultados"""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Resultados guardados en {filename}")

# Función para ejemplos de validación
def validation_examples():
    """Ejemplos para validar los resultados"""
    print("Ejemplos de validación:")
    print("1. Abertura rectangular: patrón sinc²")
    print("2. Abertura circular: patrón de Airy")
    print("3. Doble rendija: franjas de interferencia moduladas por difracción de rendija única")
    print("4. Considerar la escala: cómo cambian los patrones con la longitud de onda y las dimensiones.")

# Ejecutar aplicación
if __name__ == "__main__":
    app = FraunhoferDiffractionFFTMatplotlib()
    validation_examples()
    plt.show()
