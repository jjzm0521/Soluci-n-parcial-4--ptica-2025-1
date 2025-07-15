import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.fft import fft2, fftshift
from matplotlib.patches import Circle, Rectangle, Ellipse
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

class DiffractionSimulator:
    def __init__(self):
        # Configurar estilo
        plt.style.use('seaborn-v0_8-whitegrid')
        
        self.fig = plt.figure(figsize=(16, 10))
        
        # Crear layout con mejor distribución
        gs = self.fig.add_gridspec(2, 3, 
                                 height_ratios=[3, 1], 
                                 width_ratios=[1, 1, 1],
                                 hspace=0.3, wspace=0.3)
        
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Abertura
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # Patrón de difracción
        self.ax3 = self.fig.add_subplot(gs[0, 2])  # Perfil 1D
        
        self.fig.suptitle('Simulador de Difracción 2D - Aproximación de Fraunhofer', 
                         fontsize=18, y=0.95)
        
        # Parámetros iniciales
        self.wavelength = 550e-9  # 550 nm en metros
        self.current_scene = 'Circle + Square'
        self.matrix_size = 256  # Reducido para mejor rendimiento
        self.pixel_size = 2e-5  # 20 micrómetros por pixel
        
        # Parámetros específicos para cada abertura
        self.circle_diameter = 2e-3  # 2 mm
        self.square_width = 1.5e-3   # 1.5 mm
        self.ellipse_diameter = 2e-3
        self.ellipse_eccentricity = 0.5
        self.rect_width = 1e-3
        self.rect_height = 2e-3
        self.disorder_circle_diameter = 0.1e-3
        self.disorder_spacing = 0.5e-3
        self.disorder_amount = 0
        self.girl_height = 2e-3
        self.girl_rotation = 0
        self.cbar = None
        
        # Crear mapa de colores personalizado basado en longitud de onda
        self.setup_colormap()
        self.setup_ui()
        self.update_pattern()
        
    def setup_colormap(self):
        """Crear mapa de colores basado en la longitud de onda"""
        # Convertir longitud de onda a color RGB aproximado
        wavelength_nm = self.wavelength * 1e9
        
        if wavelength_nm < 440:
            r, g, b = 0.5, 0.0, 1.0
        elif wavelength_nm < 490:
            r = (490 - wavelength_nm) / (490 - 440)
            g = 0.0
            b = 1.0
        elif wavelength_nm < 510:
            r = 0.0
            g = (wavelength_nm - 490) / (510 - 490)
            b = 1.0
        elif wavelength_nm < 580:
            r = 0.0
            g = 1.0
            b = (580 - wavelength_nm) / (580 - 510)
        elif wavelength_nm < 645:
            r = (wavelength_nm - 580) / (645 - 580)
            g = 1.0
            b = 0.0
        else:
            r = 1.0
            g = (750 - wavelength_nm) / (750 - 645)
            b = 0.0
        
        # Crear colormap personalizado
        colors = [(0, 0, 0), (r*0.3, g*0.3, b*0.3), (r*0.7, g*0.7, b*0.7), (r, g, b)]
        n_bins = 256
        self.custom_cmap = LinearSegmentedColormap.from_list('wavelength', colors, N=n_bins)
        
    def setup_ui(self):
        # Ajustar espacio para controles
        plt.subplots_adjust(bottom=0.25, right=0.85)
        
        # Selector de escena con mejor estética
        scene_ax = plt.axes([0.02, 0.72, 0.18, 0.2])
        self.scene_selector = RadioButtons(scene_ax, 
            ('Circle + Square', 'Ellipse', 'Rectangle', 'Disorder', 'Waving Girl'))
        self.scene_selector.on_clicked(self.change_scene)
        
        # Longitud de onda (global) con mejor estética
        wavelength_ax = plt.axes([0.25, 0.15, 0.5, 0.03])
        self.wavelength_slider = Slider(
            wavelength_ax, 'Longitud de onda (nm)', 400, 700, 
            valinit=550, valfmt='%.0f nm',
            color='darkred'
        )
        self.wavelength_slider.on_changed(self.update_wavelength)
        
        # Sliders específicos para cada escena con mejor estética
        self.setup_scene_controls()
        
        # Botones con mejor estética
        reset_ax = plt.axes([0.8, 0.02, 0.1, 0.05])
        self.reset_button = Button(reset_ax, 'Reset')
        self.reset_button.on_clicked(self.reset_values)
        
        # Configurar axes con mejor estética
        self.setup_axes_style()
        
    def setup_axes_style(self):
        """Configurar estilo de los axes"""
        axes = [self.ax1, self.ax2, self.ax3]
        titles = ['Abertura', 'Patrón de Difracción', 'Perfil Central']
        
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=14, pad=20)
            ax.tick_params(labelsize=10)
            ax.grid(True, alpha=0.3)
        
    def setup_scene_controls(self):
        slider_color = 'darkred'
        
        # Circle + Square controls
        self.circle_diameter_ax = plt.axes([0.25, 0.10, 0.3, 0.03])
        self.circle_diameter_slider = Slider(
            self.circle_diameter_ax, 'Diámetro círculo (mm)', 0.04, 15, 
            valinit=2, valfmt='%.2f mm', color=slider_color
        )
        self.circle_diameter_slider.on_changed(self.update_circle_diameter)
        
        self.square_width_ax = plt.axes([0.25, 0.05, 0.3, 0.03])
        self.square_width_slider = Slider(
            self.square_width_ax, 'Ancho cuadrado (mm)', 0.04, 15, 
            valinit=1.5, valfmt='%.2f mm', color=slider_color
        )
        self.square_width_slider.on_changed(self.update_square_width)
        
        # Ellipse controls
        self.ellipse_diameter_ax = plt.axes([0.6, 0.10, 0.3, 0.03])
        self.ellipse_diameter_slider = Slider(
            self.ellipse_diameter_ax, 'Diámetro elipse (mm)', 0.04, 4, 
            valinit=2, valfmt='%.2f mm', color=slider_color
        )
        self.ellipse_diameter_slider.on_changed(self.update_ellipse_diameter)
        
        self.ellipse_eccentricity_ax = plt.axes([0.6, 0.05, 0.3, 0.03])
        self.ellipse_eccentricity_slider = Slider(
            self.ellipse_eccentricity_ax, 'Excentricidad', 0, 0.99, 
            valinit=0.5, valfmt='%.2f', color=slider_color
        )
        self.ellipse_eccentricity_slider.on_changed(self.update_ellipse_eccentricity)
        
        # Rectangle controls
        self.rect_width_ax = plt.axes([0.25, 0.20, 0.3, 0.03])
        self.rect_width_slider = Slider(
            self.rect_width_ax, 'Ancho rect (mm)', 0.04, 4, 
            valinit=1, valfmt='%.2f mm', color=slider_color
        )
        self.rect_width_slider.on_changed(self.update_rect_width)
        
        self.rect_height_ax = plt.axes([0.6, 0.20, 0.3, 0.03])
        self.rect_height_slider = Slider(
            self.rect_height_ax, 'Alto rect (mm)', 0.04, 4, 
            valinit=2, valfmt='%.2f mm', color=slider_color
        )
        self.rect_height_slider.on_changed(self.update_rect_height)
        
        # Disorder controls
        self.disorder_diameter_ax = plt.axes([0.25, 0.25, 0.3, 0.03])
        self.disorder_diameter_slider = Slider(
            self.disorder_diameter_ax, 'Diámetro círculos (mm)', 0.01, 1, 
            valinit=0.1, valfmt='%.2f mm', color=slider_color
        )
        self.disorder_diameter_slider.on_changed(self.update_disorder_diameter)
        
        self.disorder_spacing_ax = plt.axes([0.6, 0.25, 0.3, 0.03])
        self.disorder_spacing_slider = Slider(
            self.disorder_spacing_ax, 'Espaciado rejilla (mm)', 0.05, 2, 
            valinit=0.5, valfmt='%.2f mm', color=slider_color
        )
        self.disorder_spacing_slider.on_changed(self.update_disorder_spacing)
        
        # Waving Girl controls
        self.girl_height_ax = plt.axes([0.25, 0.30, 0.3, 0.03])
        self.girl_height_slider = Slider(
            self.girl_height_ax, 'Altura figura (mm)', 0.04, 4, 
            valinit=2, valfmt='%.2f mm', color=slider_color
        )
        self.girl_height_slider.on_changed(self.update_girl_height)
        
        self.girl_rotation_ax = plt.axes([0.6, 0.30, 0.3, 0.03])
        self.girl_rotation_slider = Slider(
            self.girl_rotation_ax, 'Rotación (°)', 0, 360, 
            valinit=0, valfmt='%.0f°', color=slider_color
        )
        self.girl_rotation_slider.on_changed(self.update_girl_rotation)
        
        # Inicialmente ocultar todos los controles excepto Circle + Square
        self.hide_all_controls()
        self.show_circle_square_controls()
        
    def hide_all_controls(self):
        controls = [
            self.circle_diameter_ax, self.square_width_ax,
            self.ellipse_diameter_ax, self.ellipse_eccentricity_ax,
            self.rect_width_ax, self.rect_height_ax,
            self.disorder_diameter_ax, self.disorder_spacing_ax,
            self.girl_height_ax, self.girl_rotation_ax
        ]
        for control in controls:
            control.set_visible(False)
            
    def show_circle_square_controls(self):
        self.circle_diameter_ax.set_visible(True)
        self.square_width_ax.set_visible(True)
        
    def show_ellipse_controls(self):
        self.ellipse_diameter_ax.set_visible(True)
        self.ellipse_eccentricity_ax.set_visible(True)
        
    def show_rectangle_controls(self):
        self.rect_width_ax.set_visible(True)
        self.rect_height_ax.set_visible(True)
        
    def show_disorder_controls(self):
        self.disorder_diameter_ax.set_visible(True)
        self.disorder_spacing_ax.set_visible(True)
        
    def show_girl_controls(self):
        self.girl_height_ax.set_visible(True)
        self.girl_rotation_ax.set_visible(True)
        
    def change_scene(self, label):
        self.current_scene = label
        self.hide_all_controls()
        
        if label == 'Circle + Square':
            self.show_circle_square_controls()
        elif label == 'Ellipse':
            self.show_ellipse_controls()
        elif label == 'Rectangle':
            self.show_rectangle_controls()
        elif label == 'Disorder':
            self.show_disorder_controls()
        elif label == 'Waving Girl':
            self.show_girl_controls()
            
        self.update_pattern()
        
    def draw_aperture_shapes(self):
        """Dibujar formas geométricas en lugar de matrices para mejor visualización"""
        self.ax1.set_xlim(-4, 4)
        self.ax1.set_ylim(-4, 4)
        self.ax1.set_aspect('equal')
        self.ax1.set_facecolor('#0a0a0a')
        
        if self.current_scene == 'Circle + Square':
            # Calcular la separación para que no se superpongan
            separation = (self.circle_diameter / 2 + self.square_width / 2) * 600

            # Dibujar círculo
            circle = Circle((-separation, 0), self.circle_diameter * 500,
                          facecolor='cyan', alpha=0.7, edgecolor='black', linewidth=2)
            self.ax1.add_patch(circle)
            
            # Dibujar cuadrado semitransparente
            square = Rectangle((separation - self.square_width * 500, -self.square_width * 500),
                             self.square_width * 1000, self.square_width * 1000,
                             facecolor='yellow', alpha=0.5, edgecolor='black', linewidth=2)
            self.ax1.add_patch(square)
            
        elif self.current_scene == 'Ellipse':
            # Calcular semi-ejes
            a = self.ellipse_diameter * 500
            b = a * np.sqrt(1 - self.ellipse_eccentricity**2)
            
            ellipse = Ellipse((0, 0), 2*a, 2*b,
                            facecolor='magenta', alpha=0.7, edgecolor='white', linewidth=2)
            self.ax1.add_patch(ellipse)
            
        elif self.current_scene == 'Rectangle':
            rect = Rectangle((-self.rect_width * 500, -self.rect_height * 500), 
                           self.rect_width * 1000, self.rect_height * 1000,
                           facecolor='orange', alpha=0.7, edgecolor='white', linewidth=2)
            self.ax1.add_patch(rect)
            
        elif self.current_scene == 'Disorder':
            # Dibujar rejilla de círculos (4x4)
            spacing = 1.5
            radius = self.disorder_circle_diameter * 500
            
            for i in range(-1, 2):
                for j in range(-1, 2):
                    x = i * spacing
                    y = j * spacing
                    
                    # Añadir desorden
                    if self.disorder_amount > 0:
                        disorder_range = (self.disorder_amount / 4) * 0.5
                        x += np.random.uniform(-disorder_range, disorder_range)
                        y += np.random.uniform(-disorder_range, disorder_range)
                    
                    circle = Circle((x, y), radius, 
                                  facecolor='lime', alpha=0.7, edgecolor='white', linewidth=1)
                    self.ax1.add_patch(circle)
                    
        elif self.current_scene == 'Waving Girl':
            # Figura simple de persona
            scale = self.girl_height * 400
            
            # Cabeza
            head = Circle((0, scale * 0.3), scale * 0.1, 
                        facecolor='pink', alpha=0.7, edgecolor='white', linewidth=2)
            
            # Cuerpo
            body = Rectangle((-scale * 0.15, -scale * 0.3), 
                           scale * 0.3, scale * 0.6,
                           facecolor='pink', alpha=0.7, edgecolor='white', linewidth=2)
            
            # Aplicar rotación
            t = mpatches.transforms.Affine2D().rotate_deg(self.girl_rotation) + self.ax1.transData
            head.set_transform(t)
            body.set_transform(t)
            
            self.ax1.add_patch(head)
            self.ax1.add_patch(body)
        
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlabel('Posición (mm)')
        self.ax1.set_ylabel('Posición (mm)')
        self.ax1.set_title(f'Abertura: {self.current_scene}', fontsize=12)
        
    def create_aperture_matrix(self):
        """Crear matriz de abertura para cálculos FFT"""
        if self.current_scene == 'Circle + Square':
            return self.create_circle_square_aperture()
        elif self.current_scene == 'Ellipse':
            return self.create_ellipse_aperture()
        elif self.current_scene == 'Rectangle':
            return self.create_rectangle_aperture()
        elif self.current_scene == 'Disorder':
            return self.create_disorder_aperture()
        elif self.current_scene == 'Waving Girl':
            return self.create_waving_girl_aperture()
        
    def create_circle_square_aperture(self):
        aperture = np.zeros((self.matrix_size, self.matrix_size))
        center = self.matrix_size // 2
        
        # Calcular la separación en píxeles
        separation_pixels = int((self.circle_diameter / 2 + self.square_width / 2) / self.pixel_size * 1.2)

        # Crear círculo
        y, x = np.ogrid[:self.matrix_size, :self.matrix_size]
        circle_radius = self.circle_diameter / (2 * self.pixel_size)
        circle_mask = (x - (center - separation_pixels))**2 + (y - center)**2 <= circle_radius**2
        
        # Crear cuadrado
        square_half_width = self.square_width / (2 * self.pixel_size)
        square_mask = (np.abs(x - (center + separation_pixels)) <= square_half_width) & (np.abs(y - center) <= square_half_width)
        
        # Combinar
        aperture[circle_mask] = 1.0
        aperture[square_mask] = 1.0
        
        return aperture
        
    def create_ellipse_aperture(self):
        aperture = np.zeros((self.matrix_size, self.matrix_size))
        center = self.matrix_size // 2
        
        y, x = np.ogrid[:self.matrix_size, :self.matrix_size]
        a = self.ellipse_diameter / (2 * self.pixel_size)
        b = a * np.sqrt(1 - self.ellipse_eccentricity**2)
        
        ellipse_mask = ((x - center)**2 / a**2) + ((y - center)**2 / b**2) <= 1
        aperture[ellipse_mask] = 1.0
        
        return aperture
        
    def create_rectangle_aperture(self):
        aperture = np.zeros((self.matrix_size, self.matrix_size))
        center = self.matrix_size // 2
        
        y, x = np.ogrid[:self.matrix_size, :self.matrix_size]
        half_width = self.rect_width / (2 * self.pixel_size)
        half_height = self.rect_height / (2 * self.pixel_size)
        
        rect_mask = (np.abs(x - center) <= half_width) & (np.abs(y - center) <= half_height)
        aperture[rect_mask] = 1.0
        
        return aperture
        
    def create_disorder_aperture(self):
        aperture = np.zeros((self.matrix_size, self.matrix_size))
        center = self.matrix_size // 2
        
        # Crear rejilla 3x3 (reducida)
        positions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        spacing_pixels = self.disorder_spacing / self.pixel_size
        circle_radius = self.disorder_circle_diameter / (2 * self.pixel_size)
        
        for i, j in positions:
            x_pos = center + i * spacing_pixels
            y_pos = center + j * spacing_pixels
            
            # Añadir desorden
            if self.disorder_amount > 0:
                disorder_range = (self.disorder_amount / 4) * spacing_pixels * 0.3
                x_pos += np.random.uniform(-disorder_range, disorder_range)
                y_pos += np.random.uniform(-disorder_range, disorder_range)
            
            y, x = np.ogrid[:self.matrix_size, :self.matrix_size]
            circle_mask = (x - x_pos)**2 + (y - y_pos)**2 <= circle_radius**2
            aperture[circle_mask] = 1.0
                
        return aperture
        
    def create_waving_girl_aperture(self):
        aperture = np.zeros((self.matrix_size, self.matrix_size))
        center = self.matrix_size // 2
        
        height_pixels = self.girl_height / self.pixel_size
        head_radius = height_pixels * 0.1
        body_width = height_pixels * 0.3
        body_height = height_pixels * 0.6
        
        theta = np.radians(self.girl_rotation)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        y, x = np.ogrid[:self.matrix_size, :self.matrix_size]
        
        x_rot = (x - center) * cos_theta - (y - center) * sin_theta
        y_rot = (x - center) * sin_theta + (y - center) * cos_theta
        
        head_mask = x_rot**2 + (y_rot + body_height/2)**2 <= head_radius**2
        body_mask = (np.abs(x_rot) <= body_width/2) & (np.abs(y_rot) <= body_height/2)
        
        aperture[head_mask | body_mask] = 1.0
        
        return aperture
        
    def calculate_diffraction_pattern(self, aperture):
        # Aplicar transformada de Fourier 2D
        fft_result = fft2(aperture)
        fft_shifted = fftshift(fft_result)
        
        # Calcular intensidad
        intensity = np.abs(fft_shifted)**2
        
        # Normalizar
        intensity = intensity / np.max(intensity)
        
        return intensity
        
    def update_pattern(self, val=None):
        # Limpiar los ejes antes de redibujar
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Actualizar mapa de colores basado en longitud de onda
        self.setup_colormap()
        
        # Dibujar abertura como formas geométricas
        self.draw_aperture_shapes()
        
        # Crear matriz para cálculos
        aperture = self.create_aperture_matrix()
        
        # Calcular patrón de difracción
        intensity = self.calculate_diffraction_pattern(aperture)
        
        # Mostrar patrón de difracción
        self.ax2.set_facecolor('#0a0a0a')
        
        # Usar escala logarítmica para mejor visualización
        log_intensity = np.log(intensity + 1e-10)
        
        im = self.ax2.imshow(log_intensity, cmap=self.custom_cmap, 
                           extent=[-30, 30, -30, 30], interpolation='bilinear')
        
        self.ax2.set_title(f'Patrón de Difracción (λ = {self.wavelength*1e9:.0f} nm)', 
                          fontsize=12)
        self.ax2.set_xlabel('Ángulo (mrad)')
        self.ax2.set_ylabel('Ángulo (mrad)')
        self.ax2.grid(True, alpha=0.3)
        
        # Perfil 1D central
        center_row = intensity[intensity.shape[0]//2, :]
        x_axis = np.linspace(-30, 30, len(center_row))
        
        wavelength_nm = self.wavelength * 1e9
        if wavelength_nm < 500:
            color = 'blue'
        elif wavelength_nm < 600:
            color = 'green'
        else:
            color = 'red'
            
        self.ax3.plot(x_axis, center_row, color=color, linewidth=2)
        self.ax3.set_title('Perfil Central', fontsize=12)
        self.ax3.set_xlabel('Ángulo (mrad)')
        self.ax3.set_ylabel('Intensidad')
        self.ax3.grid(True, alpha=0.3)
        
        # Colorbar
        if self.cbar:
            self.cbar.remove()
        self.cbar = plt.colorbar(im, ax=self.ax2, fraction=0.046, pad=0.04)
        self.cbar.set_label('Intensidad (log)')
        
        self.fig.canvas.draw_idle()
        
    # Funciones de actualización para los sliders
    def update_wavelength(self, val):
        self.wavelength = val * 1e-9
        self.update_pattern()
        
    def update_circle_diameter(self, val):
        self.circle_diameter = val * 1e-3
        self.update_pattern()
        
    def update_square_width(self, val):
        self.square_width = val * 1e-3
        self.update_pattern()
        
    def update_ellipse_diameter(self, val):
        self.ellipse_diameter = val * 1e-3
        self.update_pattern()
        
    def update_ellipse_eccentricity(self, val):
        self.ellipse_eccentricity = val
        self.update_pattern()
        
    def update_rect_width(self, val):
        self.rect_width = val * 1e-3
        self.update_pattern()
        
    def update_rect_height(self, val):
        self.rect_height = val * 1e-3
        self.update_pattern()
        
    def update_disorder_diameter(self, val):
        self.disorder_circle_diameter = val * 1e-3
        self.update_pattern()
        
    def update_disorder_spacing(self, val):
        self.disorder_spacing = val * 1e-3
        self.update_pattern()
        
    def update_girl_height(self, val):
        self.girl_height = val * 1e-3
        self.update_pattern()
        
    def update_girl_rotation(self, val):
        self.girl_rotation = val
        self.update_pattern()
        
    def reset_values(self, event):
        self.wavelength_slider.reset()
        self.circle_diameter_slider.reset()
        self.square_width_slider.reset()
        self.ellipse_diameter_slider.reset()
        self.ellipse_eccentricity_slider.reset()
        self.rect_width_slider.reset()
        self.rect_height_slider.reset()
        self.disorder_diameter_slider.reset()
        self.disorder_spacing_slider.reset()
        self.girl_height_slider.reset()
        self.girl_rotation_slider.reset()
        
    def run(self):
        plt.show()

def main():
    """Función principal para ejecutar el simulador."""
    print("🔬 Simulador de Difracción 2D Mejorado")
    print("=" * 50)
    print("✨ Características mejoradas:")
    print("   • Interfaz oscura profesional")
    print("   • Visualización geométrica de aberturas")
    print("   • Colores dinámicos basados en longitud de onda")
    print("   • Patrón de difracción 4x4 optimizado")
    print("   • Perfil 1D del patrón central")
    print("   • Controles interactivos mejorados")
    print("=" * 50)
    
    simulator = DiffractionSimulator()
    simulator.run()

if __name__ == "__main__":
    main()