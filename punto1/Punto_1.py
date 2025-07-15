import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.patches as patches
from scipy.special import j1
import matplotlib.colors as mcolors

class FraunhoferDiffraction:
    def __init__(self):
        """
        Initializes the simulation parameters and sets up the plot.
        """
        # System parameters (default values)
        self.params = {
            'lambda': 550e-9,      # Wavelength (m) - default green
            'D': 2e-3,            # Distance between apertures (m)
            'a': 0.5e-3,          # Width of the rectangle (m)
            'b': 0.5e-3,          # Height of the rectangle (m)
            'R1': 0.2e-3,         # Inner radius of the annulus (m)
            'R2': 0.8e-3,         # Outer radius of the annulus (m)
            'z': 1,               # Distance to the observation plane (m)
            'n': 1,               # Refractive index
            'I_source': 1,        # Source intensity
            'range': 5e-3,        # Observation range (m)
            'aperture_range': 2e-3 # Range for displaying the original aperture
        }
        
        # Grid configuration
        self.grid_size = 400
        self.aperture_grid_size = 200
        self.setup_plot()
    
    def wavelength_to_rgb(self, wavelength_nm):
        """Converts wavelength in nm to an RGB color tuple."""
        # Based on a standard algorithm for wavelength to RGB conversion.
        wavelength = wavelength_nm
        
        if 380 <= wavelength < 440:
            r = -(wavelength - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        elif 440 <= wavelength < 490:
            r = 0.0
            g = (wavelength - 440) / (490 - 440)
            b = 1.0
        elif 490 <= wavelength < 510:
            r = 0.0
            g = 1.0
            b = -(wavelength - 510) / (510 - 490)
        elif 510 <= wavelength < 580:
            r = (wavelength - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif 580 <= wavelength < 645:
            r = 1.0
            g = -(wavelength - 645) / (645 - 580)
            b = 0.0
        elif 645 <= wavelength <= 780:
            r = 1.0
            g = 0.0
            b = 0.0
        else:
            r, g, b = 0.0, 0.0, 0.0 # Outside visible spectrum

        # Adjust intensity for extreme wavelengths to make them dimmer
        if 380 <= wavelength < 420:
            factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
        elif 645 < wavelength <= 780:
            factor = 0.3 + 0.7 * (780 - wavelength) / (780 - 645)
        else:
            factor = 1.0
        
        return (r * factor, g * factor, b * factor)
    
    def create_wavelength_colormap(self, wavelength_nm):
        """Creates a colormap from black to the specified wavelength color."""
        base_color = self.wavelength_to_rgb(wavelength_nm)
        return mcolors.LinearSegmentedColormap.from_list(
            "custom_map", [(0, 0, 0), base_color]
        )
    
    def sinc(self, x):
        """Numerically stable sinc function."""
        return np.where(np.abs(x) < 1e-10, 1, np.sin(x) / x)
    
    def bessel_j1_normalized(self, x):
        """Normalized Bessel function of the first kind: J1(x)/x. Reverted to original."""
        return np.where(np.abs(x) < 1e-10, 0.5, j1(x) / x)
    
    def calculate_intensity(self, x, y):
        """
        Calculates the intensity based on the original Fraunhofer diffraction equation provided by the user.
        """
        # Parameters
        lambda_val = self.params['lambda']
        D = self.params['D']
        a = self.params['a']
        b = self.params['b']
        R1 = self.params['R1']
        R2 = self.params['R2']
        z = self.params['z']
        n = self.params['n']
        I_source = self.params['I_source']
        
        # Wavenumber
        k = 2 * np.pi / lambda_val
        
        # Radial coordinate
        r = np.sqrt(x**2 + y**2)
        
        # Arguments for sinc functions
        beta_x = k * a * x / z
        beta_y = k * b * y / z
        
        # Rectangle term
        sinc_x = self.sinc(beta_x / 2)
        sinc_y = self.sinc(beta_y / 2)
        
        # Annulus (ring) terms
        kr = k * r
        kr_safe = np.where(r > 1e-10, kr, 1e-10) # Avoid division by zero
        
        bessel_R1 = np.where(R1 > 0, R1**2 * self.bessel_j1_normalized(kr_safe * R1 / z), 0)
        bessel_R2 = np.where(R2 > 0, R2**2 * self.bessel_j1_normalized(kr_safe * R2 / z), 0)
        
        # Full intensity equation from original code
        # Term 1: rectangle^2
        term1 = (b**2 * a**2) * sinc_x**2 * sinc_y**2
        
        # Term 2: annulus^2
        term2 = (4 * np.pi)**2 * (bessel_R2 - bessel_R1)**2
        
        # Term 3: interference
        phase = k * D * y / z
        rect_field_term = (b * a) * sinc_x * sinc_y
        ring_field_term = (4 * np.pi) * (bessel_R2 - bessel_R1)
        term3 = 2 * rect_field_term * ring_field_term * np.cos(phase)
        
        # Total intensity
        intensity = I_source * (n**2 / (lambda_val**2 * z**2)) * (term1 + term2 + term3)
        
        return np.maximum(0, intensity)

    def calculate_pattern(self):
        """Calculates the 2D intensity pattern."""
        range_val = self.params['range']
        x = np.linspace(-range_val, range_val, self.grid_size)
        y = np.linspace(-range_val, range_val, self.grid_size)
        X, Y = np.meshgrid(x, y)
        intensity = self.calculate_intensity(X, Y)
        return X, Y, intensity
    
    def create_aperture_pattern(self):
        """Creates the pattern of the original aperture."""
        range_val = self.params['aperture_range']
        x = np.linspace(-range_val, range_val, self.aperture_grid_size)
        y = np.linspace(-range_val, range_val, self.aperture_grid_size)
        X, Y = np.meshgrid(x, y)
        
        aperture = np.zeros_like(X)
        
        # Rectangle centered at y = D/2
        rect_mask = (np.abs(X) <= self.params['a'] / 2) & (np.abs(Y - self.params['D'] / 2) <= self.params['b'] / 2)
        aperture[rect_mask] = 1
        
        # Annulus (ring) centered at y = -D/2
        r_aperture = np.sqrt(X**2 + (Y + self.params['D'] / 2)**2)
        ring_mask = (r_aperture >= self.params['R1']) & (r_aperture <= self.params['R2'])
        aperture[ring_mask] = 1
        
        return X, Y, aperture
    
    def setup_plot(self):
        """Configures the graphical interface using a robust GridSpec layout."""
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor('#f0f0f0')
        self.fig.suptitle("Simulador de Difracción de Fraunhofer", fontsize=16, fontweight='bold')

        gs_main = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3, left=0.05, right=0.95, top=0.92, bottom=0.08)
        self.ax_main = self.fig.add_subplot(gs_main[0, 1])
        gs_left = gs_main[0, 0].subgridspec(13, 1, hspace=2.5)
        self.ax_aperture = self.fig.add_subplot(gs_left[0:4, 0])

        self.sliders = {}
        slider_defs = {
            'lambda': {'label': 'λ (nm)', 'range': (400, 700), 'format': '%.0f', 'ax': self.fig.add_subplot(gs_left[5])},
            'D': {'label': 'D (mm)', 'range': (0.5, 5), 'format': '%.2f', 'ax': self.fig.add_subplot(gs_left[6])},
            'a': {'label': 'a (mm)', 'range': (0.1, 2), 'format': '%.2f', 'ax': self.fig.add_subplot(gs_left[7])},
            'b': {'label': 'b (mm)', 'range': (0.1, 2), 'format': '%.2f', 'ax': self.fig.add_subplot(gs_left[8])},
            'R1': {'label': 'R₁ (mm)', 'range': (0, 1), 'format': '%.2f', 'ax': self.fig.add_subplot(gs_left[9])},
            'R2': {'label': 'R₂ (mm)', 'range': (0.1, 2), 'format': '%.2f', 'ax': self.fig.add_subplot(gs_left[10])},
            'z': {'label': 'z (m)', 'range': (0.5, 3), 'format': '%.1f', 'ax': self.fig.add_subplot(gs_left[11])},
            'range': {'label': 'Rango (mm)', 'range': (2, 20), 'format': '%.1f', 'ax': self.fig.add_subplot(gs_left[12])}
        }
        
        conversions = {'lambda': 1e-9, 'D': 1e-3, 'a': 1e-3, 'b': 1e-3, 'R1': 1e-3, 'R2': 1e-3, 'z': 1, 'range': 1e-3}

        for key, sdef in slider_defs.items():
            val_init = self.params[key] / conversions[key]
            self.sliders[key] = Slider(sdef['ax'], sdef['label'], sdef['range'][0], sdef['range'][1], valinit=val_init, valfmt=sdef['format'])
        
        for slider in self.sliders.values():
            slider.on_changed(self.update_params)
        
        self.update_plot()
    
    def update_params(self, val):
        """Updates parameters when sliders change and triggers a plot update."""
        conversions = {'lambda': 1e-9, 'D': 1e-3, 'a': 1e-3, 'b': 1e-3, 'R1': 1e-3, 'R2': 1e-3, 'z': 1, 'range': 1e-3}
        for key, slider in self.sliders.items():
            self.params[key] = slider.val * conversions[key]
        
        if self.params['R1'] > self.params['R2']:
            self.params['R1'] = self.params['R2']
            self.sliders['R1'].set_val(self.params['R1'] / conversions['R1'])

        self.update_plot()
    
    def update_plot(self):
        """Updates both plots with the new parameters."""
        self.ax_main.clear()
        self.ax_aperture.clear()
        
        X_ap, Y_ap, aperture = self.create_aperture_pattern()
        range_mm_ap = self.params['aperture_range'] * 1e3
        self.ax_aperture.imshow(aperture, extent=[-range_mm_ap, range_mm_ap, -range_mm_ap, range_mm_ap],
                                cmap='gray', origin='lower')
        self.ax_aperture.set_title('Apertura Original', fontsize=12, fontweight='bold')
        self.ax_aperture.set_xlabel('x (mm)')
        self.ax_aperture.set_ylabel('y (mm)')
        self.ax_aperture.grid(True, linestyle='--', alpha=0.4)
        self.ax_aperture.set_aspect('equal', 'box')

        X, Y, intensity = self.calculate_pattern()
        
        max_intensity = np.max(intensity)
        if max_intensity > 0:
            intensity_log = np.log1p(intensity / max_intensity * 1000) 
            intensity_norm = intensity_log / np.max(intensity_log)
        else:
            intensity_norm = intensity

        wavelength_nm = self.params['lambda'] * 1e9
        custom_cmap = self.create_wavelength_colormap(wavelength_nm)
        
        range_mm = self.params['range'] * 1e3
        self.ax_main.imshow(intensity_norm, 
                            extent=[-range_mm, range_mm, -range_mm, range_mm],
                            cmap=custom_cmap, origin='lower', interpolation='bilinear')
        
        self.ax_main.set_title(f'Patrón de Difracción de Fraunhofer', fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel('x (mm)', fontsize=12)
        self.ax_main.set_ylabel('y (mm)', fontsize=12)
        
        wavelength_color = self.wavelength_to_rgb(wavelength_nm)
        info_text = f'λ = {wavelength_nm:.0f} nm'
        self.ax_main.text(0.98, 0.98, info_text, transform=self.ax_main.transAxes,
                          fontsize=12, fontweight='bold', color='white', ha='right', va='top',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor=wavelength_color, alpha=0.8))
        
        params_text = (f"D = {self.params['D']*1e3:.2f} mm\n"
                       f"a = {self.params['a']*1e3:.2f} mm, b = {self.params['b']*1e3:.2f} mm\n"
                       f"R₁ = {self.params['R1']*1e3:.2f} mm, R₂ = {self.params['R2']*1e3:.2f} mm\n"
                       f"z = {self.params['z']:.1f} m")
        self.ax_main.text(0.02, 0.02, params_text, transform=self.ax_main.transAxes,
                          fontsize=10, color='white', verticalalignment='bottom',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
        
        self.fig.canvas.draw_idle()
    
    def run(self):
        """Executes the simulation and shows the plot."""
        plt.show()

def main():
    print("=" * 60)
    print("      SIMULADOR DE DIFRACCIÓN DE FRAUNHOFER MEJORADO")
    print("=" * 60)
    print("\n✅ Cálculo de interferencia restaurado a la versión original.")
    print("✅ Interfaz gráfica mejorada conservada.")
    print("\n🚀 ¡Disfruta explorando la difracción de Fraunhofer!")
    print("=" * 60)
    
    simulator = FraunhoferDiffraction()
    simulator.run()

if __name__ == "__main__":
    main()