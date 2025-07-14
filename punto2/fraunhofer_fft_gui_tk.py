import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import a funciones de cálculo del otro archivo
from diffraction_calculator import (
    create_circular_aperture,
    create_square_aperture,
    create_single_slit_aperture,
    create_double_slit_aperture,
    calculate_diffraction_pattern
)

class DiffractionApp:
    def __init__(self, master):
        self.master = master
        master.title("Simulador de Difracción de Fraunhofer")

        # --- Parámetros de simulación ---
        self.sim_resolution = tk.IntVar(value=256) # Matriz de N x N

        # --- Variables de control ---
        self.aperture_type = tk.StringVar(value="Círculo")
        # Círculo
        self.circ_radius = tk.DoubleVar(value=20)
        # Cuadrado
        self.sq_side = tk.DoubleVar(value=40)
        # Rendija Simple
        self.slit_width = tk.DoubleVar(value=10)
        self.slit_height = tk.DoubleVar(value=80)
        # Doble Rendija
        self.dslit_width = tk.DoubleVar(value=8)
        self.dslit_height = tk.DoubleVar(value=80)
        self.dslit_separation = tk.DoubleVar(value=30)

        # Lambda (longitud de onda) - por ahora no afecta directamente el cálculo de FFT,
        # pero es un parámetro físico importante. Podría usarse para escalar ejes en el futuro.
        self.lambda_wave = tk.DoubleVar(value=500) # en nm

        # --- Layout ---
        # Frame de Controles
        control_frame = ttk.LabelFrame(master, text="Controles")
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        # Frame de Visualización
        self.plot_frame = ttk.LabelFrame(master, text="Visualización")
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        master.grid_columnconfigure(1, weight=1) # Permitir que el plot_frame se expanda
        master.grid_rowconfigure(0, weight=1)

        # --- Controles ---
        row_idx = 0
        ttk.Label(control_frame, text="Tipo de Abertura:").grid(row=row_idx, column=0, columnspan=2, sticky="w", pady=2)
        aperture_options = ["Círculo", "Cuadrado", "Rendija Simple", "Doble Rendija"]
        aperture_menu = ttk.OptionMenu(control_frame, self.aperture_type, self.aperture_type.get(), *aperture_options, command=self.update_parameter_ui)
        aperture_menu.grid(row=row_idx+1, column=0, columnspan=2, sticky="ew", pady=(0,10))
        row_idx += 2

        # Frame para parámetros específicos de la abertura (se rellenará dinámicamente)
        self.param_frame = ttk.Frame(control_frame)
        self.param_frame.grid(row=row_idx, column=0, columnspan=2, sticky="ew")
        row_idx += 1

        # Separador
        ttk.Separator(control_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=2, sticky='ew', pady=10)
        row_idx += 1

        # Parámetros generales de simulación
        ttk.Label(control_frame, text="Parámetros de Simulación:").grid(row=row_idx, column=0, columnspan=2, sticky="w", pady=2)
        row_idx += 1

        ttk.Label(control_frame, text="Resolución (N x N):").grid(row=row_idx, column=0, sticky="w")
        ttk.Entry(control_frame, textvariable=self.sim_resolution, width=7).grid(row=row_idx, column=1, sticky="e", pady=2)
        row_idx += 1

        ttk.Label(control_frame, text="Longitud de Onda (nm):").grid(row=row_idx, column=0, sticky="w")
        ttk.Entry(control_frame, textvariable=self.lambda_wave, width=7).grid(row=row_idx, column=1, sticky="e", pady=2)
        row_idx += 1

        # Botón de cálculo
        calculate_button = ttk.Button(control_frame, text="Calcular y Mostrar Patrón", command=self.run_simulation)
        calculate_button.grid(row=row_idx, column=0, columnspan=2, pady=20)
        row_idx += 1

        # --- Setup inicial de UI de parámetros y plots ---
        self.update_parameter_ui() # Llama para construir los widgets de parámetros iniciales
        self.setup_plots()
        self.run_simulation() # Mostrar un resultado inicial

    def update_parameter_ui(self, event=None):
        # Limpiar frame de parámetros anteriores
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        atype = self.aperture_type.get()
        p_row = 0
        ttk.Label(self.param_frame, text=f"Parámetros para: {atype}").grid(row=p_row, column=0, columnspan=2, sticky="w", pady=(0,5))
        p_row+=1

        if atype == "Círculo":
            ttk.Label(self.param_frame, text="Radio (px):").grid(row=p_row, column=0, sticky="w")
            ttk.Entry(self.param_frame, textvariable=self.circ_radius, width=7).grid(row=p_row, column=1, sticky="e", pady=2)
        elif atype == "Cuadrado":
            ttk.Label(self.param_frame, text="Lado (px):").grid(row=p_row, column=0, sticky="w")
            ttk.Entry(self.param_frame, textvariable=self.sq_side, width=7).grid(row=p_row, column=1, sticky="e", pady=2)
        elif atype == "Rendija Simple":
            ttk.Label(self.param_frame, text="Ancho (px):").grid(row=p_row, column=0, sticky="w")
            ttk.Entry(self.param_frame, textvariable=self.slit_width, width=7).grid(row=p_row, column=1, sticky="e", pady=2)
            p_row+=1
            ttk.Label(self.param_frame, text="Alto (px):").grid(row=p_row, column=0, sticky="w")
            ttk.Entry(self.param_frame, textvariable=self.slit_height, width=7).grid(row=p_row, column=1, sticky="e", pady=2)
        elif atype == "Doble Rendija":
            ttk.Label(self.param_frame, text="Ancho Rendija (px):").grid(row=p_row, column=0, sticky="w")
            ttk.Entry(self.param_frame, textvariable=self.dslit_width, width=7).grid(row=p_row, column=1, sticky="e", pady=2)
            p_row+=1
            ttk.Label(self.param_frame, text="Alto Rendija (px):").grid(row=p_row, column=0, sticky="w")
            ttk.Entry(self.param_frame, textvariable=self.dslit_height, width=7).grid(row=p_row, column=1, sticky="e", pady=2)
            p_row+=1
            ttk.Label(self.param_frame, text="Separación (px):").grid(row=p_row, column=0, sticky="w")
            ttk.Entry(self.param_frame, textvariable=self.dslit_separation, width=7).grid(row=p_row, column=1, sticky="e", pady=2)
        # p_row se incrementa dentro de los if para el siguiente widget

    def setup_plots(self):
        self.fig, (self.ax_aperture, self.ax_diffraction) = plt.subplots(1, 2, figsize=(8, 4))
        self.fig.tight_layout(pad=3.0)

        self.ax_aperture.set_title("Abertura")
        self.ax_aperture.axis('off')

        self.ax_diffraction.set_title("Patrón de Difracción (log)")
        self.ax_diffraction.axis('off')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def run_simulation(self):
        try:
            resolution = self.sim_resolution.get()
            if resolution <= 0:
                messagebox.showerror("Error", "La resolución debe ser un entero positivo.")
                return
            shape = (resolution, resolution)
            center_x, center_y = resolution / 2, resolution / 2

            aperture_matrix = np.zeros(shape)
            atype = self.aperture_type.get()

            if atype == "Círculo":
                radius = self.circ_radius.get()
                if radius <=0: raise ValueError("El radio debe ser positivo.")
                aperture_matrix = create_circular_aperture(shape, radius, center_x, center_y)
            elif atype == "Cuadrado":
                side = self.sq_side.get()
                if side <=0: raise ValueError("El lado debe ser positivo.")
                aperture_matrix = create_square_aperture(shape, side, center_x, center_y)
            elif atype == "Rendija Simple":
                s_width = self.slit_width.get()
                s_height = self.slit_height.get()
                if s_width <=0 or s_height <=0: raise ValueError("Ancho y alto de rendija deben ser positivos.")
                aperture_matrix = create_single_slit_aperture(shape, s_width, s_height, center_x, center_y)
            elif atype == "Doble Rendija":
                ds_width = self.dslit_width.get()
                ds_height = self.dslit_height.get()
                ds_sep = self.dslit_separation.get()
                if ds_width <=0 or ds_height <=0 or ds_sep <=0: raise ValueError("Parámetros de doble rendija deben ser positivos.")
                aperture_matrix = create_double_slit_aperture(shape, ds_width, ds_height, ds_sep, center_x, center_y)

            # Calcular patrón de difracción
            diff_pattern = calculate_diffraction_pattern(aperture_matrix, log_scale=True)

            # Actualizar plots
            self.ax_aperture.clear()
            self.ax_aperture.imshow(aperture_matrix, cmap='gray', origin='lower')
            self.ax_aperture.set_title("Abertura")
            self.ax_aperture.axis('off')

            self.ax_diffraction.clear()
            # Usar vmin y vmax puede ser útil para estabilizar el colorbar si se añade uno
            #vmin_pat, vmax_pat = np.percentile(diff_pattern, [1, 99]) # Ignorar outliers extremos para la escala
            im = self.ax_diffraction.imshow(diff_pattern, cmap='hot', origin='lower') #, vmin=vmin_pat, vmax=vmax_pat)
            self.ax_diffraction.set_title("Patrón de Difracción (log)")
            self.ax_diffraction.axis('off')

            # Podríamos añadir un colorbar, pero hay que manejarlo bien para que no se redibuje/solape
            # if hasattr(self, 'colorbar'):
            #     self.colorbar.remove()
            # self.colorbar = self.fig.colorbar(im, ax=self.ax_diffraction, fraction=0.046, pad=0.04)


            self.canvas.draw()

        except ValueError as e:
            messagebox.showerror("Error de Entrada", str(e))
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ocurrió un error: {str(e)}")


if __name__ == '__main__':
    root = tk.Tk()
    app = DiffractionApp(root)
    root.mainloop()
