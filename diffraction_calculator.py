import numpy as np

def create_circular_aperture(shape, radius, center_x, center_y):
    """
    Creates a 2D circular aperture.

    Args:
        shape (tuple): (rows, cols) of the aperture matrix.
        radius (float): Radius of the circle.
        center_x (float): X-coordinate of the circle's center.
        center_y (float): Y-coordinate of the circle's center.

    Returns:
        np.ndarray: 2D array representing the circular aperture (1s for open, 0s for opaque).
    """
    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    aperture = np.zeros(shape)
    aperture[mask] = 1
    return aperture

def create_square_aperture(shape, side_length, center_x, center_y):
    """
    Creates a 2D square aperture.

    Args:
        shape (tuple): (rows, cols) of the aperture matrix.
        side_length (float): Side length of the square.
        center_x (float): X-coordinate of the square's center.
        center_y (float): Y-coordinate of the square's center.

    Returns:
        np.ndarray: 2D array representing the square aperture.
    """
    rows, cols = shape
    aperture = np.zeros(shape)
    half_side = side_length / 2
    x_start = int(center_x - half_side)
    x_end = int(center_x + half_side)
    y_start = int(center_y - half_side)
    y_end = int(center_y + half_side)

    # Ensure indices are within bounds
    x_start = max(0, x_start)
    x_end = min(cols, x_end)
    y_start = max(0, y_start)
    y_end = min(rows, y_end)

    aperture[y_start:y_end, x_start:x_end] = 1
    return aperture

def create_single_slit_aperture(shape, slit_width, slit_height, center_x, center_y):
    """
    Creates a 2D single slit aperture.

    Args:
        shape (tuple): (rows, cols) of the aperture matrix.
        slit_width (float): Width of the slit.
        slit_height (float): Height of the slit.
        center_x (float): X-coordinate of the slit's center.
        center_y (float): Y-coordinate of the slit's center.

    Returns:
        np.ndarray: 2D array representing the single slit.
    """
    rows, cols = shape
    aperture = np.zeros(shape)
    half_width = slit_width / 2
    half_height = slit_height / 2

    x_start = int(center_x - half_width)
    x_end = int(center_x + half_width)
    y_start = int(center_y - half_height)
    y_end = int(center_y + half_height)

    x_start = max(0, x_start)
    x_end = min(cols, x_end)
    y_start = max(0, y_start)
    y_end = min(rows, y_end)

    aperture[y_start:y_end, x_start:x_end] = 1
    return aperture

def create_double_slit_aperture(shape, slit_width, slit_height, separation, center_x, center_y):
    """
    Creates a 2D double slit aperture.

    Args:
        shape (tuple): (rows, cols) of the aperture matrix.
        slit_width (float): Width of each slit.
        slit_height (float): Height of each slit.
        separation (float): Separation between the centers of the two slits.
        center_x (float): X-coordinate of the midpoint between the two slits.
        center_y (float): Y-coordinate of the center of the slits.

    Returns:
        np.ndarray: 2D array representing the double slit.
    """
    rows, cols = shape
    aperture = np.zeros(shape)

    # Slit 1
    center_x1 = center_x - separation / 2
    aperture += create_single_slit_aperture(shape, slit_width, slit_height, center_x1, center_y)

    # Slit 2
    center_x2 = center_x + separation / 2
    aperture += create_single_slit_aperture(shape, slit_width, slit_height, center_x2, center_y)

    # Ensure aperture values are binary (0 or 1) in case of overlap (though unlikely with typical params)
    aperture = np.clip(aperture, 0, 1)
    return aperture


def calculate_diffraction_pattern(aperture_matrix, log_scale=True):
    """
    Calculates the Fraunhofer diffraction pattern using FFT.

    Args:
        aperture_matrix (np.ndarray): 2D array representing the aperture.
        log_scale (bool): If True, applies a log1p scale to the intensity for better visualization.

    Returns:
        np.ndarray: 2D array of the diffraction pattern's intensity.
    """
    # Perform 2D FFT
    ft = np.fft.fft2(aperture_matrix)

    # Shift the zero-frequency component to the center
    ft_shifted = np.fft.fftshift(ft)

    # Calculate the intensity (magnitude squared)
    intensity = np.abs(ft_shifted)**2

    if log_scale:
        # Apply log scale for better visualization (handles zero intensity)
        # Adding a small epsilon to avoid log(0) if any intensity is exactly 0,
        # though np.log1p(x) = log(1+x) is generally better.
        intensity = np.log1p(intensity)

    return intensity

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    import matplotlib.pyplot as plt

    img_size = 512 # Resolution of the aperture/pattern

    # --- Circular Aperture ---
    ap_shape = (img_size, img_size)
    radius = img_size / 16
    center = img_size / 2
    circ_aperture = create_circular_aperture(ap_shape, radius, center, center)
    circ_diffraction = calculate_diffraction_pattern(circ_aperture)

    # --- Square Aperture ---
    square_side = img_size / 8
    sq_aperture = create_square_aperture(ap_shape, square_side, center, center)
    sq_diffraction = calculate_diffraction_pattern(sq_aperture)

    # --- Single Slit ---
    slit_w = img_size / 32
    slit_h = img_size / 4
    single_slit_aperture = create_single_slit_aperture(ap_shape, slit_w, slit_h, center, center)
    single_slit_diffraction = calculate_diffraction_pattern(single_slit_aperture)

    # --- Double Slit ---
    # Slit params are for each individual slit
    dslit_w = img_size / 64
    dslit_h = img_size / 4
    dslit_sep = img_size / 16 # Separation between centers of slits
    double_slit_aperture = create_double_slit_aperture(ap_shape, dslit_w, dslit_h, dslit_sep, center, center)
    double_slit_diffraction = calculate_diffraction_pattern(double_slit_aperture)

    # Plotting
    fig, axes = plt.subplots(4, 2, figsize=(8, 16))
    fig.suptitle("Apertures and their Diffraction Patterns", fontsize=16)

    apertures = [circ_aperture, sq_aperture, single_slit_aperture, double_slit_aperture]
    patterns = [circ_diffraction, sq_diffraction, single_slit_diffraction, double_slit_diffraction]
    titles = ["Circular", "Square", "Single Slit", "Double Slit"]

    for i in range(4):
        # Plot Aperture
        axes[i, 0].imshow(apertures[i], cmap='gray', origin='lower')
        axes[i, 0].set_title(f"{titles[i]} Aperture")
        axes[i, 0].axis('off')

        # Plot Diffraction Pattern
        # Using a common scaling for diffraction patterns might be better, or individual scaling.
        # For now, individual scaling based on each pattern's min/max.
        img = axes[i, 1].imshow(patterns[i], cmap='hot', origin='lower')
        axes[i, 1].set_title(f"{titles[i]} Diffraction")
        axes[i, 1].axis('off')
        fig.colorbar(img, ax=axes[i,1], fraction=0.046, pad=0.04)


    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

    print("Diffraction calculation module created and test plots generated.")
