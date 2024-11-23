import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

class AcneDetector:
    def __init__(self, debug=True):
        self.debug = debug
        self.red_params = {
            'top_percentage': 0.01,  
        }

    def show_debug_image(self, title, image):
        if self.debug:
            cv2.imshow(title, image)
            cv2.waitKey(0)

    def visualize_a_channel(self, image):
        """Create and display a color map for the a-channel of the LAB color space."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        _, a, _ = cv2.split(lab)
        
        # Normalize the a-channel
        a_normalized = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply a color map to the a-channel
        a_colormap = cv2.applyColorMap(a_normalized, cv2.COLORMAP_JET)
        
        # Show the color map
        self.show_debug_image("A-Channel Colormap", a_colormap)
        return a_normalized

    def find_red_regions(self, a_channel):
        """Find red regions and expand the mask using dilation."""
        threshold = np.percentile(a_channel, 100 - (self.red_params['top_percentage'] * 100))
        red_mask = (a_channel >= threshold).astype(np.uint8) * 255

        # Expand the red mask slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)  # Expands redness regions
        return red_mask

    def plot_3d_map(self, a_channel, block_size=3):
        """Plot a 3D map where x and y are pixel locations and z is the intensity."""
        # Downsample the image by averaging block_size x block_size pixels
        height, width = a_channel.shape
        downsampled_height = height // block_size
        downsampled_width = width // block_size
        
        downsampled_a = cv2.resize(a_channel, (downsampled_width, downsampled_height), interpolation=cv2.INTER_AREA)
        
        # Create a grid of x and y coordinates
        x, y = np.meshgrid(np.arange(downsampled_width), np.arange(downsampled_height))
        
        # Flatten the arrays
        x = x.flatten()
        y = y.flatten()
        z = downsampled_a.flatten()
        
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")
        scatter = axis.scatter(x, y, z, c=z, cmap='jet', marker=".")
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Intensity")
        plt.colorbar(scatter, ax=axis, label='Intensity')
        plt.show()

def select_image():
    """Open a file dialog to select an image."""
    Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
    image_path = askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not image_path:
        raise Exception("No image selected")
    return image_path

if __name__ == "__main__":
    try:
        image_path = select_image()
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to load image")
        
        detector = AcneDetector(debug=True)
        a_channel = detector.visualize_a_channel(image)
        red_mask = detector.find_red_regions(a_channel)
        detector.show_debug_image("Red Mask (Top 1% Redness)", red_mask)
        detector.plot_3d_map(a_channel, block_size=3)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred: {str(e)}")