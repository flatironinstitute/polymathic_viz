import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cv2
import imageio

class ChessboardPlotter:

    def __init__(self, name, data, row_size=4, color_map='viridis', plotWidth=1000, plotHeight=500):
        self.name = name
        self.data = data
        self.row_size = row_size
        (self.conditions, self.timestamps, self.width, self.height) = data.shape
        # global color scale
        vmin = data.min()
        vmax = data.max()
        self.norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        self.cmap = plt.get_cmap(color_map)
        self.fig = None
        self.plotWidth = plotWidth
        self.plotHeight = plotHeight

    def plot(self, timestamp):
        arrays = self.data[:, timestamp, :, :]
        rows = len(arrays) // self.row_size + 1
        columns = min(self.row_size, len(arrays))
        # xxx need to adjust figsize per inputs
        self.fig, axs = plt.subplots(rows, columns, figsize=(10,5), dpi=100)
        test = False
        while not test:
            try:
                # hack to handle single row case
                test = axs[0, 0]
            except:
                axs = np.array([axs])
        plt.suptitle(repr(self.name) + " Timestamp: " + str(timestamp))
        for (num, arr) in enumerate(arrays):
            ax = axs[num // self.row_size, num % self.row_size]
            ax.imshow(arr, cmap=self.cmap, norm=self.norm)
            #ax.axis('off')
            ax.set_title("Condition: " + str(num))
        # turn off the axes for the remaining empty subplots
        for axrow in axs:
            for ax in axrow:
                ax.axis('off')
        plt.tight_layout()

    def plot_image(self, timestamp):
        self.plot(timestamp)
        fig = self.fig
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image = np.roll(image, 3, axis=2)  # Convert ARGB to RGBA
        # Close the figure to free resources
        plt.close(fig)
        image_resized = cv2.resize(image, (self.plotWidth, self.plotHeight), interpolation=cv2.INTER_AREA)
        return image_resized
    
    def plot_images(self):
        images = []
        for timestamp in range(self.timestamps):
            images.append(self.plot_image(timestamp))
        self.images = np.array(images)
        return self.images
    
    def plot_gif(self, filename, fps=None):
        # If fps is not provided, set it to finish the animation in 0.5 minute
        if fps is None:
            fps = self.timestamps / 30
        print("Creating gif with fps: ", fps, "to file: ", filename, "for timestamps: ", self.timestamps, "timepoints")
        images = self.plot_images()
        imageio.mimsave(filename, images, fps=fps, loop=0)
        self.images = images

def max_value_chessboard(name, data3d, row_size=4, color_map='viridis', plotWidth=1000, plotHeight=500):
    data2d = np.max(data3d, axis=2)
    plotter = ChessboardPlotter(name, data2d, row_size=row_size, color_map=color_map, plotWidth=plotWidth, plotHeight=plotHeight)
    return plotter
    