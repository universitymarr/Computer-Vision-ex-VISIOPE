import numpy as np
import matplotlib.pyplot as plt


def plot_filtered_image(
    original_image: np.ndarray,
    filtered_image: np.ndarray,
    filter_name: str,
    original_name: str = "Original",
    grayscale: bool = True,
    figsize: tuple = (20, 5),
):
    """Plot an original image and its filtered version.

    Args:
        original_image (np.ndarray): original image
        filtered_image (np.ndarray): filtered image
        filter_name (str): name of filtered version of the image
        original_name (str): name of original version of the image
        grayscale (bool, optional): indicates whether the image is grayscale. Defaults to True.
        figsize (tuple, optional): figure size. Defaults to (20, 5).
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.set_title(f"{original_name}")
    ax2.set_title(f"{filter_name}")
    if grayscale:
        ax1.imshow(original_image, cmap="gray")
        ax2.imshow(filtered_image, cmap="gray")
    else:
        ax1.imshow(original_image)
        ax2.imshow(filtered_image)


def plot_histogram_equalization(cdf_normalized: np.ndarray, image: np.ndarray):
    plt.plot(cdf_normalized)
    plt.hist(image.flatten(), 256, [0, 255])
    plt.legend(("cdf", "histogram"), loc="upper left")
    plt.show()


def plot_equalized(
    original_image: np.ndarray,
    cdf_normalized: np.ndarray,
    figsize: tuple = (20, 5),
):
    """Plot the histogram of equalization of an equalized image and its original version.

    Args:
        original_image (np.ndarray): original image
        cdf_normalized (str): normalized cumulative distribution function.
        figsize (tuple, optional): figure size. Defaults to (20, 5).
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.set_title("Original")
    ax2.set_title("Equalized")

    ax1.imshow(original_image, cmap="gray")
    ax2.plot(cdf_normalized)
    ax2.hist(original_image.flatten(), 256, [0, 255])


def plot_multiple_rows(
    images_dict: dict,
    rows: int,
    cols: int,
    figsize: tuple = (20, 15),
):
    """Plot an original image and its filtered version.

    Args:
        original_image (np.ndarray): original image
        cdf_normalized (str): normalized cumulative distribution function.
        figsize (tuple, optional): figure size. Defaults to (20, 5).
    """
    f, axes = plt.subplots(rows, cols, figsize=figsize)

    for row, key in enumerate(images_dict):
        for col, img in enumerate(images_dict[key]):
            if len(axes.shape) > 1:
                axes[row, col].imshow(img)
            else:
                axes[col].imshow(img)


def plot_image_hist(image: np.ndarray, figsize: tuple = (20, 5)):
    """Plot the histogram of an image.

    Args:
        image (np.ndarray): an image.
        figsize (tuple, optional): figure size. Defaults to (20, 5).
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.set_title("Image")
    ax2.set_title("Histogram")

    ax1.imshow(image, cmap="gray")
    ax2.hist(image.flatten(), 256, [0, 255])

    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    ax2.plot(cdf_normalized)
