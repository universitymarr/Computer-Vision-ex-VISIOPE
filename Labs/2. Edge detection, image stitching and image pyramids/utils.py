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


def plot_cv2_images(
    *imgs: np.ndarray,
    titles: list = None,
    figsize: tuple = (30, 20),
    hide_ticks: bool() = False
):
    """Display one or multiple images.

    Args:
        titles (list, optional): a list containing the titles of the images. Defaults to None.
        figsize (tuple, optional): Defaults to (30, 20).
        hide_ticks (bool, optional): Boolean to hide the ticks in the image. Defaults to False.
    """
    f = plt.figure(figsize=figsize)
    width = int(np.ceil(np.sqrt(len(imgs))))
    height = int(np.ceil(len(imgs) / width))
    for i, img in enumerate(imgs, 1):
        ax = f.add_subplot(height, width, i)
        if hide_ticks:
            ax.axis("off")
        if len(img.shape) > 2:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray")
        if titles != None:
            ax.set_title(titles[i - 1])


# split the data in training and test sets according to fraction
def splitdata_train_test(data, fraction_training):
    # ramdomize dataset order
    np.random.seed(0)
    np.random.shuffle(data)

    split = int(data.shape[0] * fraction_training)

    training_set = data[:split]
    testing_set = data[split:]

    return training_set, testing_set


# Accuracy
def calculate_accuracy(predicted, actual):
    count = 0
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            count += 1

    return count / len(actual)