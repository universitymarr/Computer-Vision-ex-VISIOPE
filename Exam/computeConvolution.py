# -----------------------------------------------------------
# Compute convolution using Python and NumPy library
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Released under MIT License
#
# (C) 2023 | Gianmarco Scarano, Rome, Italy
# Email: gianmarcoscarano@gmail.com
# -----------------------------------------------------------
import numpy as np

# Get the padded image, starting from the original image
#
# Parameters
# ----------
# img : 
#     Image to apply padding on (Type = np.array)
# mode :
#     Type of the padding used.
#       - zero : 
#           Set all pixels outside the source image to 0.
#       - reflect :
#           In reflect padding, when padding an image, the pixels at the image border are 
#           mirrored or reflected inwards to fill the additional padding cells.
#           Also known as "Reflection / Mirror".
#       - symmetric :
#           The padding pixels are added such that they mirror the edge of the image.
#       - constant : 
#           Set all pixels outside the source image to a specified border value.
#       - edge :
#           The pixel values at the edge of the active frame are repeated to make rows 
#           and columns of padding pixels. Also known as "Clamp / Replicate"
# constantValue : 
#     Value to apply for constant padding. It's equal to 0 if using the 'zero' padding.
# ----------
def getPaddedImage(img, mode, constantValue):
    padding_modes = ['zero', 'constant', 'reflect', 'symmetric', 'edge']
    if mode not in padding_modes:
        print("Please specify a valid padding mode between: zero, reflect, symmetric & constant.")
        print(F"You used: '{mode}'. Exiting..")
        exit(0)
    if mode == 'constant':
        return np.pad(img, pad_width=1, mode=mode, constant_values=constantValue)
    if mode == 'zero':
        return np.pad(img, pad_width=1, mode='constant', constant_values=0)
    return np.pad(img, pad_width=1, mode=mode)

# ===================== A R G S =====================
image = np.array([[3, 1, 2, 0],
                [0, 2, 1, 5],
                [1, 0, 1, 3],
                [1, 2, 0, 5]], np.int32)

kernel_init = np.array([[-2, 3, 2],
                        [0, -1, -2],
                        [1, -2, 0]], np.int32)

# Refer to line 35 for further explaination
padding = 'zero'  # zero / reflect / symmetric / constant / edge

# Constant value if using the 'constant' padding
# Use C = 0 if using the 'zero' padding
C = 0

coord = [(3, 3), (1, 1), (2, 4)]
# ===================================================

kernel = np.flipud(np.fliplr(kernel_init))
print(F"Original kernel: \n{kernel_init}")
print("-------------------------------")
print(F"Flipped kernel: \n{kernel}")
print("-------------------------------")

print("# ===== STARTING ALGORITHM ===== #")
for values in coord:
    imageToBeComputed = image.copy()
    coordOne, coordTwo = values

    usePadding = (coordOne in [1, image.shape[0]]) or (coordTwo in [1, image.shape[1]])

    if usePadding:
        imageToBeComputed = getPaddedImage(image, padding, constantValue=C)
        coordOne += 1
        coordTwo += 1
        subset = imageToBeComputed[coordOne - 2: coordOne + 1, coordTwo - 2: coordTwo + 1]
    else:
        subset = imageToBeComputed[coordOne - 2: coordOne + 1, coordTwo - 2: coordTwo + 1]

    print("Subset of the image to be computed:")
    print(subset)
    
    # Convolution
    final = np.sum(np.multiply(subset, kernel))
    print(F"Convolution at coord {values}: {final}")
    print("==================================")
