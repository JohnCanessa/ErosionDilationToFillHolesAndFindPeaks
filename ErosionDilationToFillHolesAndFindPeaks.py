# **** imports ****
import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage import data

from skimage.exposure import rescale_intensity
from skimage.morphology import reconstruction


# **** read image ****
moon = io.imread('./images/moon.jpg')

# **** display moon image ****
plt.figure(figsize=(8, 8))
plt.imshow(moon, cmap='gray')
plt.title('Moon')
plt.show()


# **** let's work with a higher constrast image ****
moon_rescaled = rescale_intensity(  moon,
                                    in_range=(30, 200))

# **** display moon_rescaled image ****
plt.figure(figsize=(8, 8))
plt.imshow(moon_rescaled, cmap='gray')
plt.title('Moon Rescaled')
plt.show()


# **** create a seed 
#      the seed image used for erosion is initialized 
#      to the maximum value of the original image ****
erosion_seed = np.copy(moon_rescaled)

# **** except at the borders which is the 
#      starting point of the erosion process  ****
erosion_seed[1:-1, 1:-1] = moon_rescaled.max()

# **** create mask ****
mask = moon_rescaled

# **** apply erosion
#      eroding inwards from the border fills holes - as the holes are
#      by definition surrounded by pixels of higher intensities ****
filled = reconstruction(erosion_seed, 
                        mask, 
                        method='erosion')

# **** display side by side the images ****
fig, ax = plt.subplots( nrows=1,
                        ncols=2, 
                        figsize=(14, 12),
                        sharex=True,
                        sharey=True)

# **** flatten the axes ****
ax = ax.ravel()

# **** display the images ****
ax[0].imshow(moon_rescaled, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(filled, cmap='gray')
ax[1].set_title('Erosion Reconstruction')
ax[1].axis('off')

plt.tight_layout()
plt.show()


# **** create a seed for dilation ****
dilation_seed = np.copy(moon_rescaled)

# **** the seed image used for dilation is initialized
#      to the minimum value of the original image 
#      except at the borders which is the
#      starting point of the dilation process
dilation_seed[1:-1, 1:-1] = moon_rescaled.min()

# **** mask ****
mask = moon_rescaled

# **** apply dilation 
#      highlights bright spots in an image by
#      expanding maximal values in local areas ****
highlighted = reconstruction(   dilation_seed,
                                mask,
                                method='dilation')

# **** display side by side the images ****
fig, ax = plt.subplots( nrows=1,
                        ncols=2,
                        figsize=(14, 12),
                        sharex=True,
                        sharey=True)

# **** flatten the axes ****
ax = ax.ravel()

# **** display the images ****
ax[0].imshow(moon_rescaled, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(highlighted, cmap='gray')
ax[1].set_title('Dilation Reconstruction')
ax[1].axis('off')

plt.tight_layout()
plt.show()


# **** isolate dark regions ~ erode - original ****
holes = filled - moon_rescaled
peaks = highlighted - moon_rescaled

# **** OR issolate bright regions ~ dilate - original ****
# bright = moon_rescaled - highlighted
# peaks = highlighted - moon_rescaled


# **** display side by side the images ****
fig, ax = plt.subplots( nrows=1,
                        ncols=2,
                        figsize=(14, 12),
                        sharex=True,
                        sharey=True)

# **** flatten the axes ****
ax = ax.ravel()

# **** display the images ****
ax[0].imshow(holes, cmap='gray')
ax[0].set_title('Holes')
ax[0].axis('off')

ax[1].imshow(peaks, cmap='gray')
ax[1].set_title('Peaks')
ax[1].axis('off')

plt.tight_layout()
plt.show()



