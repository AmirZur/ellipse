import numpy as np
from math import sqrt, pi

def generate_ellipse(color, circularity, area, size=(28, 28)):
    # Calculate dimensions based on area and circularity
    a = sqrt(area / (pi * circularity))  # Area = pi * a * b, circularity = aspect ratio = b/a, a is major axis
    b = a * circularity                  # Semi-major axis, using circularity = aspect ratio = b/a and the vale of a, b is the minor axis

    # Create an empty image
    img = np.ones(size)  # Set background to white # Note that 0 is clack and 1 is white

    # Center of the image
    cy, cx = size[0] // 2, size[1] // 2 # size is the size of the 28x28 image. This finds the center.
    y, x = np.ogrid[-cy:size[0]-cy, -cx:size[1]-cx] # This creates coordinate system around the center

    # Equation of ellipse
    mask = ((x / a) ** 2 + (y / b) ** 2) <= 1 # This checks whether a given point in the grid (pixel) is in the ellipse.
    img[mask] = color  #

    return img

def normalize_area(area):
    min_area, max_area = 16 * pi, 64 * pi # The max_are was chosen so that z (major axis) wouldn't be greter than 28, the size of the image. This is to prevent scaling.
    # Normalize area to scale from 0 to 1. This is for the creation of label in the next function.
    return (area - min_area) / (max_area - min_area)

def normalize_circularity(circularity):
    return (circularity - .3) / (1 - .3) # Note that this must be coordinated with assignment of circularity below

def create_dataset(n): # n is the number of images
    min_area, max_area = 16 * pi, 64 * pi # The max_are was chosen so that z (major axis) wouldn't be greter than 28, the size of the image. This is to prevent scaling.
    images = []
    labels = []
    for _ in range(n):
        # Random latent variables for ellipse images
        color = round(0.9 * np.random.uniform(0,1), 2) #Random number between 0 and 1, multiple by .9 so that it becomes number between .9 and 0. note that 1 is white and 0 is black.
        circularity = round(np.random.uniform(0.3, 1), 2)  # This is aspect ratio (minor axis / major axis). # Choose a random number between .1 and 1 (.1 because this value can't be 0), then round to two digits
        area = round(np.random.uniform(min_area, max_area),2) # Choose a random number between  min_area and max_area, then round to two digits.

        # Generate ellipse image
        img = generate_ellipse(color, circularity, area)

        # Rotate half of images so some ellipses are extended left-right and others up-down
        if np.random.rand() < 0.5:  # 50% chance
            img = np.rot90(img)

        # Calculate label
        label = [normalize_circularity(circularity), color/.9, round(normalize_area(area),2)] # the .8 is to redistirbute the values between 0 and 1.
        labels.append(label)
        images.append(img)

    return np.array(images), np.array(labels)