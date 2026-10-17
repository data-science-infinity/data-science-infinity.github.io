---
layout: post
title: Image Processing Using NymPy
image: "/posts/camaro.jpg"
tags: [NumPy, Image Processing]
---

Digital images are seen as arrays. In the first part of this post, I am going to use NumPy to crop an image (horizontally and/or vertically). Grayscale images can be thought of as a collection of pixels on a 2D grid. Colour images can therefore be thought of as a similar pixel collection, only that each pixel is now containing information on 3 aspects - those being the Red-Green-Blue colours. In the second part of this post, I am going to create 3 separate copy images [by flooding the original image with either red, green or blue] which will then be stacked to create a poster.

---

import numpy as np

from skimage import io # the io module allows us to read in an image

import matplotlib.pyplot as plt

### provided our image is located in the same working directory as this script

camaro = io.imread('camaro.jpg')
print(camaro)   # image is actually an array

camaro.shape  # a coloured image is a 3d array

plt.imshow(camaro) # looking at the pane 'Plots' we can see the image
plt.show()


### SLICING of the array will result on cropping the image
#NOTE: the 3rd channel/ axis is the colour channels

cropped = camaro[:, :, :] # no cropping here since we have not specified any slice to crop
plt.imshow(cropped)
plt.show()

cropped = camaro[0:500, :, :] # crop horizontally only [KEEP the slice specified, i.e. pixels 0 to 500]
plt.imshow(cropped)
plt.show()

cropped = camaro[:, 0:400, :] # crop vertically only [KEEP the slice specified, i.e. pixels 0 to 400]
plt.imshow(cropped)
plt.show()

cropped = camaro[350:1100, 200:1400, :] # Crop Vertically & Horizontally [KEEP the car only!]
plt.imshow(cropped)
plt.show()


#### SAVE THE CROPPED IMAGE TO THE SAME WORKING DIRECTORY
io.imsave('camaro_cropped.jpg', cropped)

Here is the final cropped image created!
![alt text](/img/posts/camaro_cropped.jpg "Image-Processing-NumPy")


### We can also FLIP our image using the start-stop-step logic. Using -1 where step is will flip the image

h_flip = camaro[::-1, :, :] # flip the image along an imaginary horizontal binder
plt.imshow(h_flip)
plt.show()

io.imsave('camaro_h_flip.jpg', h_flip) # saves the image to our working directory

v_flip = camaro[:, ::-1, :] # flip the image along an imaginary vertical axis [hint - see plants position!]
plt.imshow(v_flip)
plt.show()

io.imsave('camaro_v_flip.jpg', v_flip) # saves the image to our working directory

Here is the horizontally flipped image created!

![alt text](/img/posts/camaro_h_flip.jpg "Image-Processing-NumPy")



### Colour Channels (RGB) - By zeroing 2 out of 2 channels every time we can see the 1 remaining channel (R, G or B)

### Create a new array with excatly same dimentions as the original image

### create a new array which only has zero values in it (same shape as our original image)
red_array = np.zeros(camaro.shape, dtype = 'uint8') 
red_array[:, :, 0] = camaro [:, :, 0] # imput the RED pixel values only, from the original image)
plt.imshow(red_array)

### create a new 2nd array which only has zero values in it (same shape as our original image)

green_array = np.zeros(camaro.shape, dtype = 'uint8')
green_array[:, :, 1] = camaro [:, :, 1] # imput the GREEN pixel values only, from the original image
plt.imshow(green_array)

### create a new 3rd array which only has zero values in it (same shape as our original image)

blue_array = np.zeros(camaro.shape, dtype = 'uint8')
blue_array[:, :, 2] = camaro [:, :, 2] # imput the BLUE pixel values only, from the original image
plt.imshow(blue_array)


### STACK the 3 newly created arrays to create a poster image


v_poster = np.vstack((red_array, green_array, blue_array)) # vertical stack --> portrait poster
plt.imshow(v_poster)

h_poster = np.hstack((red_array, green_array, blue_array)) # horizontal stack --> landscape poster
plt.imshow(h_poster)

io.imsave('camaro_h_poster.jpg', h_poster)


Here is the final poster created!

![alt text](/img/posts/camaro_h_poster.jpg "Image-Processing-NumPy")



