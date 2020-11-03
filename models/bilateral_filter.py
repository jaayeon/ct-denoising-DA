import numpy as np
import math
import torch
from skimage.external.tifffile import imread
from skimage.external.tifffile import imsave
import time

def distance(x,y,i,j):
    return torch.sqrt(torch.tensor((x-i)**2+(y-j)**2))

def gaussian(x, sigma):
    return torch.tensor((1.0/(2*math.pi*(sigma**2)))*math.exp(-(x**2)/(2*sigma**2)))

def pixel_bf(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    h1 = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j=0
        while j<diameter:
            neighbour_x = x-(h1-i)
            neighbour_y = y-(h1-j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            # print(source[neighbour_x, neighbour_y])
            gi = gaussian(source[int(neighbour_x),int(neighbour_y)] - source[int(x),int(y)], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y,x,y), sigma_s)
            w = gi*gs
            i_filtered += source[int(neighbour_x),int(neighbour_y)] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x, y] = i_filtered.clone().detach()

    return filtered_image

def bilateral_filter(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = torch.zeros(source.shape) #c,h,w
    #print(source.shape)
    h,w = source.shape

    start = time.time()
    i=0
    while i<h:
        j=0
        while j<w:
            #print(i,j)
            filtered_image = pixel_bf(source, filtered_image, i, j , filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    #print(time.time() - start)
    return filtered_image


''' 
if __name__ == "__main__":
    src = imread('D:/data/denoising/train/lp-mayo/low/L004/simens_low_L004_000.tiff')
    src = np.asarray(src)
    # src = np.expand_dims(src, axis=0)
    src = torch.tensor(src)
    filtered = bilateral_filter(src, 5, 4.0, 4.0)
    # filtered = filtered.clone().detach()
    filtered = filtered.numpy()
    imsave('D:/data/denoising/train/lp-mayo/low/L004/filtered_simens_low_L004_000.tiff', filtered ) '''