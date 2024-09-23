from PIL import Image
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import os

def get_imlist(path):
    """    Returns a list of filenames for 
        all jpg images in a directory. 
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def convert_to_grayscale(imlist):
    """    Convert a set of images to grayscale. """
    
    for imname in imlist:
        im = Image.open(imname).convert("L")
        im.save(imname)

def imresize(im,sz):
    """    Resize an image array using PIL. """
    pil_im = Image.fromarray(np.uint8(im))
    
    return np.array(pil_im.resize(sz))


def histeq(im,nbr_bins=256):
    """    Histogram equalization of a grayscale image. """
    
    # get image histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize
    
    # use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    
    return im2.reshape(im.shape), cdf

def compute_average(imlist):
    average_img = np.array(Image.open(imlist[0]), 'f')
    skipped = 0
    for imname in imlist[1:]:
        try:
            average_img += np.array(Image.open(imname))
        except:
            print('Skipped image ', imname)
            skipped += 1
        average_img /= (len(imlist) - skipped)
    return np.array(average_img, np.uint8)