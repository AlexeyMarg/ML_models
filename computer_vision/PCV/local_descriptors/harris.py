from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt

def compute_harris(img, sigma):
    '''вычисление функции отклика детектора углов Харриса для пикселей полутонового изображения'''

    # Вычисляем производные
    imx = np.zeros(img.shape)
    gaussian_filter(img, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(img.shape)
    gaussian_filter(img, (sigma, sigma), (1, 0), imy)

    # Вычисляем элементы матрицы Харриса
    Wxx = gaussian_filter(imx*imx, sigma)
    Wxy = gaussian_filter(imx* imy, sigma)
    Wyy = gaussian_filter(imy*imy, sigma)

    # определитель и след матрицы
    Wdet = Wxx * Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    '''возвращает углы на изображении, построенном по функции отклика Харриса,
    min_dist - минимальное число пикселей между углами и границей изображения'''

    # найти отчки-кандидаты, для которых функция отклика больше порога
    corner_threshold = harrisim.max() * threshold
    harrissim_t = (harrisim > corner_threshold) * 1

    # получить координаты кандидатов
    coords = np.array(harrissim_t.nonzero()).T

    # и их значения
    candidate_values = [harrissim_t[c[0], c[1]] for c in coords]

    # сортировка кандидатов
    index = np.argsort(candidate_values)

    # сохранить данные о точках-кандидатах в массиве
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist: -min_dist, min_dist: -min_dist] = 1

    # выбор наилучших точек с учетом min_dist
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0]-min_dist):(coords[i, 0]+min_dist), (coords[i, 1]-min_dist):(coords[i, 1]+min_dist)]
    return filtered_coords


def plot_harris_points(img, filtered_coords):
    '''нанести на график углы'''
    plt.figure()
    plt.gray()
    plt.imshow(img)
    plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    plt.axis('off')
    plt.show()

def match(desc1, desc2, threshold=0.5):
    '''для каждого дескриптора угловой точки в первом изображении
    найти соответсвующую ему точку во втором изображении,
    применяя нормированную взаимную корреляцию'''

    n = len(desc1)

    # попарные расстояния
    d = -np.ones((len(desc1),len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            ncc_value = sum(d1 * d2) / (n-1) 
            if ncc_value > threshold:
                d[i,j] = ncc_value
            
    ndx = np.argsort(-d)
    matchscores = ndx[:,0]
    
    return matchscores

def match_twosided(desc1,desc2,threshold=0.5):
    """ Two-sided symmetric version of match(). """
    
    matches_12 = match(desc1,desc2,threshold)
    matches_21 = match(desc2,desc1,threshold)
    
    ndx_12 = np.where(matches_12 >= 0)[0]
    
    # удалить несимметричные
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
    
    return matches_12

def appendimages(im1,im2):
    """ Return a new image that appends the two images side-by-side. """
    
    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]    
    rows2 = im2.shape[0]
    
    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1-rows2,im2.shape[1]))),axis=0)
    # if none of these cases they are equal, no filling needed.
    
    return np.concatenate((im1,im2), axis=1)
    
    
def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    """ Show a figure with lines joining the accepted matches 
        input: im1,im2 (images as arrays), locs1,locs2 (feature locations), 
        matchscores (as output from 'match()'), 
        show_below (if images should be shown below matches). """
    
    im3 = appendimages(im1,im2)
    if show_below:
        im3 = np.vstack((im3,im3))
    
    plt.imshow(im3)
    
    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plt.plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
    plt.axis('off')