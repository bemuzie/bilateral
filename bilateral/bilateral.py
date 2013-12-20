import numpy as np
from scipy import ndimage

def gauss_kernel_3d(sigma,voxel_size,distance=3):
    # make 3d gauss kernel adaptive to voxel size
    voxel_size=np.asarray(voxel_size, dtype=float)
    # calculate kernel size as distance*sigma from centre
    x,y,z=np.ceil(distance*sigma/voxel_size)
    #make 3d grid of euclidean distances from center
    distances=voxel_size*np.ogrid[-x:x+1,-y:y+1,-z:z+1]
    distances*=distances
    distances=distances[0]+distances[1]+distances[2]
    return np.exp( distances/ -2*sigma**2 )/ np.sqrt( np.pi*2*sigma**2)**3

def bilateral_kernel(data,gaussian_kernel,central_idx,sigi_double_sqr):
    #Not optimized bilateral function for generic filter
    #gaussian_kernel - raveled gaussian kernel with len(gaussian_kernel) == len(data)
    #sigi_double_sqr - 2*(intensity sigma)**2
    central_px = data[central_idx]
    diff = data-central_px
    intensity_kernel = np.exp((diff*diff)/sigi_double_sqr)
    weights = intensity_kernel*gaussian_kernel
    return np.sum(data*weights) / np.sum(weights)

def bilateral_generic(img,voxel_size,sigg,sigi,bilateral_function = bilateral_kernel):
    sigISqrDouble=float(sigi*sigi*2)
    gkern=gauss_kernel_3d(sigg, voxel_size)
    print gkern.shape
    ksize=np.shape(gkern)
    gaus_kern=np.ravel(gkern)
    #Closness function
    kwargs=dict(gaussian_kernel=gaus_kern,
    			sigi_double_sqr=sigISqrDouble,
    			central_idx=len(gaus_kern)/2,
    			data_size=len(gaus_kern))
    img_filtered=ndimage.generic_filter(img,bilateral_kernel,size=ksize,extra_keywords=kwargs)
    return img_filtered