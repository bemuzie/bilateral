# cython: profile=True
 #   Copyright 2012 Denis Nesterov
#   cireto@gmail.com
#
#   The software is licenced under BSD licence.
"""
  A cython implementation of bilateral filtering.
"""

import numpy as np
cimport numpy as np
cimport cython



DTYPE= np.int
DTYPEfloat = np.float32
ctypedef np.int_t DTYPE_t
ctypedef np.float32_t DTYPEfloat_t

cdef extern from "math.h":
     float exp(float x)


def gauss_kernel_3d(sigma,voxel_size):
    # make 3d gauss kernel adaptive to voxel size
    voxel_size=np.asarray(voxel_size, dtype=float)
    # calculate kernel size as distance*sigma from centre
    x,y,z=np.ceil(2*sigma/voxel_size)
    #make 3d grid of euclidean distances from center
    distances=voxel_size*np.ogrid[-x:x+1,-y:y+1,-z:z+1]
    print distances
    distances=np.sum( distances*distances )
    ret = np.exp( distances/ -2*(sigma**2) )
    #/ np.sqrt( np.pi*2*sigma**2)**3
    return ret
@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double calc_weight(double img_value1, double img_value2, double gauss_weight, double sigi_double_sqr):
    return gauss_weight*exp( -((img_value1-img_value2)**2)/ sigi_double_sqr )


@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def bilateral3d(np.ndarray[DTYPEfloat_t, ndim=3, mode="c"] data,voxel_size,double sigg,double sigi):
    if data.ndim<3:
        raise ValueError("Input image should have 4 dimensions")

    assert data.dtype == DTYPEfloat

    cdef int imgSize_x=data.shape[0]
    cdef int imgSize_y=data.shape[1]
    cdef int imgSize_z=data.shape[2]
    
    cdef double value

    cdef np.ndarray[DTYPEfloat_t, ndim=3, mode="c"] gaus_kern3d=np.asarray(gauss_kernel_3d(sigg, voxel_size),dtype=DTYPEfloat)
    cdef int kernelSize_x=gaus_kern3d.shape[0], kside_x=kernelSize_x // 2
    cdef int kernelSize_y=gaus_kern3d.shape[1], kside_y=kernelSize_y // 2
    cdef int kernelSize_z=gaus_kern3d.shape[2], kside_z=kernelSize_z // 2

    print kside_x,kside_y,kside_z
    cdef np.ndarray[DTYPEfloat_t, ndim=3, mode="c"] result=np.zeros([imgSize_x,imgSize_y,imgSize_z],dtype=DTYPEfloat)
    
    # calculate 2*sigma^2 of intensity closeness function out from loop
    cdef double sigiSqrDouble=2*sigi**2
    cdef double weight_i, weights
    cdef double central_voxel, kernel_voxel, gauss_voxel
    cdef unsigned int x,y,z,
    cdef int xk,yk,zk
    cdef double result_value
    cdef DTYPEfloat_t low_border = -100
    cdef DTYPEfloat_t up_border = 200



    for x in range(kside_x, imgSize_x - kside_x - 1):
        for y in range(kside_y, imgSize_y - kside_y - 1):
            for z in range(kside_z, imgSize_z - kside_z - 1):
                
                value = 0.0
                weights=0.0
                central_voxel=data[x,y,z]
                if central_voxel< low_border or central_voxel>up_border:
                    result[x,y,z] = central_voxel
                else:
                    for xk in range(-kside_x,kside_x+1):
                        for yk in range(-kside_y,kside_y+1):
                            for zk in range(-kside_z,kside_z+1):
                                kernel_voxel = data[<unsigned int>(x+xk),<unsigned int>(y+yk),<unsigned int>(z+zk)]
                                gauss_voxel = gaus_kern3d[<unsigned int>(xk+kside_x), <unsigned int> (yk+kside_y), <unsigned int> (zk+kside_z)]
                                weight_i=gauss_voxel * exp( -(kernel_voxel - central_voxel)**2 / sigiSqrDouble)
                                value+=kernel_voxel*weight_i
                                #if x%10==0 and y%10==0 and z%100 == 0 :
                                #   print weight_i, data[x+xk,y+yk,z+zk]
                                weights+=weight_i
                    result[x,y,z]= value/weights
    return result


@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def bilateral3d_optimized(float [:,:,:] data,voxel_size,float sigg,float sigi):
    if data.ndim<3:
        raise ValueError("Input image should have 4 dimensions")

    #assert data.dtype == DTYPEfloat

    #make 3d gaussian kernel
    g_kernel=gauss_kernel_3d(sigg, voxel_size)
    #making it asimmetrical. The first element of new kernel should be the cenral element of old
    #cdef np.ndarray[DTYPEfloat_t, ndim=3, mode="c"] gauss_kernel = np.asarray( g_kernel[ [slice(i/2,i) for i in g_kernel.shape] ],
    #                                                                           dtype=DTYPEfloat,
    #                                                                           order='C'
    #                                                                           )
    cdef np.ndarray[DTYPEfloat_t, ndim=3, mode="strided"] gauss_kernel = np.asarray( g_kernel,
                                                                               dtype=DTYPEfloat,
                                                                               order='C'
                                                                               )
    print [data.shape[i] for i in range(3)]
    cdef unsigned int img_size_x = data.shape[0], img_size_y = data.shape[1], img_size_z = data.shape[2] 
    cdef unsigned int ker_side_x = gauss_kernel.shape[0], ker_side_y = gauss_kernel.shape[1], ker_side_z = gauss_kernel.shape[2]
    print gauss_kernel
    print ker_side_x, ker_side_y, ker_side_z

    #make "buffer" arrays for results
    cdef float [:,:,:] result_values=np.zeros([img_size_x,img_size_y,img_size_z],dtype=DTYPEfloat)
    cdef float [:,:,:] result_weights=np.zeros([img_size_x,img_size_y,img_size_z],dtype=DTYPEfloat)



    cdef unsigned int ker_step_x, ker_step_y, ker_step_z, x, y, z
    cdef signed int img_step_x, img_step_y, img_step_z
    cdef float sigi_double_sqr = 2*sigi**2
    cdef float img_value1, img_value2, gauss_weight, intensity_weight, weight
    for ker_step_x in range(ker_side_x//2,ker_side_x):
        img_step_x = ker_step_x-ker_side_x//2
        for ker_step_y in range(0,ker_side_y):
            img_step_y = ker_step_y-ker_side_y//2
            if img_step_x==0 and img_step_y>0:
                continue
            for ker_step_z in range(0,ker_side_z):
                img_step_z = ker_step_z-ker_side_z//2
                if img_step_x==0 and img_step_y==0 and img_step_z>0:
                     continue
                gauss_weight = gauss_kernel [ker_step_x, ker_step_y, ker_step_z]


                for x in range(ker_side_x//2, img_size_x-ker_step_x):
                    for y in range(ker_side_y//2, img_size_y-ker_step_y):
                        for z in range(ker_side_z//2, img_size_z-ker_step_z):
                            img_value1 = data[x, y, z]
                            img_value2 = data [x+img_step_x, y+img_step_y, z+img_step_z]
                            
                            #weight = calc_weight(img_value1, img_value2, gauss_weight, sigi_double_sqr)
                            weight = gauss_weight * exp( -((img_value1-img_value2)**2)/ sigi_double_sqr )
                            result_values [x, y, z] += img_value2*weight
                            result_values [x+img_step_x, y+img_step_y, z+img_step_z] += img_value1*weight
                            result_weights [x, y, z] += weight
                            result_weights [x+img_step_x, y+img_step_y, z+img_step_z] += weight
    return np.array(result_values)/np.array(result_weights)

@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def bilateral3d_optimized2(np.ndarray[DTYPEfloat_t, ndim=3, mode="c"] data,voxel_size,double sigg,double sigi):
    if data.ndim<3:
        raise ValueError("Input image should have 4 dimensions")

    assert data.dtype == DTYPEfloat

    #make 3d gaussian kernel
    g_kernel=gauss_kernel_3d(sigg, voxel_size)
    #making it asimmetrical. The first element of new kernel should be the cenral element of old
    cdef np.ndarray[DTYPEfloat_t, ndim=3, mode="c"] gauss_kernel = np.asarray( g_kernel[ [slice(i/2,i) for i in g_kernel.shape] ],
                                                                               dtype=DTYPEfloat,
                                                                               order='C'
                                                                               )
   

    print [data.shape[i] for i in range(3)]
    cdef unsigned int img_size_x = data.shape[0], img_size_y = data.shape[1], img_size_z = data.shape[2] 
    cdef unsigned int ker_side_x = gauss_kernel.shape[0], ker_side_y = gauss_kernel.shape[1], ker_side_z = gauss_kernel.shape[2]
    print gauss_kernel
    print ker_side_x, ker_side_y, ker_side_z

    #make "buffer" arrays for results
    cdef np.ndarray[DTYPEfloat_t, ndim=3, mode="c"] result_values=np.zeros([img_size_x,img_size_y,img_size_z],dtype=DTYPEfloat)
    cdef np.ndarray[DTYPEfloat_t, ndim=3, mode="c"] result_weights=np.zeros([img_size_x,img_size_y,img_size_z],dtype=DTYPEfloat)



    cdef unsigned int ker_step_x, ker_step_y, ker_step_z, x, y, z

    cdef double sigi_double_sqr = 2*sigi**2
    cdef double img_value1, img_value2, gauss_weight, intensity_weight, weight

    for ker_step_x in range(ker_side_x):
        for ker_step_y in range(ker_side_y):
            for ker_step_z in range(ker_side_z):
                gauss_weight = gauss_kernel [ker_step_x, ker_step_y, ker_step_z]
                
                for x in range(img_size_x-ker_side_x):
                    for y in range(img_size_y-ker_side_y):
                        for z in range(img_size_z-ker_side_z):
                            img_value1 = data [x, y, z]
                            img_value2 = data [x+ker_step_x, y+ker_step_y, z+ker_step_z]
                            
                            intensity_weight = exp( -((img_value1-img_value2)**2)/ sigi_double_sqr )


                            weight = intensity_weight*gauss_weight

                            result_values [x, y, z] += img_value2*weight
                            result_values [x+ker_step_x, y+ker_step_y, z+ker_step_z] += img_value1*weight
                            result_weights [x, y, z] += weight
                            result_weights [x+ker_step_x, y+ker_step_y, z+ker_step_z] += weight

    return result_values,result_weights



@cython.boundscheck(False) 
@cython.wraparound(False)
@cython.cdivision(True)
def bilateral_function(np.ndarray[DTYPEfloat_t, ndim=1, mode="c"] data,
                       np.ndarray[DTYPEfloat_t, ndim=1, mode="c"] gaussian_kernel,
                       double sigi_double_sqr,
                       int central_idx,
                       int data_size):
    
    cdef double central_px, weights, weight_i, diff, px, value, gk
    cdef unsigned int i
    central_px=data[central_idx]

    
    weights = 0
    if central_px< -100. or central_px> 200:
        return central_px
    for i in range(data_size):
        px = data[i]
        diff = (px-central_px)
        gk = gaussian_kernel[i]
        weight_i = gk*exp(-diff*diff / sigi_double_sqr)
        value += px*weight_i
        weights +=weight_i
    return value / weights
