import numpy as np
import matplotlib.pyplot as plt


def gauss_kernel_3d(sigma,voxel_size):
    # make 3d gauss kernel adaptive to voxel size
    voxel_size=np.asarray(voxel_size, dtype=float)
    # calculate kernel size as distance*sigma from centre
    x,y,z=np.ceil(2*sigma/voxel_size)
    #make 3d grid of euclidean distances from center
    distances=voxel_size*np.ogrid[-x:x+1,-y:y+1,-z:z+1]
    
    distances=np.sum( distances*distances )
    ret = np.exp( distances/ -2*(sigma**2) )
    #/ np.sqrt( np.pi*2*sigma**2)**3
    return ret

a=gauss_kernel_3d(1,[0.7,0.7,0.5])
print a.shape,a.size
for i in np.unique(a):
	print i, '\n'
	print np.argwhere(a==i),'\n'



print np.argwhere(a==1)[0][1]