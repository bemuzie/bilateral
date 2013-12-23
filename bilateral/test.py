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

a=gauss_kernel_3d(2,[0.7,0.7,0.5])
idx_list=[]
inl=0
print a.shape,a.size
o=0
for x in range(a.shape[0]//2,a.shape[0]):
	x_r = x-a.shape[0]//2
	for y in range(a.shape[1]):
		y_r = y-a.shape[1]//2
		if x_r == 0 and y_r>0:
			o+=1
			continue
		for z in range(a.shape[2]):
			z_r = z-a.shape[2]//2
			if x_r == 0 and y_r==0 and z_r>0:
				o+=1
				continue
			if not [-x_r,-y_r,-z_r] in idx_list:
				idx_list+= [[x_r,y_r,z_r]]
				print x_r,y_r,z_r
				continue
			else:
				print 'in list',x_r,y_r,z_r
				inl+=1
			
print idx_list.__len__()
print inl
print o