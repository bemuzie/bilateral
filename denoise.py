import pyximport; pyximport.install()
from bilateral import cython_bilateral3d

import numpy as np
from scipy import ndimage
import nibabel as nib

import sys, os


ARGS = sys.argv[1:]
HELP = """
format 
denoise [input image] [output image] [args]
args:
-gs
-is
-crop
"""


INPUT_IMAGE = sys.argv[1]
gauss_sigma = float(sys.argv[ sys.argv.index('-gs')+1 ])
intensity_sigma = float(sys.argv[ sys.argv.index('-is')+1 ])

f = nib.load(os.path.join(INPUT_IMAGE))
im = f.get_data()
hdr = f.get_header()
mrx = hdr.get_sform()
vox_size = np.absolute( [mrx[i,i] for i in range(3)] )

try:
	crop = map(int, sys.argv[ sys.argv.index('-crop')+1 ].split(',')  )
	im=im[
		crop[0]:crop[1], 
		crop[2]:crop[3], 
		crop[4]:crop[5]
		]
except:
	print 'Error'
	pass
print im.shape


if len(im.shape)== 3:
	im=cython_bilateral3d.bilateral3d_optimized( np.asarray(im,dtype='float32', order='C'), 
												 vox_size,
												 gauss_sigma,
												 intensity_sigma)
elif len(im.shape) == 4:
	for time_idx in range(im.shape[-1]):
		
		im[...,time_idx]=cython_bilateral3d.bilateral3d_optimized(np.asarray(im[...,time_idx],dtype='float32', order='C'), 
				    											  vox_size,
				    											  gauss_sigma,
				    											  intensity_sigma)

nib.nifti1.save(nib.Nifti1Image(im, mrx), 
                        INPUT_IMAGE[:-4]+'_filtered.nii')

