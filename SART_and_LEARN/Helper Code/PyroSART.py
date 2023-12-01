# -----------------------------------------------------------------------
# Copyright: 2010-2018, imec Vision Lab, University of Antwerp
#            2013-2018, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
import argparse
from glob import glob
import SynoGenerator
import numpy as np
import os
import pdb
import math
import pylab
import pdb
from matplotlib import pyplot as plt # for png conversion

import tensorflow as tf
#Used for file processing:
import os
from glob import glob
import argparse
#Pyro-NN information:
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.geometry.geometry_fan_2d import GeometryFan2D

from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan, primitives_2d, primitives_3d
from pyronn.ct_reconstruction.helpers.misc.generate_sinogram import generate_sinogram
from pyronn.ct_reconstruction.layers import projection_2d
from pyronn.ct_reconstruction.layers import projection_3d
from pyronn.ct_reconstruction.layers import backprojection_2d


class test:
    
  def example_pyro_SART():
  
      parser = argparse.ArgumentParser(description='')
      parser.add_argument('--sino', dest='infile', default='.', help='input file -- sinogram in .flt format')
      parser.add_argument('--out', dest='outfile', default='.', help='output directory')
      parser.add_argument('--numpix',dest='numpix',type=int,default=512,help='size of volume (n x n )')
      parser.add_argument('--dx',dest='dx',type=float,default=1.,help='pixel size (cm)');
      parser.add_argument('--numbin',dest='numbin',type=int,default=729,help='number of detector pixels')
      parser.add_argument('--ntheta',dest='numtheta',type=int,default=900,help='number of angles')
      parser.add_argument('--nsubs',dest='ns',type=int,default=1,help='number of subsets. must divide evenly into number of angles')
      parser.add_argument('--range', dest='theta_range',type=float,nargs=2,default=[0, 180],help='starting and ending angles (deg)')
      parser.add_argument('--geom', dest='geom',default='fanflat',help='geometry (parallel or fanflat)')
      parser.add_argument('--dso',dest='dso',type=float,default=100,help='source-object distance (cm) (fanbeam only)')
      parser.add_argument('--dod',dest='dod',type=float,default=100,help='detector-object distance (cm) (fanbeam only)')
      parser.add_argument('--fan_angle',dest='fan_angle',default=35,type=float,help='fan angle (deg) (fanbeam only)')
      parser.add_argument('--numits',dest='num_its',default=32,type=int,help='maximum number of iterations')
      parser.add_argument('--beta',dest='beta',default=1.,type=float,help='relaxation parameter beta')
      parser.add_argument('--x0',dest='x0_file',default='',help='initial image (default: zeros)')
      parser.add_argument('--xtrue',dest='xtrue_file',default='',help='true image (if available)')
      parser.add_argument('--sup_params',dest='sup_params', type=float,nargs=3,help='superiorization parameters gamma, N, alpha_init')
  
      #get arguments from command line
      args = parser.parse_args()
      infile, outfile, x0file, xtruefile = args.infile, args.outfile, args.x0_file, args.xtrue_file
      numpix, dx, numbin, numtheta, ns, numits, beta = args.numpix, args.dx, args.numbin, args.numtheta, args.ns, args.num_its, args.beta
      theta_range, geom, dso, dod, fan_angle = args.theta_range, args.geom, args.dso, args.dod, args.fan_angle
  
      volume_shape = [numpix, numpix]
      volume_spacing = [1, 1]
      detector_spacing = 1
      projection_number = 900
      angular_range_pi = math.radians(theta_range[1] - theta_range[0])
      #phantom = shepp_logan.shepp_logan_enhanced(volume_shape)
  
      if args.sup_params is None:
          use_sup = False
      else:
           use_sup = True
           gamma = args.sup_params[0]
           N = np.uint8(args.sup_params[1])
           alpha = args.sup_params[2]
                
  
      eps = np.finfo(float).eps
  
      #read in sinogram
      sino = np.fromfile(infile,dtype='f')
      sino = sino.reshape(numtheta,numbin)
      #max_sino_pixel = np.amax(sino)
      #sino = (sino/max_sino_pixel) * 255
      #sino = tf.convert_to_tensor(sino, np.float32)
  
      head, tail = os.path.split(infile)      #get name of file for output
      head, tail = tail.split("_",1)    #extract numerical part of filename only. Assumes we have ######_sino.flt
      outname = outfile + "/" + head + "_img.flt"
  
      #generate array of angular positions
      theta_range = np.deg2rad(theta_range) #convert to radians
      angles = theta_range[0] + np.linspace(0,numtheta-1,numtheta,False)*(theta_range[1]-theta_range[0])/numtheta #
  
      if x0file == '':        
          f = np.zeros((numpix,numpix))
          f = tf.convert_to_tensor(f, np.float32)
      else:
          f = np.fromfile(x0file,dtype='f')
          f = f.reshape(numpix,numpix)
          f = tf.convert_to_tensor(f, np.float32)
  
      if xtruefile == '':
          calc_error = False
          xtrue = np.zeros((numpix,numpix))
          xtrue = tf.convert_to_tensor(xtrue, np.float32)
      else:
          xtrue = np.fromfile(xtruefile,dtype='f')
          xtrue = xtrue.reshape(numpix,numpix)
          xtrue = tf.convert_to_tensor(xtrue, np.float32)
          calc_error = True
          
      #create projectors and normalization terms (corresponding to diagonal matrices M and D) for each subset of projection data
      
      #Set tensor for generation
      config = tf.ConfigProto()
      config.gpu_options.per_process_gpu_memory_fraction = 0.5
      config.gpu_options.allow_growth = True
    
      with tf.Session(config=config) as sess:
      
        geometry = GeometryParallel2D(volume_shape,volume_spacing,numbin, 1, numtheta, angular_range_pi)
        geometry.set_ray_vectors(circular_trajectory.circular_trajectory_2d(geometry))
        temp_D = tf.ones([numtheta, numbin])
        D = backprojection_2d.parallel_backprojection2d(temp_D, geometry)
        D = tf.maximum(D,eps)
        temp_M = tf.ones([numpix, numpix])
        M = projection_2d.parallel_projection2d(temp_M, geometry)
        M = tf.maximum(M*dx,eps)
    
        x_initial = tf.zeros((numpix, numpix))
        #x_initial = tf.convert_to_tensor(x_initial, np.float32)
    
        current_x = x_initial
        for k in range(numits):
           # pdb.set_trace()
            fp = projection_2d.parallel_projection2d(current_x, geometry)
            diff = tf.math.subtract(fp*dx, sino)
    
            tmp1 = tf.divide(diff, M)
    
            bp = backprojection_2d.parallel_backprojection2d(tmp1, geometry)
            ind2 = tf.greater(tf.abs(bp), 1e3)
            zeros = tf.zeros_like(bp)
            bp = tf.where(ind2,bp,zeros)
            #bp = tf.convert_to_tensor(bp, np.float32)
    
            tmp2 = tf.divide(bp, D)
    
            current_x = tf.math.subtract(current_x, tmp2)                
            current_x = tf.maximum(current_x,eps);                  #set any negative values to small positive value
  #          current_x = tf.to_float(current_x)
  #          out_image = current_x.eval()
            
    
      #save image
        out_image = current_x.eval()

      out_image.tofile(outname)
  
      #**********save image as png**********
      max_pixel = np.amax(out_image)
      img = (out_image/max_pixel) * 255
      img = np.round(img)
    
      plt.figure(num=None, figsize=(30, 40), facecolor='w', edgecolor='k')
      #plt.style.use('grayscale')
      plt.imshow(img, interpolation = 'nearest')
      png_outname = (outname + 'testpyro.png')
      plt.savefig(png_outname)
      #**************************************
  
  

 
  if __name__ == '__main__':
    example_pyro_SART()
