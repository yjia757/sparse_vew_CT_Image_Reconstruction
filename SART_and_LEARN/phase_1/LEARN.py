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



class LEARN():

    def __init__(self,infile,outfile,x0_file,xtrue_file,
                numpix,dx,numbin,numtheta,numits, 
                geom,sino,geometry, num_epochs):
        
        self.infile, self.outfile, self.x0file, self.xtruefile = infile, outfile, x0_file, xtrue_file
        self.numpix, self.dx, self.numbin, self.numtheta, self.numits = numpix, dx, numbin, numtheta, numits
        self.geom = geom
        self.volume_shape = [numpix, numpix]
        self.true_sino = true_sino
        self.noisy_sino = noisy_sino
        self.sino = sino
        self.geometry = geometry
        self.num_epochs = num_epochs

    def executeAlgorythmn(self):
        eps = np.finfo(float).eps
    
        head, tail = os.path.split(self.infile)      #get name of file for output
        head, tail = tail.split("_",1)    #extract numerical part of filename only. Assumes we have ######_sino.flt
        outname = self.outfile + "/" + head + "_img.flt"
    
    
        if self.x0file == '':        
            f = np.zeros((self.numpix,self.numpix))
            f = tf.convert_to_tensor(f, np.float32)
        else:
            f = np.fromfile(self.x0file,dtype='f')
            f = f.reshape(self.numpix,self.numpix)
            f = tf.convert_to_tensor(f, np.float32)
    
        if self.xtruefile == '':
            calc_error = False
            xtrue = np.zeros((self.numpix,self.numpix))
            xtrue = tf.convert_to_tensor(xtrue, np.float32)
        else:
            xtrue = np.fromfile(self.xtruefile,dtype='f')
            xtrue = xtrue.reshape(self.numpix,self.numpix)
            xtrue = tf.convert_to_tensor(xtrue, np.float32)
            calc_error = True
            
            
        #Set tensor for generation
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            pdb.set_trace()
        
            x_initial = tf.zeros((self.numpix, self.numpix))
            #x_initial = tf.convert_to_tensor(x_initial, np.float32)
        
            current_x = x_initial
            for k in range(self.numits):
            
                #if(True):
                   # tf.train.AdamOptimizer(
                    #Ax
                if(self.geom == 'parallel'):
                    fp = projection_2d.parallel_projection2d(current_x, self.geometry)
                else:
                    fp = projection_2d.fan_projection2d(current_x, self.geometry)
                #Ax-b
                diff = tf.math.subtract(fp*self.dx, self.sino)
                #At(Ax-b)
                if(self.geom == 'parallel'):
                    bp = backprojection_2d.parallel_backprojection2d(diff, self.geometry)
                else:
                    bp = backprojection_2d.fan_backprojection2d(diff, self.geometry)

                #?  
                ind2 = tf.greater(tf.abs(bp), 1e3)
                zeros = tf.zeros_like(bp)
                bp = tf.where(ind2,zeros,bp)

                #x - At(Ax-b)
                current_x = tf.math.subtract(current_x, bp)                
                current_x = tf.maximum(current_x,eps);                  #set any negative values to small positive value
                
                self.lr = 1/10e4
                lr_decay_rate = ((1/10e4) - (1/10e5)) / self.epocs
                for k in range(self.epocs):
                    self.sess = sess
                    CNN_out = tf.layers.conv2d(current_x, 64, 3, padding='same', activation=tf.nn.relu) #image, filters, kernal size, padding, activation function
                    CNN_out = tf.layers.conv2d(current_x, 64, 3, padding='same', activation=tf.nn.relu)
                    CNN_out = tf.layers.conv2d(current_x, 1, 3, padding='same') #-relu
                    
                    self.loss(tf.nn.l2_loss(sino - CNN_out)
                    optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
                    self.train_op = optimizer.minimize(self.loss)
                    self.lr = self.lr - lr_decay_rate
                    init = tf.initialize_all_variables() #initialize unused tf params
                    self.sess.run(init)
                    #current_x = CNN_out
                    current_x = current_x - CNN_out
                    
                
                
                
                
                def CNN:
                    model = models.Sequential()
                    model.add(tf.layers.conv2d(current_x, 64, 3, padding='same', activation=tf.nn.relu)) #image, filters, kernal size, padding, activation function
                    model.add(tf.layers.conv2d(current_x, 64, 3, padding='same', activation=tf.nn.relu))
                    model.add(tf.layers.conv2d(current_x, 1, 3, padding='same'))
                    
                    model.compile(optimizer='adam'
                    
        
            #save image
            out_image = current_x.eval()

            out_image.tofile(outname)
    
            #**********save image as png**********
            max_pixel = np.amax(out_image)
            img = (out_image/max_pixel) * 255
            img = np.round(img)
            
            plt.figure(num=None, figsize=(30, 40), facecolor='w', edgecolor='k')
            plt.style.use('grayscale')
            plt.imshow(img, interpolation = 'nearest')
            png_outname = (outname + 'testpyro.png')
            plt.savefig(png_outname)
            plt.show()
            #**************************************

if __name__ == '__main__':
    pdb.set_trace()