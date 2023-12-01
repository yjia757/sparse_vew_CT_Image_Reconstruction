import numpy as np
import tensorflow as tf
import pdb                      #Debugger
#Used for file processing:
import os
from glob import glob
import argparse
from SART import SART

#Pyro-NN information:
from pyronn.ct_reconstruction.geometry.geometry_parallel_2d import GeometryParallel2D
from pyronn.ct_reconstruction.geometry.geometry_cone_3d import GeometryCone3D
from pyronn.ct_reconstruction.geometry.geometry_fan_2d import GeometryFan2D

from pyronn.ct_reconstruction.helpers.trajectories import circular_trajectory
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan, primitives_2d, primitives_3d
from pyronn.ct_reconstruction.helpers.misc.generate_sinogram import generate_sinogram
from pyronn.ct_reconstruction.layers import projection_2d
from pyronn.ct_reconstruction.layers import projection_3d

class Reconstructor(object):
    
    def __init__(self, infile, outfile, dx, volume_size, volume_shape, volume_spacing, angular_range, angular_range_PI, 
                detector_shape, fan_angle, detector_spacing, projection_number, geom_type, 
                source_detector_distance, source_isocenter_distance, learning_rate, num_epochs, 
                strategy_type, which_sinos, num_its, beta,x0_file, xtrue_file, sup_params
                ,ns):

        # dx, volume_size, volume_shape, volume_spacing, angular_range, angular_range_PI, 
        #         detector_shape, fan_angle, detector_spacing, projection_number, geom_type, 
        #         source_detector_distance, source_isocenter_distance, learning_rate, num_epochs, 
        #         strategy_type, which_sinos
        self.infile = infile
        self.outfile = outfile
        self.dx = dx
        self.volume_size = volume_size
        self.volume_shape = volume_shape 
        self.volume_spacing = volume_spacing
        self.angular_range = angular_range
        self.angular_range_PI = angular_range_PI 
        self.detector_shape = detector_shape
        self.fan_angle = fan_angle 
        self.detector_spacing = detector_spacing 
        self.projection_number = projection_number 
        self.geom_type = geom_type 
        self.source_detector_distance = source_detector_distance
        self.source_isocenter_distance = source_isocenter_distance
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs 
        self.strategy_type = strategy_type
        self.which_sinos = which_sinos
        self.num_its, self.beta, self.x0_file, self.xtrue_file, self.sup_params = num_its, beta,x0_file, xtrue_file, sup_params
        self.ns = ns    # Not sure if this is right
    """
    Precondition:
    - geom_type: string that specified the geometric type (parallel, fanflat, or cone)
    - volume_size: size of volume
    - volume_shape: shape of volume
    - volume_spacing: spacing of volume
    - angular_range: angular range
    - angular_range_PI: angular range as PI term
    - detector_shape: shape of detector
    - detector_spacing: spacing of detector
    - projection_number: number of projections
    - source_detector_distance: detector distance from source
    - source_isocenter_distance: detector isocenter distance
    Postcondition:
    - It returns a geometry type based on information above
    """
    def chooseGeometry(self,geom_type, volume_shape, volume_spacing, angular_range_PI, detector_shape,detector_spacing, projection_number, source_detector_distance, source_isocenter_distance):
        if geom_type == 'parallel':
            geo = GeometryParallel2D(volume_shape,volume_spacing,detector_shape, detector_spacing,projection_number, angular_range_PI)
            geo.set_ray_vectors(circular_trajectory.circular_trajectory_2d(geo))
            return geo
        elif geom_type == 'fanflat':
            geo = GeometryFan2D(volume_shape,volume_spacing,detector_shape,detector_spacing,projection_number, angular_range_PI, source_detector_distance,source_isocenter_distance)
            geo.set_central_ray_vectors(circular_trajectory.circular_trajectory_2d(geo))
            return geo
        else:
            geo = GeometryCone3D(volume_shape,volume_spacing,detector_shape, detector_spacing, projection_number, angular_range_PI, source_detector_distance,source_isocenter_distance)
            geo.set_projection_matrices(circular_trajectory.circular_trajectory_3d(geo)) 
            return geo

    def reconstruct(self):
        #(dx, volume_size, volume_shape, volume_spacing, angular_range, angular_range_PI, detector_shape, fan_angle, detector_spacing, projection_number, geom_type, source_detector_distance, source_isocenter_distance, learning_rate, num_epochs, strategy_type, which_sinos)
        # Current standard line:

        # Ex 1.:
        # python SynoGenerator.py  --in imgs/00000002_img.flt --out sinos --geom fanflat --angular_range_definition 360 --source_detector_distance 1600 --source_isocenter_distance 800 --dx 0.065

        # python Reconstructor.py --in sinos/00000002_sino.flt --out recons --geom fanflat --angular_range_definition 360 --source_detector_distance 1600 --source_isocenter_distance 800 --dx 0.065


        # Ex 2.:
        # python SynoGenerator.py  --in imgs/00000001_img.flt --out sinos --geom fanflat --angular_range_definition 360 --source_detector_distance 2000 --source_isocenter_distance 500 --dx 0.065


        # python Reconstructor.py --in sinos/00000001_sino.flt --out recons --geom fanflat --angular_range_definition 360 --source_detector_distance 2000 --source_isocenter_distance 500 --dx 0.065

        #File parameters
        args = parser.parse_args()
        infile, outfile = args.infile, args.outfile
        #Geometry parameters

        # Check for fanflat, and uses angle if needed
        if self.geom_type == 'fanflat':
            ft = np.tan( np.deg2rad(self.fan_angle / 2) )    #compute tan of 1/2 the fan angle
            self.detector_spacing = 2 * self.source_detector_distance * ft / self.detector_shape  #width of one detector pixel, calculated based on fan angle
        
        # create Geometry class
        self.geometry = self.chooseGeometry(self.geom_type, self.volume_shape, 
                                        self.volume_spacing, self.angular_range_PI, 
                                        self.detector_shape, self.detector_spacing, self.projection_number,
                                        self.source_detector_distance, self.source_isocenter_distance)
        #pdb.set_trace()

        #read in sinogram
        
        sino = np.fromfile(infile,dtype='f')
        sino = sino.reshape(self.projection_number,self.detector_shape)
        head, tail = os.path.split(infile)      #get name of file for output
        head, tail = tail.split("_",1)    #extract numerical part of filename only. Assumes we have ######_sino.flt
        outname = outfile + "/" + head + "_img.flt"

        if self.strategy_type == 'SART':
            
            strat = SART(self.infile,self.outfile,self.x0_file,self.xtrue_file,
                        self.volume_size,self.dx,self.detector_shape, self.projection_number,
                        self.num_its, self.geom_type,sino, self.geometry, self.num_epochs)
            strat.executeAlgorythmn()

        # if strategy_type == "SART":
        #     # sartStratMethod.execute(geometry,outname,outfile,sino,...)
        # else:
        #     # LEARStratMethod.execute(geometry,outname,outfile,sino,...)

        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        # config.gpu_options.allow_growth = True
        # # ------------------ Call Layers ------------------
        # with tf.Session(config=config) as sess:
        #     #***************** call to_sino rather than GenerateSino to do further processing
        #     acquired_sinogram = to_sino(phantom,projection_2d.parallel_projection2d,geometry)
        #     #***************************************************************************************************
        #     acquired_sinogram = acquired_sinogram + np.random.normal(
        #         loc=np.mean(np.abs(acquired_sinogram)), scale=np.std(acquired_sinogram), size=acquired_sinogram.shape) * 0.02

        #     zero_vector = np.zeros(np.shape(phantom), dtype=np.float32)

        #     #********************* Need to call pipeline strategy and specify method
        #     iter_pipeline = PipelineStrategy(sess, args, geometry, strategy_type)
        #     #******************************************************************
        #     iter_pipeline.train(zero_vector,np.asarray(acquired_sinogram))
        

    ########



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--in', dest='infile', default='.', help='input file -- directory or single file')
    parser.add_argument('--out', dest='outfile', default='.', help='output directory')

    parser.add_argument('--dx', dest='dx', type = float, default = 1, help = "pixel size (cm)")               #DIS
    parser.add_argument('--volume_size_definition', dest = 'volume_size', type = int, default = 512, help = "The volume size in Z, Y, X order")
    
    parser.add_argument('--volume_spacing_definition', dest = 'volume_spacing', type = float, default = [1,1], help = "Define space between volume spaces")
    parser.add_argument('--angular_range_definition', dest = 'angular_range', type = int, default = 180, help = "Range of rotation of scan")
    parser.add_argument('--detector_shape_definition', dest = 'detector_shape', type = int, default = 729, help = "Define space of volume")

    parser.add_argument('--fan_angle', dest = 'fan_angle', type = float, default = 35.0, help = "Angle of fan")
    parser.add_argument('--det_spacing_definition', dest = 'detector_spacing', type = float, default = 1.0, help = "Define space between detector spaces")
    parser.add_argument('--number_of_project', dest = 'project_number', type = int, default = 900, help = "Define number of projections")
    
    parser.add_argument('--geom', dest='geom',default='parallel',help='geometry (parallel, or fanflat (cone not supported))')
    parser.add_argument('--source_detector_distance', dest='source_detector_distance', type = int, default = 1200, help = "Define the distance of the detector") #DSO+DOD
    parser.add_argument('--source_isocenter_distance', dest='source_isocenter_distance', type = int, default = 750, help = "Distance of Isoceles")               #DIS

    parser.add_argument('--lr', dest='learning_rateArg', type=float, default=1e-2, help='initial learning rate for adam')
    parser.add_argument('--epoch', dest='num_epochsArg', type=int, default=1000000, help='# of epoch')
#********************* Add option to choose strategy and specify sinos **************************************
    parser.add_argument('--strat', dest='strategy_typeArg', type=str, default="SART", help = 'image evaluation strategy')
    parser.add_argument('--sinos', dest='which_sinosArg', type=str, default="ONE", help="specify ONE for single image processing, ALL for sinogram construction")

    parser.add_argument('--numits',dest='num_its',default=32,type=int,help='maximum number of iterations')
    parser.add_argument('--beta',dest='beta',default=1.,type=float,help='relaxation parameter beta')
    parser.add_argument('--x0',dest='x0_file',default='',help='initial image (default: zeros)')
    parser.add_argument('--xtrue',dest='xtrue_file',default='',help='true image (if available)')
    parser.add_argument('--sup_params',dest='sup_params', type=float,nargs=3,help='superiorization parameters gamma, N, alpha_init')

    parser.add_argument('--nsubs',dest='ns',type=int,default=1,help='number of subsets. must divide evenly into number of angles')
    args = parser.parse_args()

    rec = Reconstructor(args.infile, args.outfile, args.dx, args.volume_size, [args.volume_size, args.volume_size], args.volume_spacing, args.angular_range,
                        np.radians(args.angular_range), args.detector_shape, args.fan_angle, args.detector_spacing, args.project_number, 
                        args.geom, args.source_detector_distance, args.source_isocenter_distance, args.learning_rateArg, 
                        args.num_epochsArg, args.strategy_typeArg, args.which_sinosArg, args.num_its, args.beta,args.x0_file, args.xtrue_file, args.sup_params,args.ns)
    rec.reconstruct()