"""
Created on Sun Nov 10 16:25:58 2019

@author: Ben Boys
"""

import sys

sys.path.insert(1, '../../peridynamics')
sys.path.insert(1, '../../peridynamics/fem')
sys.path.insert(1, '../../peridynamics/kernels')
sys.path.insert(1, '../../peridynamics/post_processing')

from OpenCLPeriVectorized import SeqModel as MODEL
import numpy as np
import vtk as vtk
import time
import pyopencl as cl


class simpleSquare(MODEL):
    # A User defined class for a particular problem which defines all necessary
    # parameters
    def __init__(self):
        # verbose
        self.v = True
        # TODO remove dim
        self.dim = 3

        self.meshFileName = '3300beam.msh'
        self.networkFileName = 'Network.vtk'

        self.meshType = 2
        self.boundaryType = 1
        self.numBoundaryNodes = 2
        self.numMeshNodes = 3

        # Concrete Parameters from classical material model
        self.PD_HORIZON = 0.138 # TODO: check this value
        
        self.PD_POISSON_CONCRETE = 0.2
        self.PD_YOUNGS_CONCRETE = 1.*22e9
        self.PD_G_CONCRETE = 8.8e9
        self.PD_DENSITY_CONCRETE = 2400.0
        self.PD_CRIT_TSTRAIN_CONCRETE = 0.000533 # What is this value for?
        
        self.PD_K_CONCRETE = np.double(
                self.PD_YOUNGS_CONCRETE / ((1 - 2* self.PD_POISSON_CONCRETE)*(1 + self.PD_POISSON_CONCRETE)))
        self.PD_E_CONCRETE = np.double(
            (12.00 * self.PD_K_CONCRETE) / (np.pi * np.power(self.PD_HORIZON, 4))) # TODO check that should be 12 and not 18, and check that delta is actually the material horizon
        
        self.PD_S0_CONCRETE = 0.000533 # check this value
        
        self.PD_POISSON_STEEL = 0.3
        self.PD_YOUNGS_STEEL = 1.*210e9
        self.PD_G_STEEL = 78e9
        self.PD_DENSITY_STEEL = 8000.0
        self.PD_CRIT_TSTRAIN_STEEL = 0.01 # What is this value for?
        
        self.PD_K_STEEL = np.double(
                self.PD_YOUNGS_STEEL / ((1 - 2* self.PD_POISSON_STEEL)*(1 + self.PD_POISSON_STEEL)))
        self.PD_E_STEEL = np.double(
            (12.00 * self.PD_K_STEEL) / (np.pi * np.power(self.PD_HORIZON, 4))) # check that delta is actually the material horizon
        
        self.PD_S0_STEEL = 0.01 # check this value
        
        # TODO check that horizon is the same for both materials

        # User input parameters
        self.loadRate = np.double(0.00001)
        self.crackLength = np.double(0.3)
        self.dt = np.double(1e-3)
        self.max_reaction = 0.1
        self.volume_total = 3.0 * 0.6 * 0.25
        self.load_scale_rate = 0.000001

        # These parameters will eventually be passed to model via command line
        # arguments
        self.read_mesh(self.meshFileName)

        # No. coordinate dimensions
        self.DPN = np.intc(3)
        self.PD_DPN_NODE_NO = np.intc(self.DPN * self.nnodes)

        st = time.time()
        
        # If the network has already been written to file, then read, if not, setNetwork
        try:
            self.read_network(self.networkFileName)
        except:
            self.setNetwork(self.PD_HORIZON)
            
        print(
            "Building horizons took {} seconds. Horizon length: {}".format(
                (time.time() - st), self.MAX_HORIZON_LENGTH))
        # self.setH() # Will further optimise the code, TODO
        self.setVolume()
        

        self.bctypes = np.zeros((self.nnodes, self.DPN), dtype=np.intc)
        self.bcvalues = np.zeros((self.nnodes, self.DPN), dtype=np.float64)

        # Find the boundary nodes and apply the displacement values
        for i in range(0, self.nnodes):
            bnd = self.findDisplacementBoundary(self.coords[i][:])
            self.bctypes[i, 0] = np.intc((bnd))
            self.bctypes[i, 1] = np.intc((bnd))
            self.bctypes[i, 2] = np.intc((bnd))
            self.bcvalues[i, 0] = np.float64(bnd * 0.5 * self.loadRate)
        
        self.force_bctypes = np.zeros((self.nnodes, self.DPN), dtype=np.intc)
        self.force_bcvalues = np.zeros((self.nnodes, self.DPN), dtype=np.float64)
        
        # Find the force boundary nodes and find amount of boundary nodes
        num_force_bc_nodes = 0
        for i in range(0, self.nnodes):
            bnd = self.findForceBoundary(self.coords[i][:])
            if bnd == -1:
                num_force_bc_nodes += 1
            elif bnd == 1:
                num_force_bc_nodes += 1
            self.force_bctypes[i, 0] = np.intc((bnd))
            self.force_bctypes[i, 1] = np.intc((bnd))
            self.force_bctypes[i, 2] = np.intc((bnd))
            
        self.num_force_bc_nodes = num_force_bc_nodes
        
        # Calculate initial forces
        self.force_bcvalues = np.zeros((self.nnodes, self.DPN), dtype=np.float64)
        load_scale = 0.0
        for i in range(0, self.nnodes):
            bnd = self.findForceBoundary(self.coords[i][:])
            if bnd == 1:
                pass
            elif bnd == -1:
                self.force_bcvalues[i, 2] = np.float64(1.* bnd * self.max_reaction * load_scale / (self.num_force_bc_nodes * self.volume_total))
        
        print("number of boundary nodes", num_force_bc_nodes)
        print("total volume", self.total_volume)
        print("volume total", self.volume_total)
                       
    def findDisplacementBoundary(self, x):
        # Function which marks constrain particles
        # 2 == NO BOUNDARY CONDITION (the number here is an arbitrary choice)
        # -1 == DISPLACEMENT LOADED IN -ve direction
        #  1 == DISPLACEMENT LOADED IN +ve direction
        #  0 == FIXED (DIRICHLET) BOUNDARY
        if self.meshFileName == 'test.msh' or 'debug3D.msh' or 'debug3D2.msh':
            # Does not live on a boundary
            bnd = 2
            # Does live on boundary
            if x[0] < 1.5 * self.PD_HORIZON:
                bnd = 0

            #elif x[0] > 3.0 - 1.5 * self.PD_HORIZON:
                #bnd = 1
        elif self.meshFileName == '3300beam.msh':
            # Does not live on a bondary
            bnd = 2
            # Does live on boundary
            if x[0] < 1.5 * self.PD_HORIZON:
                bnd = 0           
        return bnd
    
    def findForceBoundary(self, x):
        # Function which marks body force loaded particles
        
        # 2 == NO BOUNDARY CONDITION (the number here is an arbitrary choice)
        # -1 == FORCE LOADED IN -ve direction
        #  1 == FORCE LOADED IN +ve direction
        
        if self.meshFileName == 'test.msh' or 'debug3D.msh' or 'debug3D2.msh':
            # Does not live on a bondary
            bnd = 2
            if x[0] > 1.0 - 1.5 * self.PD_HORIZON:
                bnd = 1
        elif self.meshFileName == '3300beam.msh':
            # Does not live on a bondary
            bnd = -1
# =============================================================================
#             delta = 1e-6
#             if x[2] > 0.6 - delta:
#                 bnd = -1
# =============================================================================
        return bnd

    def isRebar(self, p):
        """ Function to determine whether the node coordinate is rebar
        """
        p = p[1:] # y and z coordinates for this node
        bar_centers = [
            # Compressive bars
            np.array((0.031, 0.031)),
            np.array((0.219, 0.031)),

            # Tensile bars
            np.array((0.03825, 0.569)),
            np.array((0.21175, 0.569))]

        rad_c = 0.006
        rad_t = 0.01325

        radii = [
            rad_c,
            rad_c,
            rad_t,
            rad_t]

        costs = [ np.sum(np.square(cent - p) - (np.square(rad))) for cent, rad in zip(bar_centers, radii) ]
        if any( c <= 0 for c in costs ):
            return True
        else:
            return False
        
        
        
    def bond_type(self, x, y):
        output = 0 # default to concrete
        bool1 = self.isRebar(x)
        bool2 = self.isRebar(y)
        if bool1 and bool2:
            output = 'steel'
        elif bool1 != bool2:
            output = 'interface'
        else:
            output = 'concrete'
        return output
        
    def isCrack(self, x, y):
        output = 0
        p1 = x
        p2 = y
        if x[0] > y[0]:
            p2 = x
            p1 = y
        # 1e-6 makes it fall one side of central line of particles
        if p1[0] < 0.5 + 1e-6 and p2[0] > 0.5 + 1e-6:
            # draw a straight line between them
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            c = p1[1] - m * p1[0]
            # height a x = 0.5
            height = m * 0.5 + c
            if (height > 0.5 * (1 - self.crackLength)
                    and height < 0.5 * (1 + self.crackLength)):
                output = 1
        return output


def output_device_info(device_id):
    sys.stdout.write("Device is ")
    sys.stdout.write(device_id.name)
    if device_id.type == cl.device_type.GPU:
        sys.stdout.write("GPU from ")
    elif device_id.type == cl.device_type.CPU:
        sys.stdout.write("CPU from ")
    else:
        sys.stdout.write("non CPU of GPU processor from ")
    sys.stdout.write(device_id.vendor)
    sys.stdout.write(" with a max of ")
    sys.stdout.write(str(device_id.max_compute_units))
    sys.stdout.write(" compute units\n")
    sys.stdout.flush()


def multivar_normal(L, num_nodes):
    """
    Fn for taking a single multivar normal sample covariance matrix with
    Cholesky factor, L
    """
    zeta = np.random.normal(0, 1, size=num_nodes)
    zeta = np.transpose(zeta)

    # vector
    w_tild = np.dot(L, zeta)

    return w_tild


def noise(L, samples, num_nodes):
    """
    Takes multiple samples from multivariate normal distribution with
    covariance matrix whith Cholesky factor, L
    """
    noise = []
    for i in range(samples):
        noise.append(multivar_normal(L, num_nodes))

    return np.transpose(noise)


def sim(sample, myModel, numSteps=1000, numSamples=1, print_every=1):
    
    print("Peridynamic Simulation -- Starting")
    
    # Initializing OpenCL
    context = cl.create_some_context()
    queue = cl.CommandQueue(context)

    # Print out device info
    output_device_info(context.devices[0])

    # Build the OpenCL program from file
    kernelsource = open("opencl_peridynamics.cl").read()

    SEP = " "

    options_string = (
        "-cl-fast-relaxed-math" + SEP
        + "-DPD_DPN_NODE_NO=" + str(myModel.PD_DPN_NODE_NO) + SEP
        + "-DPD_NODE_NO=" + str(myModel.nnodes) + SEP
        + "-DMAX_HORIZON_LENGTH=" + str(myModel.MAX_HORIZON_LENGTH) + SEP
        + "-DPD_DT=" + str(myModel.dt) + SEP)

    program = cl.Program(context, kernelsource).build([options_string])
    cl_kernel_initial_values = program.InitialValues
    cl_kernel_time_marching_1 = program.TimeMarching1
    cl_kernel_time_marching_2 = program.TimeMarching2
    cl_kernel_time_marching_3 = program.TimeMarching3
    cl_kernel_check_bonds = program.CheckBonds
    cl_kernel_calculate_damage = program.CalculateDamage

    # Set initial values in host memory

    # horizons and horizons lengths
    h_horizons = myModel.horizons
    h_horizons_lengths = myModel.horizons_lengths
    print(h_horizons_lengths)
    print(h_horizons)
    print("shape horizons lengths", h_horizons_lengths.shape)
    print("shape horizons lengths", h_horizons.shape)
    print(h_horizons_lengths.dtype, "dtype")

    # Nodal coordinates
    h_coords = np.ascontiguousarray(myModel.coords, dtype=np.float64)
    
    # Displacement boundary conditions types and delta values
    h_bctypes = myModel.bctypes
    h_bcvalues = myModel.bcvalues
    
    # Force boundary conditions types and values
    h_force_bctypes = myModel.force_bctypes
    h_force_bcvalues = myModel.force_bcvalues

    # Nodal volumes
    h_vols = myModel.V
    
    # Bond stiffnesses
    h_bond_stiffness =  np.ascontiguousarray(myModel.bond_stiffness, dtype=np.float64)
    h_bond_critical_stretch = np.ascontiguousarray(myModel.bond_critical_stretch, dtype=np.float64)

    # Displacements
    h_un = np.empty((myModel.nnodes, myModel.DPN), dtype=np.float64)
    h_un1 = np.empty((myModel.nnodes, myModel.DPN), dtype=np.float64)

    # Forces
    h_udn = np.empty((myModel.nnodes, myModel.DPN), dtype=np.float64)
    h_udn1 = np.empty((myModel.nnodes, myModel.DPN), dtype=np.float64)

    # Damage vector
    h_damage = np.empty(myModel.nnodes).astype(np.float32)

    # Print the dtypes
    print("horizons", h_horizons.dtype)
    print("horizons_length", h_horizons_lengths.dtype)
    print("force_bctypes", h_bctypes.dtype)
    print("force_bcvalues", h_bcvalues.dtype)
    print("bctypes", h_bctypes.dtype)
    print("bcvalues", h_bcvalues.dtype)
    print("coords", h_coords.dtype)
    print("vols", h_vols.dtype)
    print("un", h_un.dtype)
    print("un1", h_un1.dtype)
    print("udn", h_udn.dtype)
    print("udn1", h_udn1.dtype)
    print("damage", h_damage.dtype)

    # Build OpenCL data structures

    # Read only
    d_coords = cl.Buffer(context,
                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                         hostbuf=h_coords)
    d_bctypes = cl.Buffer(context,
                          cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=h_bctypes)
    d_bcvalues = cl.Buffer(context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=h_bcvalues)
    d_force_bctypes = cl.Buffer(context,
                          cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=h_force_bctypes)
    d_force_bcvalues = cl.Buffer(context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=h_force_bcvalues)
    d_vols = cl.Buffer(context,
                       cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                       hostbuf=h_vols)
    d_bond_stiffness = cl.Buffer(context,
                       cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                       hostbuf=h_bond_stiffness)
    d_bond_critical_stretch = cl.Buffer(context,
                       cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                       hostbuf=h_bond_critical_stretch)
    d_horizons_lengths = cl.Buffer(
            context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=h_horizons_lengths)

    # Read and write
    d_horizons = cl.Buffer(
            context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=h_horizons)
    d_un = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_un.nbytes)
    d_udn = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_udn.nbytes)
    d_un1 = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_un1.nbytes)
    d_udn1 = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_udn1.nbytes)

    # Write only
    d_damage = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_damage.nbytes)
    # Initialize kernel parameters
    cl_kernel_initial_values.set_scalar_arg_dtypes([None, None])
    cl_kernel_time_marching_1.set_scalar_arg_dtypes(
        [None, None, None, None, None, None])
    cl_kernel_time_marching_2.set_scalar_arg_dtypes(
        [None, None, None, None, None, None, None, None])
    cl_kernel_time_marching_3.set_scalar_arg_dtypes([None, None, None, None])
    cl_kernel_check_bonds.set_scalar_arg_dtypes([None, None, None, None])
    cl_kernel_calculate_damage.set_scalar_arg_dtypes([None, None, None])

    global_size = int(myModel.DPN * myModel.nnodes)
    cl_kernel_initial_values(queue, (global_size,), None, d_un, d_udn)
    for t in range(1, numSteps):

        st = time.time()

        # Time marching Part 1
        cl_kernel_time_marching_1(queue, (myModel.DPN * myModel.nnodes,),
                                  None, d_udn1, d_un1, d_un, d_udn, d_bctypes,
                                  d_bcvalues)

        # Time marching Part 2
        cl_kernel_time_marching_2(queue, (myModel.nnodes,), None, d_udn1,
                                  d_un1, d_vols, d_horizons, d_coords, d_bond_stiffness, d_force_bctypes, d_force_bcvalues)

        # Time marching Part 3
        # cl_kernel_time_marching_3(queue, (myModel.DPN * myModel.nnodes,),
        #                           None, d_un, d_udn, d_un1, d_udn1)

        # Check for broken bonds
        cl_kernel_check_bonds(queue,
                              (myModel.nnodes, myModel.MAX_HORIZON_LENGTH),
                              None, d_horizons, d_un1, d_coords, d_bond_critical_stretch)

        if t % print_every == 0:
            cl_kernel_calculate_damage(queue, (myModel.nnodes,), None,
                                       d_damage, d_horizons,
                                       d_horizons_lengths)
            cl.enqueue_copy(queue, h_damage, d_damage)
            cl.enqueue_copy(queue, h_un1, d_un1)
            print("Sum of all damage is", np.sum(h_damage))
            vtk.write("output/U_"+"t"+str(t)+".vtk", "Solution time step = "+str(t),
                      myModel.coords, h_damage, h_un1)
        
        load_scale = min(1.0, myModel.load_scale_rate * t)
        temp_value = -1. * myModel.max_reaction * load_scale / (myModel.num_force_bc_nodes * myModel.volume_total)
        # update the host force_bcvalues
        h_force_bcvalues = temp_value * np.ones((myModel.nnodes, myModel.DPN), dtype=np.float64)

# =============================================================================
#         for i in range(0, myModel.nnodes):
#             bnd = myModel.findForceBoundary(myModel.coords[i][:])
#             if bnd == -1:
#                 h_force_bcvalues[i, 2] = np.float64(temp_value)
# =============================================================================
        
        # update the GPU force_bcvalues
        d_force_bcvalues = cl.Buffer(context,
                           cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                           hostbuf=h_force_bcvalues)

        print('Timestep {} complete in {} s '.format(t, time.time() - st))

    # final time step
    cl_kernel_calculate_damage(queue, (myModel.nnodes,), None, d_damage,
                               d_horizons, d_horizons_lengths)
    cl.enqueue_copy(queue, h_damage, d_damage)
    cl.enqueue_copy(queue, h_un, d_un)
    vtk.write("output/U_"+"t"+str(t)+".vtk", "Solution time step = "+str(t),
              myModel.coords, h_damage, h_un)
    return vtk.write("output/U_"+"sample"+str(sample)+".vtk",
                     "Solution time step = "+str(t), myModel.coords, h_damage,
                     h_un)


def main():
    """
    Peridynamics Example of a 2D plate under displacement loading, with an
    pre-existing crack defect.
    """

    st = time.time()
    thisModel = simpleSquare()
    no_samples = 1
    
    for s in range(no_samples):
        sim(s, thisModel)
    print('TOTAL TIME REQUIRED {}'.format(time.time() - st))


if __name__ == "__main__":
    main()
