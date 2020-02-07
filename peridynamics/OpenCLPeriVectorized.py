import numpy as np
import periFunctions as func
import meshio
import vtk as vtk

class SeqModel:
    def __init__(self):
        # self.nnodes defined when instance of readMesh called, cannot
        # initialise any other matrix until we know the nnodes
        self.v = True
        self.dim = 3

        self.meshFileName = "3300beam.msh"

        self.meshType = 2
        self.boundaryType = 1
        self.numBoundaryNodes = 2
        self.numMeshNodes = 3

        if self.dim == 3:
            self.meshType = 4
            self.boundaryType = 2
            self.numBoundaryNodes = 3
            self.numMeshNodes = 4
    
    def read_network(self, network_file):
        """ For reading a network file if it has been written to file yet. Quicker than building horizons from scratch."""
        
        f = open(network_file, "r")
        
        if f.mode == "r":
            iline = 0
            
            # Read the Max horizons length first
            find_MHL = 0
            while (find_MHL == 0):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = row.split()
                
                find_MHL = 1 if 'MAX_HORIZON_LENGTH' in rowAsList else 0
            
            
            MAX_HORIZON_LENGTH = int(rowAsList[1])
            
            # Now read nnodes
            find_nnodes = 0
            while (find_nnodes == 0):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = row.split()
                
                find_nnodes = 1 if 'NNODES' in rowAsList else 0
            
            nnodes = int(rowAsList[1])

            # Now read horizons lengths
            find_horizons_lengths = 0
            while (find_horizons_lengths == 0):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = row.split()
                
                find_horizons_lengths = 1 if 'HORIZONS_LENGTHS' in rowAsList else 0
            
            horizons_lengths = np.zeros(nnodes, dtype=int)
            for i in range(0, nnodes):
                iline += 1
                line = f.readline()
                horizons_lengths[i] = np.intc(line.split())
                
            print('Building family matrix from file')
            # Now read family matrix
            find_family = 0
            while (find_family == 0):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = row.split()
                
                find_family = 1 if 'FAMILY' in rowAsList else 0
            
            family = []
            for i in range(nnodes):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = line.split()
                family.append(np.zeros(len(rowAsList), dtype=np.intc))
                for j in range(0, len(rowAsList)):
                    family[i][j] = np.intc(rowAsList[j])
            
            print('Finding stiffness values')
            # Now read stiffness values
            find_stiffness = 0
            while (find_stiffness == 0):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = row.split()
                
                find_stiffness = 1 if 'STIFFNESS' in rowAsList else 0
            print('Building stiffnesses from file')
            
            bond_stiffness_family = []
            for i in range(nnodes):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = line.split()
                bond_stiffness_family.append(np.zeros(len(rowAsList), dtype=np.float64))
                for j in range(0, len(rowAsList)):
                    bond_stiffness_family[i][j] = (rowAsList[j])
            
            print('Finding critical stretch values')
            # Now read critcal stretch values
            find_stretch = 0
            while (find_stretch == 0):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = row.split()
                
                find_stretch = 1 if 'STRETCH' in rowAsList else 0
            
            print('Building critical stretch values from file')
            bond_critical_stretch_family = []
            for i in range(nnodes):
                iline += 1
                line = f.readline()
                row = line.strip()
                rowAsList = line.split()
                bond_critical_stretch_family.append(np.zeros(len(rowAsList), dtype=np.float64))
                for j in range(0, len(rowAsList)):
                    bond_critical_stretch_family[i][j] = rowAsList[j]
            
            # Maximum number of nodes that any one of the nodes is connected to
            MAX_HORIZON_LENGTH_CHECK = np.intc(
                len(max(family, key=lambda x: len(x)))
                )
            
            assert MAX_HORIZON_LENGTH == MAX_HORIZON_LENGTH_CHECK, 'Read failed on MAX_HORIZON_LENGTH check'
            
            horizons = -1 * np.ones([nnodes, MAX_HORIZON_LENGTH])
            for i, j in enumerate(family):
                horizons[i][0:len(j)] = j
                
            bond_stiffness = -1. * np.ones([nnodes, MAX_HORIZON_LENGTH])
            for i, j in enumerate(bond_stiffness_family):
                bond_stiffness[i][0:len(j)] = j
            
            bond_critical_stretch = -1. * np.ones([nnodes, MAX_HORIZON_LENGTH])
            for i, j in enumerate(bond_critical_stretch_family):
                bond_critical_stretch[i][0:len(j)] = j

            # Make sure it is in a datatype that C can handle
            self.horizons = horizons.astype(np.intc)
            self.bond_stiffness = bond_stiffness
            self.bond_critical_stretch = bond_critical_stretch
            
            self.horizons_lengths = horizons_lengths
            self.family = family
            self.MAX_HORIZON_LENGTH = MAX_HORIZON_LENGTH
            self.nnodes = nnodes
            
            initiate_crack = 0
            # Initiate crack
            if initiate_crack == 1:
                for i in range(0, self.nnodes):
        
                    for k in range(0, self.MAX_HORIZON_LENGTH):
                        j = self.horizons[i][k]
                        if self.isCrack(self.coords[i, :], self.coords[j, :]):
                            self.horizons[i][k] = np.intc(-1)
            f.close()
            
    def read_mesh(self, mesh_file):
        mesh = meshio.read(mesh_file)
        if self.dim == 3:
            
            # Get coordinates, encoded as mesh points
            self.coords = mesh.points
            self.nnodes = self.coords.shape[0]
        
            # Get connectivity, mesh triangle cells
            self.connectivity = mesh.cells['tetra']
            self.nelem = self.connectivity.shape[0]
        
            # Get boundary connectivity, mesh lines
            self.connectivity_bnd = mesh.cells['triangle']
            self.nelem_bnd = self.connectivity_bnd.shape[0]
        if self.dim == 2:
            # Get coordinates, encoded as mesh points
            self.coords = mesh.points
            self.nnodes = self.coords.shape[0]
        
            # Get connectivity, mesh triangle cells
            self.connectivity = mesh.cells['triangle']
            self.nelem = self.connectivity.shape[0]
        
            # Get boundary connectivity, mesh lines
            self.connectivity_bnd = mesh.cells['line']
            self.nelem_bnd = self.connectivity_bnd.shape[0]
            
    def setVolume(self):
        V = np.zeros(self.nnodes, dtype=np.float64)
        total_volume = 0
        for ie in range(0, self.nelem):
            n = self.connectivity[ie]
            # Compute Area or Volume
            # Define area of element
            val = 1. / n.size
            
            if self.dim == 2:
                
                xi = self.coords[int(n[0])][0]
                yi = self.coords[int(n[0])][1]
                xj = self.coords[int(n[1])][0]
                yj = self.coords[int(n[1])][1]
                xk = self.coords[int(n[2])][0]
                yk = self.coords[int(n[2])][1]
                
                element_area = 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
                val *= element_area
                total_volume += element_area
                
            elif self.dim == 3:
    
                a = self.coords[int(n[0])]
                b = self.coords[int(n[1])]
                c = self.coords[int(n[2])]
                d = self.coords[int(n[3])]
                
                # Volume of a tetrahedron
                i = np.subtract(a,d)
                j = np.subtract(b,d)
                k = np.subtract(c,d)
                
                element_volume = (1./6) * np.absolute(np.dot(i, np.cross(j,k)))
                val*= element_volume
                total_volume += element_volume
            else:
                raise ValueError('dim', 'dimension size can only take values 2 or 3')
                
            for j in range(0, n.size):
                V[int(n[j])] += val
        self.V = V.astype(np.float64)
        self.total_volume = total_volume
        
        # assert that total_volume is the expected value, given mesh dimensions

    def setNetwork(self, horizon):
        """
        Sets the family matrix, and converts to horizons matrix. Calculates
        horizons_lengths
        """

        # Container for nodal family
        family = []
        bond_stiffness_family = []
        bond_critical_stretch_family = []

        # Container for number of nodes (including self) that each of the nodes
        # is connected to
        self.horizons_lengths = np.zeros(self.nnodes, dtype=int)

        for i in range(0, self.nnodes):
            print('node', i, 'networking...')
            tmp = []
            tmp2 = []
            tmp3 = []
            for j in range(0, self.nnodes):
                if i != j:
                    l2_sqr = func.l2_sqr(self.coords[i, :], self.coords[j, :])
                    if np.sqrt(l2_sqr) < horizon:
                        tmp.append(j)
                        # Determine the material properties for that bond
                        material_flag = self.bond_type(self.coords[i, :], self.coords[j, :])
                        if material_flag == 'steel':
                            tmp2.append(self.PD_E_STEEL)
                            tmp3.append(self.PD_S0_STEEL)
                        elif material_flag == 'interface':
                            tmp2.append(self.PD_E_CONCRETE) # choose the weakest stiffness of the two bond types
                            tmp3.append(self.PD_S0_CONCRETE * 3.0) # 3.0 is used for interface bonds in the literature
                        elif material_flag == 'concrete':
                            tmp2.append(self.PD_E_CONCRETE)
                            tmp3.append(self.PD_S0_CONCRETE)
            
            
            
            family.append(np.zeros(len(tmp), dtype=np.intc))
            bond_stiffness_family.append(np.zeros(len(tmp2), dtype=np.float64))
            bond_critical_stretch_family.append(np.zeros(len(tmp3), dtype=np.float64))
            
            
            self.horizons_lengths[i] = np.intc((len(tmp)))
            for j in range(0, len(tmp)):
                family[i][j] = np.intc(tmp[j])
                bond_stiffness_family[i][j] = np.float64(tmp2[j])
                bond_critical_stretch_family[i][j] = np.float64(tmp3[j])
            
        
        assert len(family) == self.nnodes
        # As numpy array
        self.family = np.array(family)
        
        # Do the bond critical ste
        self.bond_critical_stretch_family = np.array(bond_critical_stretch_family)
        self.bond_stiffness_family = np.array(bond_stiffness_family)
        
# =============================================================================
#         # Calculate stiffening factor - surface corrections for 3D problem, for this we need family matrix
#         for i in range(0, self.nnodes):
#             len(family[i]) = nnodes_i_family
#             nodei_family_volume = nnodes_i_family * self.PD_NODE_VOLUME_AVERAGE # Possible to calculate more exactly, we have the volumes for free
#             for j in range(len(family[i])):
#                 nnodes_j_family = len(family[j])
#                 nodej_family_volume = nnodes_j_family* self.PD_NODE_VOLUME_AVERAGE # Possible to calculate more exactly, we have the volumes for free
#                 
#                 stiffening_factor = 2.* self.PD_FAMILY_VOLUME /  (nodej_family_volume + nodei_family_volume)
#                 
#                 bond_stiffness_family[i][j] *= stiffening_factor
#         
# =============================================================================
        self.family_v = np.zeros(self.nnodes)
        for i in range(0, self.nnodes):
            tmp = 0 # tmp family volume
            family_list = family[i]
            for j in range(0, len(family_list)):
                tmp += self.V[family_list[j]]
            self.family_v[i] = tmp
            
        # Calculate stiffening factor nore accurately using actual nodal volumes
        for i in range(0, self.nnodes):
            family_list = family[i]
            nodei_family_volume = self.family_v[i] # Possible to calculate more exactly, we have the volumes for free
            for j in range(len(family_list)):
                nodej_family_volume = self.family_v[j]
                stiffening_factor = 2.* self.PD_FAMILY_VOLUME /  (nodej_family_volume + nodei_family_volume)
                print('Stiffening factor {}'.format(stiffening_factor))
                bond_stiffness_family[i][j] *= stiffening_factor
        
        # Maximum number of nodes that any one of the nodes is connected to
        self.MAX_HORIZON_LENGTH = np.intc(
            len(max(self.family, key=lambda x: len(x)))
            )

        horizons = -1 * np.ones([self.nnodes, self.MAX_HORIZON_LENGTH])
        for i, j in enumerate(self.family):
            horizons[i][0:len(j)] = j
            
        bond_stiffness = -1. * np.ones([self.nnodes, self.MAX_HORIZON_LENGTH])
        for i, j in enumerate(self.bond_stiffness_family):
            bond_stiffness[i][0:len(j)] = j
            
        bond_critical_stretch = -1. * np.ones([self.nnodes, self.MAX_HORIZON_LENGTH])
        for i, j in enumerate(self.bond_critical_stretch_family):
            bond_critical_stretch[i][0:len(j)] = j

        # Make sure it is in a datatype that C can handle
        self.horizons = horizons.astype(np.intc)
        self.bond_stiffness = bond_stiffness
        self.bond_critical_stretch = bond_critical_stretch
        
        # Initiate crack
        for i in range(0, self.nnodes):

            for k in range(0, self.MAX_HORIZON_LENGTH):
                j = self.horizons[i][k]
                if self.isCrack(self.coords[i, :], self.coords[j, :]):
                    self.horizons[i][k] = np.intc(-1)
        vtk.writeNetwork("Network"+".vtk", "Network",
                      self.MAX_HORIZON_LENGTH, self.horizons_lengths, self.family, self.bond_stiffness_family, self.bond_critical_stretch_family)