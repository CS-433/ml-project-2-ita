
import numpy as np
import os
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.spatial.distance import cdist
import meshio


DATASET_PATH=r'/Users/enzos/Desktop/ml-project-2-ita-temp/dataset'



"""
Functions provided by: Riccardo Tenderini (SCI SB SD Group, EPFL)
"""

def load_params():
    basis_space_pressure = np.load(os.path.join(DATASET_PATH, 'basis', 'pressure', 'space_basis.npy'))  # spatial basis
    basis_space = np.load(os.path.join(DATASET_PATH, 'basis', 'velocity', 'space_basis.npy'))  # spatial basis
    Nh_space, nmodes_space_full = basis_space.shape  # number of FOM and ROM unknowns in space
    basis_time = np.load(os.path.join(DATASET_PATH, 'basis', 'velocity', 'time_basis.npy'))  # temporal basis
    Nh_time, nmodes_time_full = basis_time.shape  # number of FOM and ROM unknowns in time
    nmodes_full = nmodes_space_full * nmodes_time_full  # total dimension of the reduced basis
    
    # UPDATE VELOCITY BASES TO ACCOUNT FOR SUPREMIZERS AND STABILIZERS
    N_supr_space = basis_space_pressure.shape[1] + 66  # number of extra bases in space for the velocity
    N_supr_time = 5  # number of extra bases in time for the velocity
    
    # UPDATE THE NUMBER OF VELOCITY MODES TO ACCOUNT FOR SUPREMIZERS AND STABILIZERS
    nmodes_space = nmodes_space_full - N_supr_space
    nmodes_time = nmodes_time_full - N_supr_time
    nmodes= nmodes_space * nmodes_time
    
    # UPDATE VELOCITY BASES TO ACCOUNT FOR SUPREMIZERS AND STABILIZERS
    basis_space = basis_space[:, :nmodes_space]
    basis_time = basis_time[:, :nmodes_time]

    #Load solutions (ouput)
    
    _sol = np.load(os.path.join(DATASET_PATH, 'RB_data', 'solutions.npy'))
    
    # velocity reduced solutions (with and without supremizers and stabilizers)
    solutions_full = np.reshape(_sol[:, :nmodes_full],
                                            (-1, nmodes_space_full, nmodes_time_full))
    solutions = solutions_full[:, :nmodes_space, :nmodes_time,]
    
    #Load parameters (input)
    params = np.load(os.path.join(DATASET_PATH, 'RB_data', 'parameters.npy'))
    params = np.delete(params, 2, axis=1)

    return params, solutions, basis_space, basis_time, Nh_space, Nh_time



def project(sol, normed_basis_space, basis_time):
    """ Project a full-order solution in space-time."""
    return (normed_basis_space.T.dot(sol)).dot(basis_time) # !! REMARK: here we need the normed basis in space !!

def expand(sol, basis_space, basis_time):
    """ Expand a reduced-order solution in space-time."""
    return (basis_space.dot(sol)).dot(basis_time.T)

#Useful functions to create data to visualize the fields in 3D (e.g. ParaView)

def read_vtk(filename):
    """Read .vtk file and return the polydata"""

    fn_dir, fn_ext = os.path.splitext(filename)

    if fn_ext == '.vtk':
        reader = vtk.vtkPolyDataReader()
    elif fn_ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif fn_ext == '.stl':
        reader = vtk.vtkSTLReader()
    elif fn_ext == '.obj':
        reader = vtk.vtkOBJReader()
    elif fn_ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif fn_ext == '.pvtu':
        reader = vtk.vtkXMLPUnstructuredGridReader()
    else:
        raise ValueError(F"File extension {fn_ext} not supported")

    reader.SetFileName(filename)
    reader.Update(0)
    mesh = reader.GetOutput()

    return mesh

def write_vtk(mesh, fn):
    """ Write a mesh (vtk polydata or unstructured grid) to disk """

    _, extension = os.path.splitext(fn)

    if extension == '.vtk':
        writer = vtk.vtkPolyDataWriter()
    elif extension == '.stl':
        writer = vtk.vtkSTLWriter()
    elif extension == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif extension == '.vtu':
        writer = vtk.vtkXMLUnstructuredGridWriter()
    elif extension == '.obj':
        writer = vtk.vtkOBJWriter()
    else:
        raise ValueError(f"Unrecognized extension {extension}")

    writer.SetInputData(mesh)
    writer.SetFileName(fn)
    writer.Update(0)
    writer.Write()

    return

def add_array(mesh, array, name):
    """Add numpy array as new field to a vtk file"""

    new_array = numpy_to_vtk(array)
    new_array.SetName(name)
    mesh.GetPointData().AddArray(new_array)

    return mesh

def compute_matching_idxs():
    """Compute correspondences bewteen indices on the .vtu and on the .mesh file for plotting"""

    mesh = read_vtk(os.path.join(DATASET_PATH, 'geometries', 'bif_sym_alpha50_h0.10_ref.vtu'))
    points = vtk_to_numpy(mesh.GetPoints().GetData())

    mesh_2 = meshio.read(os.path.join(DATASET_PATH, 'geometries','bif_sym_alpha50_h0.10.mesh'))
    points_2 = mesh_2.points

    dist = cdist(mesh_2.points, points, metric='euclidean')

    idxs = np.argmin(dist, axis=0)

    return idxs


def visualize_solution(field_array, basis_space, basis_time, Nh_space, Nh_time, step_t=10):

    """ Export the solution corresponding to the n-th snapshot (every step_t steps) to a .vtu file."""

    os.makedirs('solutions', exist_ok=True)

    idxs = compute_matching_idxs()

    mesh = read_vtk(os.path.join(DATASET_PATH, 'geometries', 'bif_sym_alpha50_h0.10.vtu'))

    cur_idxs = np.hstack([idxs + k * (Nh_space/3) for k in range(3)])
    fom_solution = expand(field_array, basis_space, basis_time)[cur_idxs]

    for cnt_t in range(0, Nh_time, step_t):
        cur_fom_solution = np.reshape(fom_solution[:, cnt_t], (3, -1)).T
        mesh = add_array(mesh, cur_fom_solution, 'velocity')

        write_vtk(mesh, os.path.join('solutions', f"solution_{cnt_t}" + '.vtu'))

    return


def normalize(tensor, min, max, margin=0):
    """
         Min-Max normalization function

          Args:
              tensor: torch tensor of shape (*,N,M)
              min: min tensor of shape (N,M)
              max: max tensor of shape (N,M)
              margin: scalar to compress data in a subinterval of [0,1] (to leave a margin for outliers)
          """
    device = tensor.device
    tensor_normalized = (tensor - min.to(device)*(1-margin)) / (max.to(device)*(1+margin) - min.to(device)*(1-margin))
    return tensor_normalized


def denormalize(tensor_normalized, min, max, margin=0):
    """
         Min-Max denormalization function
         
          Args:
              tensor_normalized: torch tensor to denormalize of shape (*,N,M)
              min: min tensor of shape (N,M)
              max: max tensor of shape (N,M)
              margin: scalar used in the normalization to compress data in a subinterval of [0,1] (to leave a margin for outliers)
          """
    device = tensor_normalized.device
    tensor = tensor_normalized * ((1+margin) * max.to(device) - (1-margin) * min.to(device)) + (1-margin) * min.to(device)
    return tensor


def build_k_indices(N, k_fold):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    interval = int(N / k_fold) #number of data per fold-set
    indices = np.random.permutation(N) #permutation of array from 0 to num_row
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

