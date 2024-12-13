
import numpy as np
import os

#functions provided by the professor to load data

def load_params(DATASET_PATH):
    
    fields = {'velocity', 'pressure'}

    basis_space, sv_space, Nh_space, nmodes_space = dict(), dict(), dict(), dict()
    basis_time, sv_time, Nh_time, nmodes_time = dict(), dict(), dict(), dict()
    nmodes = dict()
    for field in fields:
        basis_space[field] = np.load(os.path.join(DATASET_PATH, 'basis', field, 'space_basis.npy'))  # spatial basis
        sv_space[field] = np.load(os.path.join(DATASET_PATH, 'basis', field, 'space_sv.npy'))  # singular values in space
        Nh_space[field], nmodes_space[field] = basis_space[field].shape  # number of FOM and ROM unknowns in space
        basis_time[field] = np.load(os.path.join(DATASET_PATH, 'basis', field, 'time_basis.npy'))  # temporal basis
        sv_time[field] = np.load(os.path.join(DATASET_PATH, 'basis', field, 'time_sv.npy'))  # singular values in time
        Nh_time[field], nmodes_time[field] = basis_time[field].shape  # number of FOM and ROM unknowns in time
        nmodes[field] = nmodes_space[field] * nmodes_time[field]  # total dimension of the reduced basis
    
    # UPDATE VELOCITY BASES TO ACCOUNT FOR SUPREMIZERS AND STABILIZERS
    N_supr_space = basis_space['pressure'].shape[1] + 66  # number of extra bases in space for the velocity
    N_supr_time = 5  # number of extra bases in time for the velocity
    
    # STORE ORIGINAL NUMBER OF VELOCITY MODES IN THE DICTIONARY
    nmodes_space['velocity_full'] = nmodes_space['velocity']
    nmodes_time['velocity_full'] = nmodes_time['velocity']
    nmodes['velocity_full'] = nmodes['velocity']
    
    # UPDATE THE NUMBER OF VELOCITY MODES TO ACCOUNT FOR SUPREMIZERS AND STABILIZERS
    nmodes_space['velocity'] -= N_supr_space
    nmodes_time['velocity'] -= N_supr_time
    nmodes['velocity'] = nmodes_space['velocity'] * nmodes_time['velocity']
    
    # UPDATE VELOCITY BASES TO ACCOUNT FOR SUPREMIZERS AND STABILIZERS
    basis_space['velocity'] = basis_space['velocity'][:, :nmodes_space['velocity']]
    basis_time['velocity'] = basis_time['velocity'][:, :nmodes_time['velocity']]
    
    # LOAD NORMED BASIS MATRICES IN SPACE (needed for projections)
    basis_space_normed = dict()
    #norm = dict()
    for field in fields:
        basis_space_normed[field] = np.load(os.path.join(DATASET_PATH, 'basis', field, 'basis_space_normed.npy'))
    #Load solutions (ouput)
    
    _sol = np.load(os.path.join(DATASET_PATH, 'RB_data', 'solutions.npy'))
    
    solutions = dict()
    
    # velocity reduced solutions (with and without supremizers and stabilizers)
    solutions['velocity_full'] = np.reshape(_sol[:, :nmodes['velocity_full']],
                                            (-1, nmodes_space['velocity_full'], nmodes_time['velocity_full']))
    solutions['velocity'] = solutions['velocity_full'][:, :nmodes_space['velocity'], :nmodes_time['velocity']]
    
    # pressure reduced solutions
    solutions['pressure'] = np.reshape(_sol[:, :nmodes['pressure']],
                                       (-1, nmodes_space['pressure'], nmodes_time['pressure']))
    
    #Load parameters (input)
    params = np.load(os.path.join(DATASET_PATH, 'RB_data', 'parameters.npy'))
    params = np.delete(params, 2, axis=1)
    return params, solutions['velocity']



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


def visualize_solution(field_array, fields=None, step_t=10):
    """ Export the solution corresponding to the n-th snapshot (every step_t steps) to a .vtu file."""

    if fields is None:
        fields = {'velocity': 3, 'pressure': 1}  # fields and corresponding dimensions

    os.makedirs('solutions', exist_ok=True)

    idxs = compute_matching_idxs()

    mesh = read_vtk(os.path.join(DATASET_PATH, 'geometries', 'bif_sym_alpha50_h0.10.vtu'))

    fom_solution = dict()
    for field in fields:
        # print(f"Processing field {field} - Dimension: {fields[field]}")
        cur_idxs = np.hstack([idxs + k * (Nh_space[field]//fields[field]) for k in range(fields[field])])
        fom_solution[field] = expand(field_array, basis_space[field], basis_time[field])[cur_idxs]

    for cnt_t in range(0, Nh_time['velocity'], step_t):
        # print(f"\nProcessing timestep {cnt_t} of {Nh_time['velocity']}")
        for field in fields:
            cur_fom_solution = np.reshape(fom_solution[field][:, cnt_t], (fields[field], -1)).T
            mesh = add_array(mesh, cur_fom_solution, field)

        # write_vtk(mesh, os.path.join('solutions', f"solution_{n}_{cnt_t}" + '.vtu'))
        write_vtk(mesh, os.path.join('solutions', f"solution_{cnt_t}" + '.vtu'))

    return
