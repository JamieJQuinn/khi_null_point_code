import sys
import math
from time import asctime
import gc

import sdf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from plotting import latexify, save_plot
from plotting_parameters import *
from parameters import *

INCLUDE_CBARS=True
INCLUDE_TITLE=True
INCLUDE_AXIS_LABELS=True

def load_sdf_file(sdfFilename):
    return sdf.read(sdfFilename)

def get_dx_dy_dz(sdfFile, cell_centre=False):
    extents = sdfFile.Grid_Grid.extents
    dims = sdfFile.Grid_Grid.dims

    if cell_centre:
        dims = [dim - 1 for dim in dims]

    return [(extents[i+3] - extents[i])/dims[i] for i in range(3)]

def get_variable(sdfFile, variable_name):
    return getattr(sdfFile, variable_name)

def get_variable_data(sdfFile, variable_name):
    if variable_name == "magnitude_current_density":
        return get_magnitude_current(sdfFile)
    elif variable_name == "vorticity_density":
        return get_magnitude_vorticity(sdfFile)
    elif variable_name == "kinetic_energy":
        return calc_kinetic_energy(sdfFile)
    elif variable_name == "kinetic_energy_z":
        return calc_kinetic_energy_z(sdfFile)
    elif variable_name == "abs_Velocity_Vz":
        return np.abs(get_variable_data(sdfFile, "Velocity_Vz"))
    elif variable_name == "alfven_velocity":
        return calc_alfven_velocity(sdfFile)
    elif variable_name == "sound_speed":
        return calc_sound_speed(sdfFile)
    elif variable_name == "pressure":
        return calc_pressure(sdfFile)
    elif variable_name == "magnetic_pressure":
        return calc_magnetic_pressure(sdfFile)
    elif variable_name == "parallel_electric_field":
        return calc_parallel_electric_field(sdfFile)
    else:
        return get_variable(sdfFile, variable_name).data

def get_variable_slice(sdfFile, variable_name, slice_dim, slice_loc):
    if variable_name == "lorentz_force_x":
        return calc_lorentz_force_component_slice(sdfFile, "x", slice_dim, slice_loc)
    elif variable_name == "lorentz_force_y":
        return calc_lorentz_force_component_slice(sdfFile, "y", slice_dim, slice_loc)
    elif variable_name == "lorentz_force_z":
        return calc_lorentz_force_component_slice(sdfFile, "z", slice_dim, slice_loc)
    elif variable_name == "current_density_x":
        return calc_current_slice(sdfFile, slice_dim, slice_loc)[0]
    elif variable_name == "current_density_y":
        return calc_current_slice(sdfFile, slice_dim, slice_loc)[1]
    elif variable_name == "current_density_z":
        return calc_current_slice(sdfFile, slice_dim, slice_loc)[2]
    elif variable_name == "magnitude_current_density":
        return np.linalg.norm(
            calc_current_slice(sdfFile, slice_dim, slice_loc),
            axis=0)
    elif variable_name == "vorticity_density_x":
        return calc_vorticity_slice(sdfFile, slice_dim, slice_loc)[0]
    elif variable_name == "vorticity_density_y":
        return calc_vorticity_slice(sdfFile, slice_dim, slice_loc)[1]
    elif variable_name == "vorticity_density_z":
        return calc_vorticity_slice(sdfFile, slice_dim, slice_loc)[2]
    elif variable_name == "magnitude_vorticity_density":
        return np.linalg.norm(
            calc_vorticity_slice(sdfFile, slice_dim, slice_loc),
            axis=0)
    elif variable_name == "pressure_force_x":
        return calc_pressure_force_component_slice(sdfFile, "x", slice_dim, slice_loc)
    elif variable_name == "pressure_force_y":
        return calc_pressure_force_component_slice(sdfFile, "y", slice_dim, slice_loc)
    elif variable_name == "pressure_force_z":
        return calc_pressure_force_component_slice(sdfFile, "z", slice_dim, slice_loc)
    elif variable_name == "magnetic_tension_x":
        return calc_magnetic_tension_force_component_slice(sdfFile, "x", slice_dim, slice_loc)
    elif variable_name == "magnetic_tension_y":
        return calc_magnetic_tension_force_component_slice(sdfFile, "y", slice_dim, slice_loc)
    elif variable_name == "magnetic_tension_z":
        return calc_magnetic_tension_force_component_slice(sdfFile, "z", slice_dim, slice_loc)
    elif variable_name == "isotropic_viscous_force_x":
        return calc_isotropic_viscous_force_component_slice(sdfFile, "x", slice_dim, slice_loc)
    elif variable_name == "isotropic_viscous_force_y":
        return calc_isotropic_viscous_force_component_slice(sdfFile, "y", slice_dim, slice_loc)
    elif variable_name == "isotropic_viscous_force_z":
        return calc_isotropic_viscous_force_component_slice(sdfFile, "z", slice_dim, slice_loc)
    else:
        return get_slice(sdfFile, variable_name, slice_dim, slice_loc)

def calc_centred_velocity(var):
    x_av = var[1:,:,:] + var[:-1,:,:]
    y_av = x_av[:,1:,:] + x_av[:,:-1,:]
    z_av = y_av[:,:,1:] + y_av[:,:,:-1]
    return z_av / 8.0

def calc_centred_velocity_slice(var):
    x_av = var[1:,:] + var[:-1,:]
    y_av = x_av[:,1:] + x_av[:,:-1]
    return y_av / 4.0

def calc_pressure(sdfFile):
    rho = get_variable_data(sdfFile, "Fluid_Rho")
    energy = get_variable_data(sdfFile, "Fluid_Energy")
    pressure = rho*energy * (GAMMA - 1.0)
    return pressure

def calc_magnetic_pressure(sdfFile):
    bx = get_variable_data(sdfFile, "Magnetic_Field_bx_centred")
    by = get_variable_data(sdfFile, "Magnetic_Field_by_centred")
    bz = get_variable_data(sdfFile, "Magnetic_Field_bz_centred")

    return np.power(bx, 2) + np.power(by, 2) + np.power(bz, 2)

def calc_mag_velocity2(sdfFile):
    vx = get_variable_data(sdfFile, "Velocity_Vx")
    vy = get_variable_data(sdfFile, "Velocity_Vy")
    vz = get_variable_data(sdfFile, "Velocity_Vz")

    return np.power(vx, 2) + np.power(vy, 2) + np.power(vz, 2)

def calc_parallel_electric_field(sdfFile):
    bx = get_variable_data(sdfFile, "Magnetic_Field_bx_centred")
    by = get_variable_data(sdfFile, "Magnetic_Field_by_centred")
    bz = get_variable_data(sdfFile, "Magnetic_Field_bz_centred")

    dx, dy, dz = get_dx_dy_dz(sdfFile, cell_centre=True)

    mag_mag = np.sqrt(np.power(bx,2) + np.power(by,2) + np.power(bz,2))

    gradbx = np.gradient(bx, dx, dy, dz)
    gradby = np.gradient(by, dx, dy, dz)
    gradbz = np.gradient(bz, dx, dy, dz)

    return (bx*(gradbz[1] - gradby[2]) + by*(gradbz[0] - gradbx[2]) + bz*(gradby[0] - gradbx[1]))/mag_mag

def calc_sound_speed(sdfFile):
    # From Newton-Laplace formula
    # sound speed = sqrt(temperature)
    return np.sqrt(calc_temperature(sdfFile))

def calc_temperature(sdfFile):
    # Nondimensional temperature is just 1-gamma * internal energy
    return (GAMMA-1.0)*get_variable_data(sdfFile, "Fluid_Energy")

def calc_kinetic_energy(sdfFile):
    magV2 = calc_centred_velocity(
        calc_mag_velocity2(sdfFile)
    )

    rho = get_variable_data(sdfFile, "Fluid_Rho")

    return 0.5*rho*magV2

def calc_kinetic_energy_z(sdfFile):
    vz = calc_centred_velocity(
        get_variable_data(sdfFile, "Velocity_Vz")
    )

    rho = get_variable_data(sdfFile, "Fluid_Rho")

    return 0.5*rho*np.power(vz, 2)

def calc_mag(vector):
    return np.power(vector[0], 2) + np.power(vector[1], 2) + np.power(vector[2], 2)

def calc_alfven_velocity(sdfFile):
    B = calc_mag(get_magnetic_field(sdfFile))
    rho = get_variable_data(sdfFile, "Fluid_Rho")
    return B/np.sqrt(rho)

def get_magnetic_field(sdfFile):
    components = ['x', 'y', 'z']
    field = []
    for i, variable in enumerate(
        ['Magnetic_Field_Bx', 'Magnetic_Field_By', 'Magnetic_Field_Bz']
    ):
        b = get_variable_data(sdfFile, variable)
        b_centred = centre_field_component(b, components[i])
        del b
        field.append(b_centred)

    return field


def plot_slice(sdfFile, variable_name, dimension, slice_loc,
               cbar=INCLUDE_CBARS,
               include_title=INCLUDE_TITLE,
               include_axis_labels=INCLUDE_AXIS_LABELS,
               xlim=False, ylim=False, title=False, cmap="viridis",
              vmax = None, vmin = None):
    velocity = get_slice(sdfFile, variable_name, dimension, slice_loc)
    extents = get_slice_extents(sdfFile, dimension)

    latexify(columns=1)
    fig, axis = plt.subplots()

    im = axis.imshow(velocity.T,\
                vmax=vmax, vmin=vmin,\
                interpolation='bilinear',\
                cmap=plt.get_cmap(cmap),
               extent=extents, origin='lower')

    if xlim:
        axis.set_xlim(xlim)
    if ylim:
        axis.set_ylim(ylim)

    if include_title:
        if title:
            axis.title.set_text(" ".join(title.split("_")))
        else:
            axis.title.set_text(" ".join(variable_name.split("_")))

    if include_axis_labels:
        labels = get_axis_labels(dimension)
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])

    if cbar:
        attach_colorbar(axis, im)


def get_axis_labels(dimension):
    if type(dimension) is str:
        dimension = get_dimension_index(dimension)
    labels = ["x", "y", "z"]
    labels.pop(dimension)
    return labels


def get_slice_extents(sdfFile, dimension):
    if type(dimension) is str:
        dimension = get_dimension_index(dimension)

    extents = list(sdfFile.Grid_Grid.extents)
    extents.pop(dimension+3)
    extents.pop(dimension)
    extents[1], extents[2] = extents[2], extents[1]
    return extents

def get_extents(sdfFile):
    return list(sdfFile.Grid_Grid.extents)

def get_slice(sdfFile, variable_name, dimension, slice_loc):
    if type(dimension) is str:
        dimension = get_dimension_index(dimension)

    index = length_to_index(sdfFile, slice_loc, dimension)
    data = get_variable_data(sdfFile, variable_name)

    return np.take(data, index, axis=dimension)


def slice_variable(sdf_var, x_min=0, x_max=-1, y_min=0, y_max=-1, z_min=0, z_max=-1):
    if(x_max < 0):
        x_max = sdf_var.dims[0]
    if(y_max < 0):
        y_max = sdf_var.dims[1]
    if(z_max < 0):
        z_max = sdf_var.dims[2]
    
    return sdf_var.data[x_min:x_max, y_min:y_max, z_min:z_max]

def curl(bx, by, bz, dx, dy, dz):
    gradbx = np.gradient(bx, dx, dy, dz)
    gradby = np.gradient(by, dx, dy, dz)
    gradbz = np.gradient(bz, dx, dy, dz)

    return np.array([(gradbz[1] - gradby[2]), -(gradbz[0] - gradbx[2]), (gradby[0] - gradbx[1])])

def mag_curl(bx, by, bz, dx, dy, dz):
    gradbx = np.gradient(bx, dx, dy, dz)
    gradby = np.gradient(by, dx, dy, dz)
    gradbz = np.gradient(bz, dx, dy, dz)

    return np.sqrt(np.power(gradbz[1] - gradby[2], 2) + np.power(gradbz[0] - gradbx[2], 2) + np.power(gradby[0] - gradbx[1], 2))

def get_magnitude_vorticity(sdfFile):
    vx = get_variable_data(sdfFile, "Velocity_Vx")
    vy = get_variable_data(sdfFile, "Velocity_Vy")
    vz = get_variable_data(sdfFile, "Velocity_Vz")

    dx, dy, dz = get_dx_dy_dz(sdfFile)

    return mag_curl(vx, vy, vz, dx, dy, dz)

def centre_field_component(b, comp):
    if comp == "x":
        b = 0.5*(b[:-1,:,:] + b[1:,:,:])
    elif comp == "y":
        b = 0.5*(b[:,:-1,:] + b[:,1:,:])
    elif comp == "z":
        b = 0.5*(b[:,:,:-1] + b[:,:,1:])

    return b


def centre_magnetic_field(bx, by, bz):
    # This operates inplace
    bx_centred = centre_field_component(bx, "x")
    by_centred = centre_field_component(by, "y")
    bz_centred = centre_field_component(bz, "z")

    return bx_centred, by_centred, bz_centred

def get_magnitude_current(sdfFile):
    bx = get_variable_data(sdfFile, "Magnetic_Field_Bx")
    by = get_variable_data(sdfFile, "Magnetic_Field_By")
    bz = get_variable_data(sdfFile, "Magnetic_Field_Bz")

    bx, by, bz = centre_magnetic_field(bx, by, bz)

    dx, dy, dz = get_dx_dy_dz(sdfFile, cell_centre=True)

    return mag_curl(bx, by, bz, dx, dy, dz)

def calc_current_slice(sdfFile, slice_dim, slice_loc):
    slice_index = length_to_index(sdfFile, slice_loc, slice_dim)
    indices_around_slice = (slice_index-1, slice_index, slice_index+1)
    slice_dim = sanitise_dimension(slice_dim)

    print(asctime(), "Loading magnetic field")
    bx_raw = get_variable_data(sdfFile, "Magnetic_Field_Bx")
    centre_field_component(bx_raw, "x")
    bx = np.copy(bx_raw.take(indices_around_slice, axis=slice_dim))
    del bx_raw
    if slice_dim != 0:
        bx = bx[:-1]

    by_raw = get_variable_data(sdfFile, "Magnetic_Field_By")
    centre_field_component(by_raw, "y")
    by = np.copy(by_raw.take(indices_around_slice, axis=slice_dim))
    del by_raw
    if slice_dim != 1:
        by = by[:,:-1]

    bz_raw = get_variable_data(sdfFile, "Magnetic_Field_Bz")
    centre_field_component(bz_raw, "z")
    bz = np.copy(bz_raw.take(indices_around_slice, axis=slice_dim))
    del bz_raw
    if slice_dim != 2:
        bz = bz[:,:,:-1]

    gc.collect()

    dx = get_dx_dy_dz(sdfFile, cell_centre=True)

    print(asctime(), "Calculating current")
    cx, cy, cz = calc_curl(bx, by, bz, dx)

    cx = np.squeeze(cx[1:-1,:,:])
    cy = np.squeeze(cy[:,1:-1,:])
    cz = np.squeeze(cz[:,:,1:-1])

    return (cx, cy, cz)

def calc_vorticity_slice(sdfFile, slice_dim, slice_loc):
    slice_index = length_to_index(sdfFile, slice_loc, slice_dim)
    indices_around_slice = (slice_index-1, slice_index, slice_index+1)
    slice_dim = sanitise_dimension(slice_dim)

    print(asctime(), "Loading velocity")
    vx_raw = get_variable_data(sdfFile, "Velocity_Vx")
    vx = np.copy(vx_raw.take(indices_around_slice, axis=slice_dim))
    del vx_raw

    vy_raw = get_variable_data(sdfFile, "Velocity_Vy")
    vy = np.copy(vy_raw.take(indices_around_slice, axis=slice_dim))
    del vy_raw

    vz_raw = get_variable_data(sdfFile, "Velocity_Vz")
    vz = np.copy(vz_raw.take(indices_around_slice, axis=slice_dim))
    del vz_raw

    gc.collect()

    dx = get_dx_dy_dz(sdfFile, cell_centre=True)

    print(asctime(), "Calculating vorticity")
    cx, cy, cz = calc_curl(vx, vy, vz, dx)

    cx = np.squeeze(cx[1:-1,:,:])
    cy = np.squeeze(cy[:,1:-1,:])
    cz = np.squeeze(cz[:,:,1:-1])

    return (cx, cy, cz)

def differentiate(var, dimension, dx):
    # Differentiate var along given dimension (either string or index)
    # using central difference. dx is vector of grid spacing (dx, dy, dz)
    if type(dimension) is str:
        dimension = get_dimension_index(dimension)

    var_length = var.shape[dimension]
    ip = np.arange(2, var_length)
    im = np.arange(0, var_length-2)

    return (var.take(ip, axis=dimension) - var.take(im, axis=dimension))/(2.0*dx[dimension])


def calc_curl_component(var1, var2, dimension, dx):
    # Calculates the curl of var using var1 and var2 as the components involved in the curl
    # dx here is a vector of (dx, dy, dz)
    if type(dimension) is str:
        dimension = get_dimension_index(dimension)

    if dimension == 0:
        return differentiate(var2, "y", dx)[:,:,1:-1] - differentiate(var1, "z", dx)[:,1:-1,:]
    elif dimension == 1:
        return -(differentiate(var2, "x", dx)[:,:,1:-1] - differentiate(var1, "z", dx)[1:-1,:,:])
    elif dimension == 2:
        return differentiate(var2, "x", dx)[:,1:-1,:] - differentiate(var1, "y", dx)[1:-1,:,:]

def calc_cross_component(var1, var2, dimension):
    if type(dimension) is str:
        dimension = get_dimension_index(dimension)

    if dimension == 0:
        return var1[1]*var2[2] - var1[2]*var2[1]
        # return - var1[2]*var2[1]
        # return var1[1]*var2[2]
    elif dimension == 1:
        return -(var1[0]*var2[2] - var1[2]*var2[0])
    elif dimension == 2:
        return var1[0]*var2[1] - var1[1]*var2[0]

def calc_curl(varx, vary, varz, dx):
    return (
        calc_curl_component(vary, varz, "x", dx),
        calc_curl_component(varx, varz, "y", dx),
        calc_curl_component(varx, vary, "z", dx)
    )

def sanitise_dimension(dim):
    if type(dim) is str:
        return get_dimension_index(dim)
    else:
        return dim

def get_thick_slice(var, slice_dim, slice_index):
    # Returns a 3 gridpoint thick slice of the given variable
    indices_around_slice = (slice_index-1, slice_index, slice_index+1)
    return var.take(indices_around_slice, axis=slice_dim)

def calc_pressure_force_component_slice(sdfFile, var_dim, slice_dim, slice_loc):
    var_dim = sanitise_dimension(var_dim)
    slice_dim = sanitise_dimension(slice_dim)
    pressure = get_variable_data(sdfFile, "pressure")
    slice_index = length_to_index(sdfFile, slice_loc, slice_dim)
    # if slice_dim == var_dim:
    pressure = get_thick_slice(pressure, slice_dim, slice_index)
    # else:
        # pressure = pressure.take(slice_index, axis=slice_dim)

    dx = get_dx_dy_dz(sdfFile, cell_centre=True)
    pressure_gradient = -differentiate(pressure, var_dim, dx)

    for idx in [0,1,2]:
        if idx != var_dim:
            pressure_gradient = pressure_gradient.take(np.arange(1, pressure_gradient.shape[idx]-1), axis = idx)

    pressure_gradient = np.squeeze(pressure_gradient)
    return pressure_gradient

def get_velocity_component(sdfFile, var_dim):
    return get_variable_data(sdfFile, "Velocity_V" + var_dim)

def calc_second_derivative(var, dimension, dx):
    # calculate second deriative of var along given dimensionension (either string or index)
    # using central difference. dx is vector of grid spacing (dx, dy, dz)
    dimension = sanitise_dimension(dimension)

    var_length = var.shape[dimension]
    ip = np.arange(2, var_length)
    i = np.arange(1, var_length-1)
    im = np.arange(0, var_length-2)

    return (var.take(ip, axis=dimension) - 2.0*var.take(i, axis=dimension) + var.take(im, axis=dimension))/(dx[dimension]**2)

def calc_laplacian(var, dx):
    ddx = calc_second_derivative(var, "x", dx)
    for idx in [1, 2]:
        ddx = ddx.take(np.arange(1, ddx.shape[idx]-1), axis = idx)

    ddy = calc_second_derivative(var, "y", dx)
    for idx in [0, 2]:
        ddy = ddy.take(np.arange(1, ddy.shape[idx]-1), axis = idx)

    ddz = calc_second_derivative(var, "z", dx)
    for idx in [0, 1]:
        ddz = ddz.take(np.arange(1, ddz.shape[idx]-1), axis = idx)

    return ddx + ddy + ddz

def calc_isotropic_viscous_force_component_slice(sdfFile, var_dim, slice_dim, slice_loc):
    vel = get_velocity_component(sdfFile, var_dim)
    var_dim = sanitise_dimension(var_dim)
    slice_dim = sanitise_dimension(slice_dim)
    slice_index = length_to_index(sdfFile, slice_loc, slice_dim)
    vel = get_thick_slice(vel, slice_dim, slice_loc)
    dx = get_dx_dy_dz(sdfFile)

    laplacian = calc_laplacian(vel, dx)
    laplacian = laplacian.take(0, axis=slice_dim)

    return laplacian

def calc_lorentz_force_component_slice(sdfFile, var_dim, slice_dim, slice_loc):
    slice_dim = sanitise_dimension(slice_dim)
    var_dim = sanitise_dimension(var_dim)

    bx, by, bz = get_thick_slice_magnetic_field(sdfFile, slice_dim, slice_loc)

    dx = get_dx_dy_dz(sdfFile, cell_centre=True)

    # calculate current density
    cx, cy, cz = calc_curl(bx, by, bz, dx)

    # Get into correct dimensions
    cx = np.squeeze(cx[1:-1,:,:])
    cy = np.squeeze(cy[:,1:-1,:])
    cz = np.squeeze(cz[:,:,1:-1])

    # properly slice field now
    bx = bx.take(1, axis=slice_dim)
    by = by.take(1, axis=slice_dim)
    bz = bz.take(1, axis=slice_dim)

    bx = bx[1:-1,1:-1]
    by = by[1:-1,1:-1]
    bz = bz[1:-1,1:-1]

    return calc_cross_component((cx, cy, cz), (bx, by, bz), var_dim)

def get_thick_slice_magnetic_field(sdfFile, slice_dim, slice_loc):
    slice_dim = sanitise_dimension(slice_dim)
    bx = get_variable_data(sdfFile, "Magnetic_Field_Bx")
    by = get_variable_data(sdfFile, "Magnetic_Field_By")
    bz = get_variable_data(sdfFile, "Magnetic_Field_Bz")

    bx, by, bz = centre_magnetic_field(bx, by, bz)

    slice_index = length_to_index(sdfFile, slice_loc, slice_dim)
    indices_around_slice = (slice_index-1, slice_index, slice_index+1)

    # Take chunk of field around slice
    bx = bx.take(indices_around_slice, axis=slice_dim)
    by = by.take(indices_around_slice, axis=slice_dim)
    bz = bz.take(indices_around_slice, axis=slice_dim)

    return bx, by, bz

def calc_magnetic_pressure_force_component_slice(sdfFile, var_dim, slice_dim, slice_loc):
    var_dim = sanitise_dimension(var_dim)
    slice_dim = sanitise_dimension(slice_dim)

    bx, by, bz = get_thick_slice_magnetic_field(sdfFile, slice_dim, slice_loc)

    dx = get_dx_dy_dz(sdfFile, cell_centre=True)

    magnetic_pressure = np.sqrt(bx*bx + by*by + bz*bz) * 0.5

    mag_press_grad = differentiate(magnetic_pressure, var_dim, dx)

    for idx in [0,1,2]:
        if idx != var_dim:
            mag_press_grad = mag_press_grad.take(np.arange(1, mag_press_grad.shape[idx]-1), axis = idx)

    mag_press_grad = np.squeeze(mag_press_grad)

    return mag_press_grad

def calc_magnetic_tension_force_component_slice(sdfFile, var_dim, slice_dim, slice_loc):
    lorentz_force = calc_lorentz_force_component_slice(sdfFile, var_dim, slice_dim, slice_loc)
    mag_press_grad = calc_magnetic_pressure_force_component_slice(sdfFile, var_dim, slice_dim, slice_loc)

    return lorentz_force + mag_press_grad


def get_magnitude_current_at(sdfFile, zSliceIdx, xLimits = (0,-1), yLimits=(0,-1)):
    bx = slice_variable(\
        get_variable(sdfFile, "Magnetic_Field_bx_centred"),\
        x_min=xLimits[0], x_max = xLimits[1],\
        y_min=yLimits[0], y_max = yLimits[1],\
        z_min=zSliceIdx - 1, z_max = zSliceIdx + 2)
    by = slice_variable(\
        get_variable(sdfFile, "Magnetic_Field_by_centred"),\
        x_min=xLimits[0], x_max = xLimits[1],\
        y_min=yLimits[0], y_max = yLimits[1],\
        z_min=zSliceIdx - 1, z_max = zSliceIdx + 2)
    bz = slice_variable(\
        get_variable(sdfFile, "Magnetic_Field_bz_centred"),\
        x_min=xLimits[0], x_max = xLimits[1],\
        y_min=yLimits[0], y_max = yLimits[1],\
        z_min=zSliceIdx - 1, z_max = zSliceIdx + 2)
    
    dx, dy, dz = get_dx_dy_dz(sdfFile)

    gradbx = np.gradient(bx, dx, dy, dz)
    gradby = np.gradient(by, dx, dy, dz)
    gradbz = np.gradient(bz, dx, dy, dz)

    current_density = np.sqrt(np.power(gradbz[1] - gradby[2], 2) + np.power(gradbz[0] - gradbx[2], 2) + np.power(gradby[0] - gradbx[1], 2))
    current_density = current_density[:,::-1,int(current_density.shape[2]/2)]
    return current_density.transpose()

def get_temperature_at(sdfFile, zSliceIdx, xLimits = (0,-1), yLimits=(0,-1)):
    # Temperature in nondim units is just \gamma - 1 times the internal energy
    temp = (GAMMA-1) * slice_variable(\
        get_variable(sdfFile, "Fluid_Energy"),\
        x_min=xLimits[0], x_max = xLimits[1],\
        y_min=yLimits[0], y_max = yLimits[1],\
        z_min=zSliceIdx, z_max = zSliceIdx+1)
    
    temp = temp[:,:].squeeze(axis=2).transpose()
    
    return temp

def get_dimension_index(dimension):
    if dimension == "x":
        return 0
    elif dimension == "y":
        return 1
    elif dimension == "z":
        return 2

def length_to_index(sdfFile, x, dimension, relative=False):
    """converts length along dimension to index distance"""
    if type(dimension) is str:
        dimension = get_dimension_index(dimension)

    extents = sdfFile.Grid_Grid.extents
    dims = sdfFile.Grid_Grid.dims

    x0 = extents[dimension]
    xN = extents[dimension + 3]
    N = dims[dimension]

    dx = (xN-x0)/(N-1)

    if relative:
        # Don't move origin
        return int(x/dx)
    else:
        # Move point to origin before converting
        return int((x - x0)/dx)

def index_to_length(sdfFile, idx, dimension, relative=False):
    if type(dimension) is str:
        dimension = get_dimension_index(dimension)

    extents = sdfFile.Grid_Grid.extents
    dims = sdfFile.Grid_Grid.dims

    x0 = extents[dimension]
    xN = extents[dimension + 3]
    N = dims[dimension]

    dx = (xN-x0)/(N-1)

    if relative:
        return idx*dx
    else:
        return idx*dx + x0

def dimensionalise_temperature(tempIn):
    B0 = 5
    mf = 1.2
    mh_si = 1.672621777
    kb_si = 1.3806488
    mu0_si = 4.0 * np.pi
    RHO0 = 1.67
    T0 = (B0*B0)*mf*mh_si/(kb_si*mu0_si*RHO0) * 1e9

    return T0*tempIn
