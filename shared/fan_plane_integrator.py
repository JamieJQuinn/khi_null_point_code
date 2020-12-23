"""
Functions for integrating quantities azimuthally around a central axis
and calculating Mach numbers based on integrated velocities
"""

from sdf_helper import *
import scipy
from scipy.signal import find_peaks, peak_widths
from scipy.signal import savgol_filter

def get_r_z_limits(sdfFile, r_min, r_max, z_min, z_max):
    extents = get_extents(sdfFile)
    x_min = extents[0]
    r_limits = (
        int(length_to_index(sdfFile, r_min + x_min, "x")),
        int(length_to_index(sdfFile, r_max + x_min, "x"))
    )
    z_limits = (
        int(length_to_index(sdfFile, z_min, "z")),
        int(length_to_index(sdfFile, z_max, "z"))
    )
    return r_limits, z_limits

def cart2cyl(data, r_limits, z_limits):
    """
    Converts data to cyl coords within r_limits and z_limits.
    Assumes radial centre of data is exactly at the centre of the box
    """
    # Calc centre
    x_res = data.shape[0]
    y_res = data.shape[1]
    # calculate true centre (interpolation can handle non-int indices)
    centre = ((x_res-1)/2, (y_res-1)/2)
    
    # Calc new coords
    r = np.array(range(r_limits[0], r_limits[1]))
    theta_res = int(len(r)*2) # automatic calculation of theta resolution
    thetas = np.linspace(0, 2*np.pi, theta_res)
    z = np.array(range(z_limits[0], z_limits[1]))
    R, THETA, Z = np.meshgrid(r, thetas, z, indexing='ij')

    # Calc new coord positions in old coords
    X = R*np.cos(THETA) + centre[0]
    Y = R*np.sin(THETA) + centre[1]
    coord_transform = np.array([X, Y, Z])

    # Transform data
    data_cyl = scipy.ndimage.map_coordinates(data, coord_transform, order=1)

    return data_cyl

def calc_angular_mean(data, r_limits, z_limits):
    """Calculates the mean of data in the x-y about the centre"""
    data_cyl = cart2cyl(data, r_limits, z_limits)
    return np.mean(data_cyl, axis=1)

def calc_velocity_shear(velocity_slice, z_min, z_max):
    peaks, _ = find_peaks(np.abs(velocity_slice))
#     print(peaks)
#     print(velocity_slice[peaks])
#     print(np.argsort(velocity_slice[peaks]))
#     print(np.argsort(velocity_slice[peaks])[-2:])
#     print(peaks[np.argsort(velocity_slice[peaks])[-2:]])
    if len(peaks) >= 2:
        p0, p1 = peaks[np.argsort(velocity_slice[peaks])[-2:]]
        v0 = velocity_slice[p0]
        v1 = velocity_slice[p1]
        delta_v = v0 + v1
        layer_width = abs(p1 - p0)*(z_max-z_min)/velocity_shear.shape[0]
    else:
        delta_v = layer_width = 0
    return (delta_v, layer_width)

def calc_tallest_peak(variable):
    peaks, _ = find_peaks(variable)
    if not peaks.size == 0:
        return peaks[np.argmax(variable[peaks])]
    else:
        return np.argmax(variable)
#     return np.argmax(variable)

def calc_theta_component(vx, vy):
    """Calculates the theta component of the vector given by (vx, vy)"""
    theta = np.linspace(0, 2*np.pi, num=vx.shape[1])
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    v_theta = np.zeros_like(vx)
    
    for i, th in enumerate(theta):
        v_theta[:,i,:] = -vx[:,i,:]*np.sin(th) + vy[:,i,:]*np.cos(th)
    
    return v_theta

def calc_peaks(variable):
    n_radius = variable.shape[0]
    peaks = np.zeros((n_radius, 2), dtype=int)

    for rad_idx in range(n_radius):
        var_slice  = variable[rad_idx]
        midpoint = int(len(var_slice)/2)
        lower_peak = calc_tallest_peak(np.abs(var_slice[:midpoint]))
        upper_peak = midpoint + \
            calc_tallest_peak(np.abs(var_slice[midpoint:]))

        peaks[rad_idx] = (lower_peak, upper_peak)
    
    return peaks

def calc_peak_locations(peaks, z):
    return np.array([
        (z[peak[0]], z[peak[1]]) for peak in peaks
    ])
    
def calc_layer_widths(peak_locations):
    return np.array([
        abs(loc[1] - loc[0]) for loc in peak_locations
    ])

def calc_delta_layer(variable, peaks):
    return np.array([
        abs(variable[i, peak[1]] - variable[i, peak[0]]) for i, peak in enumerate(peaks)
    ])

def calc_mach_numbers(sdfFile, r_min, r_max, z_min, z_max):
    SMOOTHING_ENABLED = False
    SMOOTHING_WINDOW_SIZE = 31
    SMOOTHING_POLYNOMIAL_DEGREE = 6
    
    r_limits, z_limits = get_r_z_limits(sdfFile, r_min, r_max, z_min, z_max)

    # CALC VELOCITY SHEAR PEAKS
    vx = cart2cyl(calc_centred_velocity(
        get_variable_data(sdfFile, "Velocity_Vx")),
        r_limits, z_limits)
    vy = cart2cyl(calc_centred_velocity(
        get_variable_data(sdfFile, "Velocity_Vy")),
        r_limits, z_limits)

    velocity_mean = np.mean(calc_theta_component(vx, vy), axis=1)

    vel_peaks = calc_peaks(velocity_mean)

    z = np.linspace(z_min, z_max, num=velocity_mean.shape[1])
    vel_peak_locations = calc_peak_locations(vel_peaks, z)
    vel_delta = calc_delta_layer(velocity_mean, vel_peaks)

    # Smooth velocity deltas and peak locations
    if SMOOTHING_ENABLED:
        vel_delta = savgol_filter(vel_delta, SMOOTHING_WINDOW_SIZE, SMOOTHING_POLYNOMIAL_DEGREE)
        vel_peak_locations[:,0] = savgol_filter(vel_peak_locations[:,0], SMOOTHING_WINDOW_SIZE, SMOOTHING_POLYNOMIAL_DEGREE)
        vel_peak_locations[:,1] = savgol_filter(vel_peak_locations[:,1], SMOOTHING_WINDOW_SIZE, SMOOTHING_POLYNOMIAL_DEGREE)

    vel_layer_widths = calc_layer_widths(vel_peak_locations)

    r = np.linspace(r_min, r_max, num=vel_layer_widths.size)
#     ax.imshow(np.transpose(velocity_mean), extent = (r_min, r_max, z_min, z_max))
#     ax.plot(r, vel_peak_locations[:,0], 'w')
#     ax.plot(r, vel_peak_locations[:,1], 'w')
#     plt.show()

    # CALC ALFVEN VELOCITY
    VA = get_variable_data(sdfFile, "alfven_velocity")
    VA_mean = calc_angular_mean(VA, r_limits, z_limits)
    z0_idx = int((z_limits[1] - z_limits[0])/2)
    VA_mean_slice = VA_mean[:, z0_idx]

    #     ax.imshow(np.transpose(VA_mean), extent = (r_min, r_max, z_min, z_max))
    #     plt.colorbar()
    #     plt.show()

    # CALC SOUND SPEED
    sound_speed = get_variable_data(sdfFile, "sound_speed")
    sound_speed_mean = calc_angular_mean(sound_speed, r_limits, z_limits)
    sound_speed_slice = sound_speed_mean[:, z0_idx]

    #     ax.imshow(np.transpose(sound_speed_mean), extent = (r_min, r_max, z_min, z_max))
    #     plt.colorbar()
    #     plt.show()

    # CALC MAGNETIC SHEAR PEAKS
    bx = cart2cyl(get_variable_data(sdfFile, "Magnetic_Field_Bx_centred"), r_limits, z_limits)
    by = cart2cyl(get_variable_data(sdfFile, "Magnetic_Field_By_centred"), r_limits, z_limits)

    magnetic_mean = np.mean(calc_theta_component(bx, by), axis=1)
    #     ax.imshow(np.transpose(magnetic_mean), extent = (r_min, r_max, z_min, z_max))

    mag_peaks = calc_peaks(magnetic_mean)

    z = np.linspace(z_min, z_max, num=magnetic_mean.shape[1])
    mag_peak_locations = calc_peak_locations(mag_peaks, z)
    mag_delta = calc_delta_layer(magnetic_mean, mag_peaks)

    # smooth
    if SMOOTHING_ENABLED:
        mag_delta = savgol_filter(mag_delta, SMOOTHING_WINDOW_SIZE, SMOOTHING_POLYNOMIAL_DEGREE)
        mag_peak_locations[:,0] = savgol_filter(mag_peak_locations[:,0], SMOOTHING_WINDOW_SIZE, SMOOTHING_POLYNOMIAL_DEGREE)
        mag_peak_locations[:,1] = savgol_filter(mag_peak_locations[:,1], SMOOTHING_WINDOW_SIZE, SMOOTHING_POLYNOMIAL_DEGREE)

    mag_layer_widths = calc_layer_widths(mag_peak_locations)

    #     r = np.linspace(r_min, r_max, num=mag_layer_widths.size)
    #     ax.plot(r, mag_peak_locations[:,0], 'k')
    #     ax.plot(r, mag_peak_locations[:,1], 'k')
    #     plt.show()

    sqrt_rho = np.sqrt(get_variable_data(sdfFile, "Fluid_Rho"))
    sqrt_rho_mean = calc_angular_mean(sqrt_rho, r_limits, z_limits)
    sqrt_rho_slice = sqrt_rho_mean[:, z0_idx]

    fast_mach = vel_delta / np.sqrt(np.power(sound_speed_slice, 2) + np.power(VA_mean_slice, 2))
    alfven_mach = vel_delta * sqrt_rho_slice / mag_delta
    # Due to dividing two small terms, we add one to both before division
    delta1 = (mag_layer_widths+1)/(vel_layer_widths+1) * np.power(alfven_mach, 2.0/3.0)
#     delta = mag_layer_widths / vel_layer_widths * np.power(alfven_mach, 2.0/3.0)

    return fast_mach, alfven_mach, delta1
