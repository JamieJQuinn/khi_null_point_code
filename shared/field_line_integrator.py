import sdf
import numpy as np
from sdf_helper import get_magnetic_field
from multiprocessing import Pool
from functools import partial

class Integrator:
    def __init__(self, mag_field, integrand_field, h):
        self.h = h
        self.mag_field = mag_field
        self.integrand_field = integrand_field
        self.max_steps = 1e4

    def integrate_in_direction(self, pos, direction):
        # direction is +1 or -1 depending on forward or backwards
        running = self.mag_field.in_field(pos)
        integral = 0.0
        length = 0.0
        steps = 0

        left_integrand = self.integrand_field.get_value_at(pos)
        while running:
            # Calculate derivative
            try:
                derivative = get_derivative(
                    self.mag_field, pos, self.h)
            except IndexError:
                running = False
                continue

            # Calculate distance between points (i.e. ds)
            ds = np.linalg.norm(derivative)

            pos += direction*derivative

            if self.mag_field.in_field(pos) and steps < self.max_steps:
                right_integrand = self.integrand_field.get_value_at(pos)

                # Integrate via mid-point method
                integral += ds*0.5*(right_integrand + left_integrand)
                length += ds

                left_integrand  = right_integrand
                steps += 1
            else:
                running = False

        return integral / length

    def integrate(self, pos, direction='forward'):
        if direction == 'forward':
            return self.integrate_in_direction(pos, +1)
        elif direction == 'backward':
            return self.integrate_in_direction(pos, -1)
        elif direction == 'both':
            return self.integrate_in_direction(pos, +1) + self.integrate_in_direction(pos, -1)

class Line:
    def __init__(self, x, y, z, num_points):
        self.x = np.zeros(num_points+1)
        self.y = np.zeros(num_points+1)
        self.z = np.zeros(num_points+1)
        self.x[0], self.y[0], self.z[0] = x, y, z
        self.current = 0

    def get_pos(self):
        return np.array([
            self.x[self.current],
            self.y[self.current],
            self.z[self.current]])

    def add_derivative(self, derivative):
        self.x[self.current+1] = self.x[self.current] + derivative[0]
        self.y[self.current+1] = self.y[self.current] + derivative[1]
        self.z[self.current+1] = self.z[self.current] + derivative[2]
        self.current += 1

    def plot(self, axis, **kwargs):
        axis.plot(self.x[:self.current+1],
                  self.y[:self.current+1],
                  self.z[:self.current+1], **kwargs)

class VectorField():
    def __init__(self, x, y, z, extents):
        self.x = ScalarField(x, extents)
        self.y = ScalarField(y, extents)
        self.z = ScalarField(z, extents)
        self.extents = extents

    def get_field_at(self, v):
        return np.array([
            self.x.get_value_at(v),
            self.y.get_value_at(v),
            self.z.get_value_at(v)
        ])

    def calc_dx_dy_dz(self):
        return self.x.calc_dx_dy_dz()

    def in_field(self, pos):
        return self.x.in_field(pos)

class ScalarField():
    def __init__(self, data, extents):
        self.data = data
        self.extents = extents

        self.x0 = self.extents[0]
        self.y0 = self.extents[1]
        self.z0 = self.extents[2]

        self.xN = self.extents[3]
        self.yN = self.extents[4]
        self.zN = self.extents[5]

        self.xDim = data.shape[0]
        self.yDim = data.shape[1]
        self.zDim = data.shape[2]

        self.xToIndex = 1.0/(self.xN-self.x0) * (self.xDim-1)
        self.yToIndex = 1.0/(self.yN-self.y0) * (self.yDim-1)
        self.zToIndex = 1.0/(self.zN-self.z0) * (self.zDim-1)

    def calc_dx_dy_dz(self):
        return (self.xN - self.x0)/(self.xDim-1),\
               (self.yN - self.y0)/(self.yDim-1),\
               (self.zN - self.z0)/(self.zDim-1),\

    def x_to_index(self, length):
        return (length - self.x0)*self.xToIndex

    def y_to_index(self, length):
        return (length - self.y0)*self.yToIndex

    def z_to_index(self, length):
        return (length - self.z0)*self.zToIndex

    def interpolate_linear(self, v):
        x_index = self.x_to_index(v[0])
        y_index = self.y_to_index(v[1])
        z_index = self.z_to_index(v[2])
        i = int(x_index)
        j = int(y_index)
        k = int(z_index)
        xw = x_index - i
        yw = y_index - j
        zw = z_index - k

        xwp = 1.0-xw
        ywp = 1.0-yw
        zwp = 1.0-zw

        field_cube = self.data[i:i+2, j:j+2, k:k+2]

        z_av = zwp*field_cube[:,:,0]\
              + zw*field_cube[:,:,1]
        y_av = ywp*z_av[:,0]\
              + yw*z_av[:,1]
        x_av = xwp*y_av[0]\
              + xw*y_av[1]

        return x_av

    def get_value_at(self, v):
        return self.interpolate_linear(v)

    def in_field(self, pos):
        i, j, k = int(self.x_to_index(pos[0])),\
                  int(self.y_to_index(pos[1])),\
                  int(self.z_to_index(pos[2]))
        in_x = i >= 1 and i < self.xDim-3
        in_y = j >= 1 and j < self.yDim-3
        in_z = k >= 1 and k < self.zDim-3
        return in_x and in_y and in_z

def rk4(field, y, h):
    k1 = h*field.get_field_at(y)
    k2 = h*field.get_field_at(y+0.5*k1)
    k3 = h*field.get_field_at(y+0.5*k2)
    k4 = h*field.get_field_at(y+0.5*k3)
    return 1.0/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

def rk2(field, y, h):
    k1 = h*field.get_field_at(y)
    k2 = h*field.get_field_at(y+0.5*k1)
    return k2

def get_derivative(field, y, h):
    return rk2(field, y, h)
#     return rk4(field, y, h)

def calculate_parallel_electric_field(field):
    bx, by, bz = field.x.data, field.y.data, field.z.data,
    dx, dy, dz = field.calc_dx_dy_dz()

    gradbx = np.gradient(bx, dx, dy, dz)
    gradby = np.gradient(by, dx, dy, dz)
    gradbz = np.gradient(bz, dx, dy, dz)

    j_dot_B = (gradbz[1] - gradby[2])*bx - (gradbz[0] - gradbx[2])*by + (gradby[0] - gradbx[1])*bz
    integrand = j_dot_B / (np.sqrt(bx*bx + by*by + bz*bz))

    return integrand

def load_magnetic_field(sdfFile):
    mag_field = get_magnetic_field(sdfFile)
    extents = sdfFile.Grid_Grid.extents
    field = VectorField(mag_field[0], mag_field[1], mag_field[2], extents)
    return field

def run_field_line_integrator(vector_field, scalar_field, z_level, side_length, h=None):
    extents = [-side_length, side_length,
           -side_length, side_length]
    dx, _, _ = scalar_field.calc_dx_dy_dz()
    if h == None:
        h = 0.5*dx
    n_values = 2*int(2.0*side_length / h)

    print("step size:", h)
    print("field lines per side:", n_values)
    print("total number of field lines:", n_values**2)

    seeds = np.mgrid[
        extents[0]:extents[1]:(n_values)*1j,
        extents[2]:extents[3]:(n_values)*1j,
        z_level:z_level:1j].reshape(3,-1).T

    integrator = Integrator(vector_field, scalar_field, h)

    with Pool(4) as p:
        result = np.array(
            p.map(partial(integrator.integrate, direction='forward'), seeds)
        ).reshape(n_values, n_values)

    return result, n_values
