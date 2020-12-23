#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import struct

VERBOSE = False

class Energy:
    def __init__(self, fname, slice_vec=[], skip=0, nvars=None):
        self.valid = True
        self.fromfile(fname, slice_vec, skip, nvars)

    def read_data(self, f, slice_vec, en_nvars):
        if VERBOSE:
            print("LOADING DATA")
        self.data = np.fromfile(f, dtype=np.float64)
        if VERBOSE:
            print("data size:", len(self.data))

        n_timesteps = int(len(self.data)/en_nvars)
        if VERBOSE:
            print("n_timesteps:", n_timesteps)

        offset = len(self.data) - n_timesteps*en_nvars
        if VERBOSE:
            print("offset:", offset)
        if offset:
            if VERBOSE:
                print("spare numbers:", self.data[-offset:])
            self.data = self.data[:-offset]

        self.data = self.data.reshape((n_timesteps, en_nvars))
        if slice_vec:
            if VERBOSE:
                print("Slice:", slice_vec)
            self.data = self.data[slice_vec[0] : slice_vec[1]]

    def fromfile(self, fname, slice_vec=[], skip=0, nvars=0):
        f = open(fname, 'rb')

        if skip == 0:
            if VERBOSE:
                print("LOADING HEADER")
            self.magic         = f.read(3)
            self.version       = self.read_int(f)
            self.revision      = self.read_int(f)
            self.endianness    = self.read_int(f)
            self.header_length = self.read_int(f)
            self.num_sz        = self.read_int(f)
            self.en_nvars      = self.read_int(f)
            self.id_length     = self.read_int(f)
            self.varnames      = f.read(self.en_nvars*self.id_length).split()
            if VERBOSE:
                print("version:", self.version)
                print("revision:", self.revision)
                print("endianness:", self.endianness)
                print("header_length:", self.header_length)
                print("num_sz:", self.num_sz)
                print("n_vars:", self.en_nvars)
                print("id_length:", self.id_length)
            self.read_data(f, slice_vec, self.en_nvars)
        else:
            f.read(skip)
            self.read_data(f, slice_vec, nvars)

        f.close()

    def fortran_to_int(self, n):
        return int.from_bytes(n, byteorder='little')

    def read_int(self, fp):
        integer_byte_size = 4
        fortran_int = fp.read(integer_byte_size)
        return self.fortran_to_int(fortran_int)

    def fix_current_restart(self):
        if self.en_nvars > 8:
            current = self.data[:, 8]
            minimum = current[1:].argmin()
            if current[minimum+1] < 0.001:
                current[minimum+1] = 0.5*(current[minimum] + current[minimum+2])

    def fix_ohmic_heating_restart(self):
        ohmic_heating = self.data[:, 5]
        for i in range(2):
            diff = ohmic_heating[1:] - ohmic_heating[:-1]
            if np.any(np.abs(diff) > 0.01):
                max_point = np.argmax(np.abs(diff))
                if diff[max_point] > 0:
                    ohmic_heating[max_point+1:] += diff[max_point]
                else:
                    ohmic_heating[max_point+1:] -= diff[max_point]

    def get_max(self, index, min_time=-1, max_time=-1):
        if max_time > 0:
            max_index = self.time_to_index(max_time)
        else:
            max_index = len(self.data[:,0])+1

        if min_time > 0:
            min_index = self.time_to_index(min_time)
        else:
            min_index = 0

        return np.max(self.data[min_index:max_index, index])

    def time_to_index(self, time):
        return self.__find_nearest(self.data[:, 0], time)

    def get_at_time(self, index, time):
        return self.data[self.time_to_index(time), index]

    def __find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def __format_number(self, num):
        num = float(num)
        num = "{:.2e}".format(num)
        mant, exp = num.split('e')
        return "{0} \\times 10^{{{1}}}".format(mant, exp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Print/plot lare3d en.dat files')

    parser.add_argument('filenames', help='energy files to open', nargs='+')
    parser.add_argument('--print_latest', action='store_true',
                        help='print latest row')
    parser.add_argument('--print_all', action='store_true',
                        help='print all rows')
    parser.add_argument('--print_varnames', action='store_true',
                        help='print saved variable names')
    parser.add_argument('--plot_columns', type=int, nargs='*',
                        help='plot given columns')
    parser.add_argument('--slice_vec', type=int, nargs=2,
                        help='take slice of data to plot')
    parser.add_argument('--nvars', type=int,
                        help='Number of variables in en.dat')
    parser.add_argument('--bytes_to_skip', type=int, default=0)
    parser.add_argument('--xlim', type=float, nargs=2,
                        help='limits of x-axis')
    parser.add_argument('--ylim', type=float, nargs=2,
                        help='limits of y-axis')
    parser.add_argument('--plot_dt', action='store_true',
                        help='plot dt')
    parser.add_argument('--plot_all', action='store_true',
                        help='plot all variables')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--save', action='store_true',
                        help='Save graph')
    parser.add_argument('--output',
                        help='output filename')
    parser.add_argument('--logy', action='store_true',
                        help='Plot x against logy')
    parser.add_argument('--verbose', action='store_true',
                        help='Print extra information')

    args = parser.parse_args()

    if args.debug:
        print(args)

    energies = [Energy(f, slice_vec=args.slice_vec, skip=args.bytes_to_skip, nvars=args.nvars) for f in args.filenames]

    if args.print_varnames:
        print(energies[0].varnames)

    if args.print_latest:
        print(energies[0].data[-1])

    if args.print_all:
        for row in energies[0].data:
            np.set_printoptions(linewidth=140)
            print(row)

    if args.plot_all:
        args.plot_columns = range(1, energies[0].en_nvars)

    if args.plot_dt:
        time = energies[0].data[:, 0]
        dt = time[1:] - time[:-1]
        plt.plot(time[:-1],dt)
        plt.show()

    def formLineStyle(filename):
        if len(filename.split("/")) <= 1:
            return ''
        linestyle = ''
        # if filename.split("/")[-3].split("-")[-1] == "switching":
            # linestyle = '--'
        # if filename.split("/")[-3].split("-")[-3] == "20r":
            # linestyle = ':'

        return linestyle


    if args.plot_columns:
        fig, axes = plt.subplots(len(args.plot_columns), sharex=True, figsize=(2*len(args.plot_columns), 10))
        colours = ['C' + str(i) for i in range(10)]
        colourMap = {}
        for en, f in zip(energies, args.filenames):
            simMode = "".join(f.split("-")[:-1])
            if simMode not in colourMap:
                colourMap[simMode] = colours.pop()
            for axis, index in zip(axes, args.plot_columns):
                if args.logy:
                    axis.semilogy(en.data[:, 0], en.data[:,index],
                              colourMap[simMode] + formLineStyle(f),
                              label="/".join(f.split("/")[:-2]))
                else:
                    axis.plot(en.data[:, 0], en.data[:,index],
                              colourMap[simMode] + formLineStyle(f),
                              label="/".join(f.split("/")[:-2]))
                axis.set_title(en.varnames[index])
                axis.legend()
                if args.ylim:
                    axis.set_ylim(args.ylim[0], args.ylim[1])
                if args.xlim:
                    axis.set_xlim(args.xlim[0], args.xlim[1])
        plt.tight_layout()
        if args.save:
            if args.output:
                output = args.output
            else:
                output = "print_energy.png"
            plt.savefig(output)
        else:
            plt.show()
