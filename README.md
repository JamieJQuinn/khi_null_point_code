# khi_null_point_code
This is the analysis code for my paper on the effect of anisotropic viscosity on the Kelvin-Helmhotz instability in the fan plane of a magnetic null point.

## Requirements

**Required**
- MPI (for the SDF library)

**Optional**
- VisIt (for field line plotting)

## Setup

These steps set up the SDF library and python requirements and link the SDF python bindings.

Ensure SDF submodule and all its submodules are set up:
```
[in /]
git submodule update --init --recursive
```

Make SDF's C interface:
```
[in code/SDF/C]
make
```

Install virtual environment and python requirements via pipenv:
```
[in code/]
pipenv install
```

Make and link SDF's python interface inside pipenv environment:
```
[in code/]
pipenv shell
cd SDF/utilities
./build -3 -s
```

## Running an analysis

Run notebook:
```
[in code/<project>]
pipenv run jupyter <notebook>
```

## Plotting field lines

The field lines are the only plots not plotted in jupyter notebooks. These are, instead, plotted using the package VisIt. The plotting scripts can be found in the `field_line_plots` folder. A field line plot is produced by specifying the input SDF files in the `fill_filelist.sh` script and calling `make`. This runs VisIt with the plotting script `plot.py`. This script is mostly constructed from the output from VisIt's Macro feature which records the user's actions in the GUI and represents them programatically. Saving this output in `plot.py` allows quick reproduction of a figure.

## Files

The code is split into the `shared` directory containing useful functions and the `main` directory containing Juptyer notebooks which run the analysis and produce the figures found in the paper. The explanation of each file can be found below. 

### main

- `khi_typical_case.Rmd` - single case analysis
- `parameter_study.Rmd` - parameter study analysis
- `exploratory_data_analysis.Rmd` - exploratory analysis (does not produce figures)
- `field_line_integrator.Rmd` - development of field line integrator (does not produce figures)
- `fan_plane_integrator.Rmd` - development of fan plane integrator (does not produce figures)
- `sdf_file_subsampler.Rmd` - notebook used to extract and save parts of the SDF files for more efficient analysis (does not produce figures)

### shared

- `energy.py` - functions for manipulating and plotting the continuous energy outputs from `lare3d` stored in `en.dat` files
- `fan_plane_integrator.py` - functions for integrating around the fan plane in chapter 6 (null_point_khi)
- `field_line_integrator.py` - functions for integrating along field lines in chapters 5 and 6
- `parameters.py` - holds relevant physical parameters used in the simulations
- `plotting.py` - functions for plotting
- `sdf_helper.py` - functions for manipulating SDF files

## Reproduction of simulation data

The code which implements only the anisotropic viscosity module can be found at \url{https://github.com/jamiejquinn/lare3d} in the `feature/anisotropic_viscosity` branch and it should be a simple process to merge the changes into another version of Lare3d for future research. To facilitate reproduction of the simulation data presented in the paper, the specific code, including initial conditions, boundary conditions and parameters, is packaged in the `run/null-point-khi` branch pf the repository (\url{https://github.com/jamiejquinn/lare3d}). The specific parameters used in each individual simulation can be found in the methods section of the paper. The parameters were inputted to the simulations using the tools found in the `run_scripts` folder of this repository. These can be used to quickly generate multiple simulations suitable for a parameter study.

The scripts in `run_scripts` consist of 4 individual scripts, designed to be run from the location at which the data will be outputted:

`archie_copy.sh` will copy the entire lare3d folder and change the appropriate source files in the way required for a parameter run, specific run, etc. This should be heavily edited.

`archie_build.sh` will build the simulations (with folders passed at command line arguments) with the correct viscosity model, detected via the folder name. A folder with the suffix `-isotropic` will use isotropic viscosity and `-switching` will use switching. The script should be read to understand the other options available.

`archie_start.sh` will submit the given folder to the archie-west queue. This is included for completeness.

`archie_set_restart.sh` is used to restart a simulation after it has completed or has been cancelled by a cluster scheduler.

The most important script for reproduction of the data is within `archie_copy.sh`. It should be relatively clear how to edit this script to recreate out simulations but if not, please get in touch.
