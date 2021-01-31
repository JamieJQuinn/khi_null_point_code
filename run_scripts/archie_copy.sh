#!/usr/bin/env bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" >/dev/null && pwd  )"

resolution=640
time_end=15.0
dt_snapshots=0.5

visc='-isotropic'
switching_param=150

damping=".TRUE."

z_min=-0.25
z_max=0.25

visc_mantissa='1'
resist_mantissa='1'

for n in 4; do
  for m in 4; do
    for visc in '-switching'; do
      #for twisting_velocity in 0.005 0.01 0.05 0.25; do
        folder=${twisting_velocity}v-${n}r-$m$visc
        cp -rv $SCRIPT_DIR/lare3d $folder
        cd $folder
        viscosity=$visc_mantissa'.0_num*10.0_num**(-'$n'_num)'
        resistivity=$resist_mantissa'.0_num*10.0_num**(-'$m'_num)'
        sed -i -e 's/visc3 =.*/visc3 = '$viscosity'/' src/control.f90
        sed -i -e 's/eta_background =.*/eta_background = '$resistivity'/' src/control.f90
        sed -i -e 's/\(switching_param = \).*\(_num\)/\1'${switching_param}'\2/' src/control.f90
        sed -i -e 's/\(nx_global = \).*/\1'${resolution}'/' src/control.f90
        sed -i -e 's/\(ny_global = \).*/\1'${resolution}'/' src/control.f90
        sed -i -e 's/\(nz_global = \).*/\1'${resolution}'/' src/control.f90
        sed -i -e 's/\(t_end = \).*\(_num\)/\1'${time_end}'\2/' src/control.f90
        sed -i -e 's/\(dt_snapshots = \).*\(_num\)/\1'${dt_snapshots}'\2/' src/control.f90
        sed -i -e 's/\(damping = \).*/\1'${damping}'/' src/control.f90
        sed -i -e 's/\(z_min = \).*\(_num\)/\1'${z_min}'\2/' src/control.f90
        sed -i -e 's/\(z_max = \).*\(_num\)/\1'${z_max}'\2/' src/control.f90
        sed -i -e 's/\(initial = \).*/\1IC_NEW/' src/control.f90
        sed -i -e 's/\(restart_snapshot = \).*/\11/' src/control.f90
        #sed -i -e 's/\(twisting_velocity = \).*\(_num\)/\1'${twisting_velocity}'\2/' src/control.f90
        #sed -i -e 's/\(energy = \).*/\1 1.0_num\/(gamma-1.0_num)*2.0e4_num\/T0/' src/initial_conditions.f90
        cd ..
      #done
    done
  done
done
