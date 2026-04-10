#!/bin/csh -xy

module load nco

set user = wwieder
set case = ctsm5.4.029_f09_SP_GenericCrop_154_2000
set target_dir = /glade/derecho/scratch/${user}/ADF_unstruct/${case}/ts
#set target_dir = /glade/derecho/scratch/${user}/archive/${case}/lnd/proc/ts
#set pout = ${target_dir}/regrid
set wgts = /glade/work/wwieder/mapping_files/map_fv0.9x1.25_to_1x1_scripgrids_conserve_nomask_c260408.nc 

# Check if directory exists before moving
if ( -d $target_dir ) then
    cd $target_dir
    echo "Now in directory: `pwd`"
else
    echo "Error: Directory $target_dir not found."
endif

cd ${target_dir}
rm -rf regrid
mkdir -p regrid

foreach file ( ${case}.clm2.h0a.* )
  echo "  " ${file}
  ncremap -t 1 --sgs_frc=landfrac --sgs_msk=landmask -m ${wgts} ${file} regrid/${file} >& /dev/null
end

exit

