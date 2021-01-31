#!/usr/bin/env bash
ISOTROPIC_DIR=/nas/1101974q/archie-latest-organisation-sept-2018/null-point-stressing/twisting-driver/high-resolution/v-4r-4-isotropic
SWITCHING_DIR=/nas/1101974q/archie-latest-organisation-sept-2018/null-point-stressing/twisting-driver/high-resolution/v-4r-4-switching
#ls ${ISOTROPIC_DIR}/Data/*.vti ${SWITCHING_DIR}/Data/*.sdf | sed -e 's/ /\\n/g' > filename_list

#echo ${ISOTROPIC_DIR}/Data/0000_preprocessed.vti | sed -e 's/ /\\n/g' > filename_list
echo ${ISOTROPIC_DIR}/Data/0008_preprocessed.vti | sed -e 's/ /\\n/g' > filename_list
echo ${ISOTROPIC_DIR}/Data/0030_preprocessed.vti | sed -e 's/ /\\n/g' >> filename_list
echo ${ISOTROPIC_DIR}/Data/0037_preprocessed.vti | sed -e 's/ /\\n/g' >> filename_list
#echo ${ISOTROPIC_DIR}/Data/0009.sdf | sed -e 's/ /\\n/g' >> filename_list
#echo ${ISOTROPIC_DIR}/Data/0010.sdf | sed -e 's/ /\\n/g' >> filename_list
#echo ${ISOTROPIC_DIR}/Data/0035.sdf | sed -e 's/ /\\n/g' >> filename_list
#echo ${SWITCHING_DIR}/Data/0035.sdf | sed -e 's/ /\\n/g' >> filename_list
