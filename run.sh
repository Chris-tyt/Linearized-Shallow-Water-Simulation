# option: mpi gpu serial serial_omp basic_serial 
# --scenario water_drop dam_break wave river
mod=gpu
scenario=water_drop
nx=250
ny=250
niter=10000
make $mod
./build/${mod} --nx ${nx} --nx ${ny} --num_iter ${niter} --scenario ${scenario} --output ${mod}.out
# python utils/visualizer.py ${mod}.out ${mod}.gif

# ncu --target-processes all --export report_${mod}.ncu-rep ./build/${mod}
# ncu --import report_${mod}.ncu-rep > report_${mod}.txt