#!/bin/bash
set -e

HIPIFY=/home/jieyang/dev/HIPIFY/build/hipify-clang

CUDA_SRC=./src/cuda
HIP_SRC=./src/hip
INCLUDE=./include
rm -rf $HIP_SRC
rm -rf $INCLUDE/hip
for f in $CUDA_SRC/*.cu
do

	echo "Hipifying "$f
	$HIPIFY $f --o-dir=$HIP_SRC -I $INCLUDE
	base=`basename "$f.hip" .cu.hip`
	mv $HIP_SRC/$base.cu.hip $HIP_SRC/$base.cpp
	sed -i 's/CUDA/HIP/g' $HIP_SRC/$base.cpp
	sed -i 's/cuda/hip/g' $HIP_SRC/$base.cpp
	rename  's/cuda/hip/' $HIP_SRC/$base.cpp
	# sed -i 's/tb = min/tb = std::min/g' $f
	# sed -i 's/B_adjusted = min/B_adjusted = std::min/g' $f
	# sed -i 's/B_adjusted = max/B_adjusted = std::max/g' $f
	# sed -i 's/tb = max/tb = std::max/g' $f
	# sed -i 's/tbx = min/tbx = std::min/g' $f
	# sed -i 's/tby = min/tby = std::min/g' $f
	# sed -i 's/tbz = min/tbz = std::min/g' $f
	# sed -i 's/tbx = max/tbx = std::max/g' $f
	# sed -i 's/tby = max/tby = std::max/g' $f
	# sed -i 's/tbz = max/tbz = std::max/g' $f
done

for f in $INCLUDE/cuda/*.h
do

	echo "Hipifying "$f
	$HIPIFY $f --o-dir=$INCLUDE/hip -I $INCLUDE
	base=`basename "$f.hip" .h.hip`
	mv $INCLUDE/hip/$base.h.hip $INCLUDE/hip/$base.h
	sed -i 's/CUDA/HIP/g' $INCLUDE/hip/$base.h
	sed -i 's/cuda/hip/g' $INCLUDE/hip/$base.h
	rename  's/cuda/hip/' $INCLUDE/hip/$base.h
done


cp $INCLUDE/mgard_api_cuda.h $INCLUDE/mgard_api_hip.h
sed -i 's/CUDA/HIP/g' $INCLUDE/mgard_api_hip.h
sed -i 's/cuda/hip/g' $INCLUDE/mgard_api_hip.h