#!/usr/bin/env sh

FILE=~/.tvm/tophub/opencl_v0.04.log
if [ -f "$FILE" ]; then
	cat ./resnet-50/resnet-50_Best.tuningLog \
		./squeezenet_v1.1/squeezenet_v1.1_Win2.tuningLog >> $FILE
	echo "Successfully merged MPPA autotuning to $FILE"
else
	mkdir -p ~/.tvm/tophub
	cat ./resnet-50/resnet-50_Best.tuningLog \
		./squeezenet_v1.1/squeezenet_v1.1_Win2.tuningLog > $FILE
	echo "Successfully created MPPA autotuning to $FILE"
fi
