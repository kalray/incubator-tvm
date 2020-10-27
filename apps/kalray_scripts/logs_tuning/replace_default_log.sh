#!/usr/bin/env sh

FILE=~/.tvm/tophub/opencl_v0.04.log
if [ -f "$FILE" ]; then
	cp "$FILE" "$FILE.bak"
	cat ./resnet-50/resnet-50_Best.tuningLog \
		./squeezenet_v1.1/squeezenet_v1.1_Win2.tuningLog >> $FILE
	echo "Backed up old parameters to $FILE.bak and replaced with MPPA autotuning on $FILE"
else
	mkdir -p ~/.tvm/tophub
	cat ./resnet-50/resnet-50_Best.tuningLog \
		./squeezenet_v1.1/squeezenet_v1.1_Win2.tuningLog > $FILE
	echo "Successfully created MPPA autotuning to $FILE"
fi
