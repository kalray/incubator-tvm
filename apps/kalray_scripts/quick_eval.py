"""
Quick Start Tutorial for Compiling Deep Learning Models
=======================================================
Original script Authors: Yao Wang <https://github.com/kevinthesun>, Truman Tian <https://github.com/SiNZeRo>

Modification Author: Aurélien Potin

This example shows how to build a neural network with Relay python frontend and
run it.
Notice that you need to build TVM llvm enabled.
"""

import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_runtime
# save the graph, lib and params into separate files
from tvm.contrib import util

######################################################################
# Define Neural Network in Relay
# ------------------------------
# First, let's define a neural network with relay python frontend.
# For simplicity, we'll use pre-defined squeezenet v1.1 network in Relay.
# Parameters are initialized with Xavier initializer.
# Relay also supports other model formats such as MXNet, CoreML, ONNX and
# Tensorflow.
#
# In this tutorial, we assume we will do inference on our device
# and the batch size is set to be 1. Input images are RGB color
# images of size 224 * 224. We can call the :any:`tvm.relay.TupleWrapper.astext()`
# to show the network structure.

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

mod, params = relay.testing.squeezenet.get_workload(version='1.1', batch_size=batch_size)

# set show_meta_data=True if you want to show meta data
# print(mod.astext(show_meta_data=False))

######################################################################
# Compilation
# -----------
# Next step is to compile the model using the Relay/TVM pipeline.
# Users can specify the optimization level of the compilation.
# Currently this value can be 0 to 3. The optimization passes include
# operator fusion, pre-computation, layout transformation and so on.
#
# :py:func:`relay.build` returns three components: the execution graph in
# json format, the TVM module library of compiled functions specifically
# for this graph on the target hardware, and the parameter blobs of
# the model. During the compilation, Relay does the graph-level
# optimization while TVM does the tensor-level optimization, resulting
# in an optimized runtime module for model serving.
#
# Behind the scene, :py:func:`relay.build`
# first does a number of graph-level optimizations, e.g. pruning, fusing, etc.,
# then registers the operators (i.e. the nodes of the optimized graphs) to
# TVM implementations to generate a `tvm.module`.
# To generate the module library, TVM will first transfer the high level IR
# into the lower intrinsic IR of the specified target backend
# Then the machine code will be generated as the module library.

opt_level = 1
target='opencl -device=kmppa -max_num_threads=16'
with tvm.transform.PassContext(opt_level=opt_level):
    print("Creating graph")
    graph, lib, params = relay.build(mod,
                                     target=target,
                                     #target_host='llvm -mtriple=x86_64-unknown-linux',
                                     params=params)
#####################################################################
# Run the generate library
# ------------------------

# create random input
ctx = tvm.context(target)
data = np.random.uniform(-1, 1, size=data_shape).astype("float32")
# create module
print('Runtime creation')
module = graph_runtime.create(graph, lib, ctx)
print('Runtime created')

# set input and parameters
module.set_input("data", data)
module.set_input(**params)


print("Get Host LLVM IR code")
with open("host-code.ll", 'w') as hostfile:
    hostfile.write(lib.get_source())

print("Get OpenCL source code, number of files = ", len(lib.imported_modules))
for i in range(len(lib.imported_modules)):
    with open("file"+str(i)+".cl", 'w') as openclfile:
        openclfile.write(lib.imported_modules[i].get_source())

lib.export_library("deploy_lib.tar")
print("Finished saving")

# run
print("Run started")
module.run()
print("Run finished")

#Uncomment the next block of code to evaluate the execution time
# evaluate
print("Evaluate inference time cost...")
ftimer = module.module.time_evaluator("run", ctx, number=20, repeat=2)
prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
      (np.mean(prof_res), np.std(prof_res)))
