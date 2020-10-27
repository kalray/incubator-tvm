<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Kalray TVM scripts

This folder contains Kalray scripts to run and evaluate TVM using the [MPPA3
(Coolidge)](https://www.kalrayinc.com/wp-content/uploads/2019/09/Kalray_HiPEAC_Paper_Award_DAC_2019.pdf)
as an accelerator using the OpenCL framework.

## Build and Execution

The process described here has been tested on an Ubuntu based Kalray developer
machine with MPPA PCIe driver version 2.0 and the SDK ACB4.2.0

### Build TVM

Before you build TVM shared object, follow
[these](https://tvm.apache.org/docs/install/from_source.html#tvm-package)
instructions to install the required dependencies.
Then export these environmental variables according to your repository clone
location:

```bash
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

If needed, source the Kalray toolchain so executables prefixed with `kvx-` will
become available on your path. Now we will create the build folder and setup the
configuration.

```bash
mkdir build
cd build
cp ../cmake/config.cmake .
```

Change the following variables in this config file

```cmake
set(USE_OPENCL ON)
set(USE_LLVM /usr/lib/llvm-10/bin/llvm-config)
set(USE_GRAPH_RUNTIME_DEBUG ON)
set(USE_RELAY_DEBUG ON)
```

If tensorflow is used you must install ANTLR.
LLVM versions < 10 should work but have not been tested.
Still inside the `build` folder it is time to make our TVM library

```bash
make -j$(nproc)
```

Once the process is done you should have the `libtvm_runtime.so` and `libtvm.so`
shared objects. Please verify that they were linked against Kalray's libraries
with `ldd`. You should see `libOpenCL.so` and others pointing to a Kalray's
accesscore lib path.

### Execution

At this point you should have all the required binaries to run TVM via OpenCL on
a Kalray processor.
Before running any script, please make sure you have sourced the Kalray
toolchain and defined the environmental variables `$TVM_HOME` and
`$PYTHON_PATH`.

Also, specific tuning has been done for the MPPA3 on some neural networks which
is available at the subfolder `logs_tuning`. These logs are used by the OpenCL
runtime to determine the appropriate parameters to use according to the platform
and it is important that they are in your TVM cache. To do this, either replace

```bash
./replace_default_log.sh
```

or merge

```bash
./merge_with_default_log.sh
```

with the default logs by running the respective script on the `logs_tuning`
folder.

Once this is done the getting started script to deploy is the `quick_eval` which
can be run by issuing

```bash
python3 quick_eval.py
```

At the end of the run you should get inference time information.
Intermediate files are output into the current folder for debugging and analysis
purposes: `file0.cl` is the OpenCL code run on the MPPA, `host-code.ll` is the
LLVM IR host file and `deploy_lib.tar` contains the module serialization object
files.

The `evaluate_network` is another script that can be easily run to get other
networks running as well as their execution time.

The other scripts have different purposes and are mostly WIP for internal use.
Each one contains a header with additional info.

## Troubleshooting and good practices

### General info on TVM

The scripts in `kalray_scripts` folder are based on information from
[this](https://tvm.apache.org/docs/tutorials/get_started/relay_quick_start.html)
page.  For newcomers, it is highly recommended to read this to have a global
overview of how TVM works.

### POCL cache

If you have limited home directory size permissions, before running the scripts,
defining a POCL cache folder. This is done with the environmental variable
`$POCL_CACHE_DIR`. Refer to this
[link](http://portablecl.org/docs/html/env_variables.html) for additional
information.

### POCL debug

Debug log messages can be obtained from POCL with the environmental variable
`$POCL_DEBUG`. With this set to `1` the user has minimal info that can help to
identify common problems. Refer to this
[link](http://portablecl.org/docs/html/debug.html) for additional information.

### OpenCL MPPA debug

In order to properly debug applications at runtime, please read Kalray's
OpenCLManual mainly section 8.2 Debug PCIe Acceleration Application.
