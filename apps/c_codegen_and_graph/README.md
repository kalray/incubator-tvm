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

# C codegen and graphgen

This folder contains scripts to generate pure C source code to target basically
any architecture that has a proper compiler backend.
It also contains a script to export the graph from different neural networks and
some helper tools in the subfolder `graph-gen` to obtain a good visualization of
the generated graph.
It is still pretty much a WIP for industrial / academical exploration of TVM
capabilities.

## Execution

In order to run any of the scripts you need to install TVM according to the
[documentation](https://tvm.apache.org/docs/install/index.html) and then set the
environmental variables `$TVM_HOME` and `$PYTHON_PATH` according to your
setup.

Then simply run the scripts with `python3`:
```bash
python3 c_source_export.py
python3 json_export.py
python3 opt_cmp.py
```

A quick summary of what each script is in their head.

The subfolder `graph-gen` contains its own README explaining how to run these
helper tools.
