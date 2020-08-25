# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Import a model compiled with Relay and run it with provided image.
========================
Takes a model compiled previously with TVM and run inference on it
with provided image an data.
As it is a POC, it will only target llvm and execute on CPU, it may be parametrised
but compiled model needs to be compatible
"""

# parser
import argparse

import pickle

# Specify args
parser = argparse.ArgumentParser()
parser.add_argument("path_mod", help="Relay lib to load")
parser.add_argument("path_arg", help="Pickled arg file")
args = parser.parse_args()

# tvm
import tvm
from tvm import nd
# os and numpy
import numpy as np

target = 'opencl'
ctx = tvm.context(target)

# Open model
loaded_mod = tvm.runtime.load_module(args.path_mod)
with open(args.path_arg, 'rb') as arg_file:
    loaded_args = pickle.load(arg_file)


time_f = loaded_mod.time_evaluator(loaded_mod.entry_name, ctx, number=1, repeat=1)

args = [nd.empty(x[0], dtype=x[1], ctx=ctx) for x in loaded_args]
args = [nd.array(x, ctx=ctx) for x in args]
ctx.sync() #Unsure FIXME

costs = time_f(*args).results
