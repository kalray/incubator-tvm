"""
This script evaluates a TVM library.

Author: Aurélien Potin
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
ctx.sync() # FIXME: Unsure if needed

costs = time_f(*args).results
