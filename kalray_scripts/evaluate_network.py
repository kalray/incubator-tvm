"""
Script that evaluates the execution time of a neural network and can write sources for debug purposes

Author: Aurélien POTIN
"""
import os
import numpy as np

from pathlib import Path
import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
import tvm.contrib.graph_runtime as runtime

# parser
import argparse

# Specify args
parser = argparse.ArgumentParser()
parser.add_argument("network_name", help="Neural network to evaluate")
parser.add_argument("--path_log", help="Path of the tuning log")
parser.add_argument("--write", help="Write source files")

args = parser.parse_args()

# Target string, change it to suit your target
target = "opencl -device=kmppa -max_num_threads=16"
dtype="float32"

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = tvm.IRModule.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape

def eval_log(log_file, network, output_source=False):
    """
    Evaluates the network with the help of the logs given in argument.
    The logs may not correspond to the network's kernels and only the pertinent ones
    will be used.
    """

    mod, params, input_shape, out_shape = get_network(network, batch_size=1)

    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # export source if instructed to
        if output_source:
            with open("output_evaluator/source_host_mod", 'w') as source:
                source.write(lib.get_source())
            for i in range(len(lib.imported_modules)):
                with open("output_evaluator/source_imp_mod"+str(i), 'w') as source_file:
                    source_file.write(lib.imported_modules[i].get_source())

        #evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=500)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
            (np.mean(prof_res), np.std(prof_res)))

        return

def eval_fallback(network, output_source=None):
    """
    Evaluates the network without any tuning log, forcing it to fallback configuration
    for all kernels. It may also use tophup logs if they wre downloaded.
    """
    mod, params, input_shape, out_shape = get_network(network, batch_size=1)
    print("Compile...")
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build_module.build(
            mod, target=target, params=params)

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # export source if instructed to
        if output_source:
            with open("output_evaluator/source_host_mod", 'w') as source:
                source.write(lib.get_source())
            for i in range(len(lib.imported_modules)):
                with open("output_evaluator/source_imp_mod"+str(i), 'w') as source_file:
                    source_file.write(lib.imported_modules[i].get_source())

        #evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=500)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
            (np.mean(prof_res), np.std(prof_res)))

    return

if args.write:
    Path("output_evaluator/").mkdir(parents=True, exist_ok=True)
    output=True
else:
    output=False

if args.path_log:
    eval_log(args.path_log, args.network_name, output)
else:
    eval_fallback(args.network_name, output)
