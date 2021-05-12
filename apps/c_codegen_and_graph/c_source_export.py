"""
Script that exports all source code to files

Author: Aurelien POTIN
"""
from pathlib import Path
import json
import numpy as np

import tvm
from tvm import relay
from tvm.relay import testing

class NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, tvm.runtime.ndarray.NDArray):
            return obj.asnumpy().tolist()
        return json.JSONEncoder.default(self, obj)

directory_name = "c-source-export-output/"
Path(directory_name).mkdir(parents=True, exist_ok=True)
# Defining neural network
batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

# Getting existing relay IR of the neural network
mod, params = relay.testing.squeezenet.get_workload(batch_size, image_shape=image_shape, version = '1.1', num_classes=num_class)

target = 'c'

# Apply optimisations according to opt_level and generate source code for target
with tvm.transform.PassContext(opt_level=0):
    print("Optimizing and building target module...")
    graph, lib, params = relay.build(mod,
                                     target=target,
                                     params=params)

print("---Writing Relay graph to JSON---")
with open(directory_name + 'relay_graph.json', 'w') as export_file:
    export_file.write(graph)
print("Done")

print("---Writing raw params---")
json_str = json.dumps(params, cls=NDArrayEncoder, indent=4)
with open(directory_name + 'params.json', 'w') as export_file:
    export_file.write(json_str)

print("---Writing params---")
with open(directory_name + 'params.bin', 'wb') as export_file:
    export_file.write(relay.save_param_dict(params))
print("Done")

print("---Writing source files---")
print("\tWriting host code")
with open(directory_name + 'host_source', 'w') as source_file:
    source_file.write(lib.get_source())
print("\tWriting imported modules sources, number to do: ", len(lib.imported_modules))
for i in range(len(lib.imported_modules)):
    with open(directory_name + "imported_source"+str(i), 'w') as source_file:
        source_file.write(lib.imported_modules[i].get_source())

print("Done")
