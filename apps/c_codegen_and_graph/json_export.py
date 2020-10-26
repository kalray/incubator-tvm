"""
This script allows to test the export process of JSON Relay graph as well
as the parameters in a binary file.

Author: Aurélien POTIN
"""
from pathlib import Path

import tvm
from tvm import relay
from tvm.relay import testing

directory_name = "json-export-output/"
Path(directory_name).mkdir(parents=True, exist_ok=True)
# Définition du reseau de neuronnes
batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

# Neural network generation from Relay pre-translated networks
mod, params = relay.testing.squeezenet.get_workload(batch_size, image_shape=image_shape, version = '1.1', num_classes=num_class)

target='c'
# Apply optimisations according to opt_level and generates "true" Relay graph
with tvm.transform.PassContext(opt_level=0):
    print("Optimizing and building target module...")
    graph, lib, params = relay.build(mod,
                                    target=target,
                                    params=params)


print("---Writing Relay graph to JSON---")
with open(directory_name + 'relay_graph.json', 'w') as export_file:
    export_file.write(graph)
print("Done")

print("---Writing params---")
with open(directory_name + 'params.bin', 'wb') as export_file:
    export_file.write(relay.save_param_dict(params))
print("Done")
