"""

Author: Aurélien POTIN
"""
from pathlib import Path

import tvm
from tvm import relay
from tvm.relay import testing

directory_name = "opt-cmp-output/"
Path(directory_name).mkdir(parents=True, exist_ok=True)
# Defining neural network
batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)

# Getting existing relay IR of the neural network
mod, params = relay.testing.squeezenet.get_workload(batch_size, image_shape=image_shape, version = '1.1', num_classes=num_class)

opt_to_generate = [0, 1, 2, 3]
target = 'c'

for curr_opt in opt_to_generate:
    prefix = 'opt' + str(curr_opt) + "_"
    print("---Writing Relay graph to JSON---")
    with open(directory_name + 'relay_graph.json', 'w') as fichier:
        fichier.write(tvm.ir.save_json(mod))
    print("Done")

    print("---Writing params---")
    with open(directory_name + 'params.bin', 'wb') as fichier:
        fichier.write(relay.save_param_dict(params))
    print("Done")


    # Apply optimisations and generate source code for target
    with tvm.transform.PassContext(opt_level=curr_opt):
        print("Optimizing and building target module...")
        graph, lib, params = relay.build(mod,
                                    target=target,
                                    params=params)

    print("---Writing Relay graph to JSON---")
    with open(directory_name + prefix + 'relay_graph.json', 'w') as export_file:
        export_file.write(graph)
    print("Done")

    print("---Writing params---")
    with open(directory_name + prefix + 'params.bin', 'wb') as export_file:
        export_file.write(relay.save_param_dict(params))
    print("Done")


    print("---Writing source files---")
    print("\tWriting host code")
    with open(directory_name + prefix + 'host_source', 'w') as source_file:
        source_file.write(lib.get_source())

    print("\tWriting imported modules sources, number to do: ", len(lib.imported_modules))
    for i in range(len(lib.imported_modules)):
        with open(directory_name + prefix + "imported_source"+str(i), 'w') as source_file:
            source_file.write(lib.imported_modules[i].get_source())

print("Done")
