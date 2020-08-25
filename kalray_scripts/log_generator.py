import os
import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime

log_filename = "resnet-50.log"
tmp_log_file = log_filename + ".tmp"


# pick best records to a cache file
autotvm.record.pick_best(tmp_log_file, log_filename)
