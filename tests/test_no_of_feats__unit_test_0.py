import gc
import inspect
import os
import time

import dill as pickle
import numpy as np
import pytest
from src.ClassicML.OutlierStatMethods import base_class
from src.ClassicML.OutlierStatMethods.outlier_sum_stat_approx import OSPerm


def _log__test__values(values, duration, test_name, invocation_id):
    iteration = os.environ["CODEFLASH_TEST_ITERATION"]
    with open(os.path.join("/var/folders/ng/bwk7d4ds7wz95l011dbvtc9r0000gn/T/codeflash_f4g30tnf/", f"test_return_values_{iteration}.bin"), "ab") as f:
        return_bytes = pickle.dumps(values)
        _test_name = f"{test_name}".encode("ascii")
        f.write(len(_test_name).to_bytes(4, byteorder="big"))
        f.write(_test_name)
        f.write(duration.to_bytes(8, byteorder="big"))
        f.write(len(return_bytes).to_bytes(4, byteorder="big"))
        f.write(return_bytes)
        f.write(len(invocation_id).to_bytes(4, byteorder="big"))
        f.write(invocation_id.encode("ascii"))

def codeflash_wrap(wrapped, test_module_name, test_class_name, test_name, function_name, line_id, *args, **kwargs):
    test_id = f"{test_module_name}:{test_class_name}:{test_name}:{line_id}"
    if not hasattr(codeflash_wrap, "index"):
        codeflash_wrap.index = {}
    if test_id in codeflash_wrap.index:
        codeflash_wrap.index[test_id] += 1
    else:
        codeflash_wrap.index[test_id] = 0
    codeflash_test_index = codeflash_wrap.index[test_id]
    invocation_id = f"{line_id}_{codeflash_test_index}"
    print(f"!######{test_module_name}:{(test_class_name + '.' if test_class_name else '')}{test_name}:{function_name}:{invocation_id}######!")
    gc.disable()
    counter = time.perf_counter_ns()
    return_value = wrapped(*args, **kwargs)
    codeflash_duration = time.perf_counter_ns() - counter
    gc.enable()
    return (return_value, codeflash_duration, invocation_id)

def test_no_of_feats_typical_case():
    disease_data = np.random.rand(100, 10)
    control_data = np.random.rand(100, 10)
    osperm = OSPerm(disease_data, control_data)

def test_no_of_feats_empty_disease_data():
    control_data = np.random.rand(100, 10)
    with pytest.raises(ValueError, match='Input disease data is invalid'):
        OSPerm(np.array([]).reshape(0, 10), control_data)

def test_no_of_feats_empty_control_data():
    disease_data = np.random.rand(100, 10)
    with pytest.raises(ValueError, match='Input disease data is invalid'):
        OSPerm(disease_data, np.array([]).reshape(0, 10))

def test_no_of_feats_disease_data_zero_columns():
    disease_data = np.random.rand(100, 0)
    control_data = np.random.rand(100, 10)
    with pytest.raises(ValueError, match='Input disease data is invalid'):
        OSPerm(disease_data, control_data)

def test_no_of_feats_control_data_zero_columns():
    disease_data = np.random.rand(100, 10)
    control_data = np.random.rand(100, 0)
    with pytest.raises(ValueError, match='Input disease data is invalid'):
        OSPerm(disease_data, control_data)

def test_no_of_feats_none_disease_data():
    control_data = np.random.rand(100, 10)
    with pytest.raises(ValueError, match='Input disease data is invalid'):
        OSPerm(None, control_data)

def test_no_of_feats_none_control_data():
    disease_data = np.random.rand(100, 10)
    with pytest.raises(ValueError, match='Input disease data is invalid'):
        OSPerm(disease_data, None)

def test_no_of_feats_different_number_of_rows():
    disease_data = np.random.rand(100, 10)
    control_data = np.random.rand(50, 10)
    osperm = OSPerm(disease_data, control_data)

def test_no_of_feats_different_number_of_columns():
    disease_data = np.random.rand(100, 10)
    control_data = np.random.rand(100, 5)
    osperm = OSPerm(disease_data, control_data)

def test_no_of_feats_large_datasets():
    disease_data = np.random.rand(10000, 1000)
    control_data = np.random.rand(10000, 1000)
    osperm = OSPerm(disease_data, control_data)

def test_no_of_feats_stress_test_max_size():
    disease_data = np.random.rand(100000, 10000)
    control_data = np.random.rand(100000, 10000)
    osperm = OSPerm(disease_data, control_data)

def test_no_of_feats_minimum_valid_data():
    disease_data = np.random.rand(1, 1)
    control_data = np.random.rand(1, 1)
    osperm = OSPerm(disease_data, control_data)

def test_no_of_feats_single_row_data():
    disease_data = np.random.rand(1, 10)
    control_data = np.random.rand(1, 10)
    osperm = OSPerm(disease_data, control_data)

def test_no_of_feats_single_column_data():
    disease_data = np.random.rand(100, 1)
    control_data = np.random.rand(100, 1)
    osperm = OSPerm(disease_data, control_data)

def test_no_of_feats_nan_values():
    disease_data = np.random.rand(100, 10)
    disease_data[0, 0] = np.nan
    control_data = np.random.rand(100, 10)
    osperm = OSPerm(disease_data, control_data)

def test_no_of_feats_inf_values():
    disease_data = np.random.rand(100, 10)
    disease_data[0, 0] = np.inf
    control_data = np.random.rand(100, 10)
    osperm = OSPerm(disease_data, control_data)

def test_no_of_feats_different_data_types():
    disease_data = np.random.randint(0, 100, size=(100, 10))
    control_data = np.random.rand(100, 10)
    osperm = OSPerm(disease_data, control_data)

def test_no_of_feats_multiple_accesses():
    disease_data = np.random.rand(100, 10)
    control_data = np.random.rand(100, 10)
    osperm = OSPerm(disease_data, control_data)
