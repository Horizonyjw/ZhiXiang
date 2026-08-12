import json
import numpy as np

path = "demo/demo_20_samples.npz"
data = np.load(path, allow_pickle=False)

inputs = data["inputs"]
targets = data["targets"]
metadata = json.loads(str(data["metadata_json"]))

print("=== B1 demo interface check ===")
print("sample count :", inputs.shape[0])
print("inputs shape :", inputs.shape)
print("targets shape:", targets.shape)
print("dtype        :", inputs.dtype)
print("value range  :", float(inputs.min()), float(inputs.max()))
print("first sample :", metadata[0]["start_time"])
print("last sample  :", metadata[-1]["start_time"])

assert inputs.shape == (20, 5, 1, 352, 512)
assert targets.shape == (20, 3, 1, 352, 512)
assert inputs.dtype == np.float32
assert targets.dtype == np.float32

print("PASS")
