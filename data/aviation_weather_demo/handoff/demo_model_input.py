import json
import numpy as np

data = np.load("demo/demo_20_samples.npz", allow_pickle=False)
inputs = data["inputs"]
targets = data["targets"]
metadata = json.loads(str(data["metadata_json"]))

batch_size = 4

for start in range(0, len(inputs), batch_size):
    x = inputs[start:start + batch_size]
    y = targets[start:start + batch_size]
    m = metadata[start:start + batch_size]
    print(
        f"batch {start // batch_size + 1}: "
        f"inputs={x.shape}, targets={y.shape}, "
        f"start={m[0]['start_time']}"
    )
