# v3：针对当前 ZhiXiang `03_index` 真实列名的修复

根据当前仓库输出，真实字段为：

`frames.csv`

```text
frame_id, filename, relative_path, timestamp, split, segment_id,
center_code, product_code, lat_min, lat_max, lon_min, lon_max,
width, height, mode
```

`samples.csv`

```text
sample_id, split, segment_id, start_time, end_time,
input_rel_paths_json, target_rel_paths_json,
input_times_json, target_times_json,
center_code, product_code, lat_min, lat_max, lon_min, lon_max
```

v3 已将其精确映射为认证层字段：

```text
relative_path            -> path
input_rel_paths_json     -> input_paths
target_rel_paths_json    -> target_paths
input_times_json         -> input_timestamps
target_times_json        -> target_timestamps
```

请重新执行 **02 和 03**。无需重新运行数据清洗，也无需修改 CSV。
