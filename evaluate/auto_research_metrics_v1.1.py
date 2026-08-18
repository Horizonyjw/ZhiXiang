"""eval_v1.1 第二层 Auto Research 批次评测器。

本脚本只读取调度器生成的 experiment_registry.jsonl，不读取雷达图片。
每行是一条实验记录，计算 VCR、ACR、Error Recovery、Gain/GPU-hour、
Gain/100K Tokens 和 Validation-Test Gap。

必需字段：
  experiment_id:          非空字符串，全文件唯一
  candidate_id:           非空字符串
  split:                  "val" 或 "test"
  run_completed:          bool
  evaluation_status:      "completed"、"invalid" 或 "failed"
  reproducible:           bool
  manual_intervention:    bool
  recoverable_failure:    bool
  auto_recovery_succeeded: bool
  gpu_seconds:            >= 0 的数值，包含失败和重试
  token_count:            >= 0 的数值，包含失败和重试
  test_accessed:          bool；val 搜索记录必须为 false
  mae:                    第一层 overall MAE；无有效结果时为 null
  persistence_mae:        同 split 的 Persistence overall MAE；无结果时为 null

可选字段可由调度器继续保存，例如 parent_experiment_id、code_version、
dataset_version、started_at、ended_at、failure_type 等；本评测器不会删除或改写它们。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EVALUATOR_VERSION = "auto_research_metrics_v1.1"
DEFAULT_SPEC_PATH = Path(__file__).resolve().parents[1] / "docs" / "eval_v1.1.md"
ALLOWED_SPLITS = {"val", "test"}
ALLOWED_EVALUATION_STATUSES = {"completed", "invalid", "failed"}
REQUIRED_FIELDS = {
    "experiment_id",
    "candidate_id",
    "split",
    "run_completed",
    "evaluation_status",
    "reproducible",
    "manual_intervention",
    "recoverable_failure",
    "auto_recovery_succeeded",
    "gpu_seconds",
    "token_count",
    "test_accessed",
    "mae",
    "persistence_mae",
}


class AutoResearchEvaluationInvalid(ValueError):
    """实验登记簿不符合 eval_v1.1 第二层接口。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_nonnegative(value: Any, field: str, line_number: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AutoResearchEvaluationInvalid(
            f"第 {line_number} 行 {field} 必须是非负数值"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise AutoResearchEvaluationInvalid(
            f"第 {line_number} 行 {field} 必须是有限非负数"
        )
    return number


def _optional_positive_metric(value: Any, field: str, line_number: int) -> float | None:
    if value is None:
        return None
    number = _finite_nonnegative(value, field, line_number)
    if field == "persistence_mae" and number <= 0:
        raise AutoResearchEvaluationInvalid(
            f"第 {line_number} 行 persistence_mae 必须大于 0"
        )
    return number


def _require_bool(record: dict[str, Any], field: str, line_number: int) -> bool:
    value = record[field]
    if type(value) is not bool:
        raise AutoResearchEvaluationInvalid(
            f"第 {line_number} 行 {field} 必须是 true 或 false"
        )
    return value


def _validate_record(record: Any, line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise AutoResearchEvaluationInvalid(f"第 {line_number} 行必须是 JSON 对象")
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise AutoResearchEvaluationInvalid(
            f"第 {line_number} 行缺少字段：{', '.join(missing)}"
        )

    normalized = dict(record)
    for field in ("experiment_id", "candidate_id"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise AutoResearchEvaluationInvalid(
                f"第 {line_number} 行 {field} 必须是非空字符串"
            )
        normalized[field] = value.strip()

    if record["split"] not in ALLOWED_SPLITS:
        raise AutoResearchEvaluationInvalid(
            f"第 {line_number} 行 split 只能是 val 或 test"
        )
    if record["evaluation_status"] not in ALLOWED_EVALUATION_STATUSES:
        raise AutoResearchEvaluationInvalid(
            f"第 {line_number} 行 evaluation_status 必须是 completed、invalid 或 failed"
        )

    for field in (
        "run_completed",
        "reproducible",
        "manual_intervention",
        "recoverable_failure",
        "auto_recovery_succeeded",
        "test_accessed",
    ):
        normalized[field] = _require_bool(record, field, line_number)

    normalized["gpu_seconds"] = _finite_nonnegative(
        record["gpu_seconds"], "gpu_seconds", line_number
    )
    normalized["token_count"] = _finite_nonnegative(
        record["token_count"], "token_count", line_number
    )
    normalized["mae"] = _optional_positive_metric(
        record["mae"], "mae", line_number
    )
    normalized["persistence_mae"] = _optional_positive_metric(
        record["persistence_mae"], "persistence_mae", line_number
    )

    valid = (
        normalized["run_completed"]
        and normalized["evaluation_status"] == "completed"
        and normalized["reproducible"]
    )
    if valid and (
        normalized["mae"] is None or normalized["persistence_mae"] is None
    ):
        raise AutoResearchEvaluationInvalid(
            f"第 {line_number} 行是有效实验，但 mae 或 persistence_mae 为 null"
        )
    if normalized["auto_recovery_succeeded"] and not normalized["recoverable_failure"]:
        raise AutoResearchEvaluationInvalid(
            f"第 {line_number} 行自动恢复成功，但未标记 recoverable_failure"
        )
    if normalized["split"] == "test" and not normalized["test_accessed"]:
        raise AutoResearchEvaluationInvalid(
            f"第 {line_number} 行是 test 评测，但 test_accessed 不是 true"
        )
    return normalized


def load_registry(path: str | Path) -> list[dict[str, Any]]:
    registry_path = Path(path)
    if not registry_path.is_file():
        raise AutoResearchEvaluationInvalid(f"找不到实验登记簿：{registry_path}")
    records: list[dict[str, Any]] = []
    try:
        with registry_path.open("r", encoding="utf-8-sig") as file:
            for line_number, raw_line in enumerate(file, start=1):
                if not raw_line.strip():
                    continue
                try:
                    parsed = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise AutoResearchEvaluationInvalid(
                        f"第 {line_number} 行不是合法 JSON：{error.msg}"
                    ) from error
                records.append(_validate_record(parsed, line_number))
    except OSError as error:
        raise AutoResearchEvaluationInvalid(f"无法读取实验登记簿：{registry_path}") from error
    if not records:
        raise AutoResearchEvaluationInvalid("实验登记簿不能为空")
    ids = [record["experiment_id"] for record in records]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise AutoResearchEvaluationInvalid(
            f"experiment_id 重复：{duplicates[:5]}"
        )
    return records


def _is_valid(record: dict[str, Any]) -> bool:
    return bool(
        record["run_completed"]
        and record["evaluation_status"] == "completed"
        and record["reproducible"]
    )


def _relative_gain(record: dict[str, Any]) -> float | None:
    if not _is_valid(record):
        return None
    mae = record["mae"]
    baseline = record["persistence_mae"]
    if mae is None or baseline is None:
        return None
    return float((baseline - mae) / baseline)


def _ratio_metric(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator == 0:
        return {
            "value": None,
            "numerator": numerator,
            "denominator": denominator,
            "status": "undefined_zero_denominator",
        }
    return {
        "value": float(numerator / denominator),
        "numerator": numerator,
        "denominator": denominator,
        "status": "computed",
    }


def _efficiency_metric(gain: float | None, resource_units: float) -> dict[str, Any]:
    if gain is None:
        return {
            "value": None,
            "relative_mae_gain": None,
            "resource_units": resource_units,
            "status": "undefined_no_valid_val_gain",
        }
    if resource_units == 0:
        return {
            "value": None,
            "relative_mae_gain": gain,
            "resource_units": resource_units,
            "status": "undefined_zero_resource",
        }
    return {
        "value": float(gain / resource_units),
        "relative_mae_gain": gain,
        "resource_units": resource_units,
        "status": "computed",
    }


def _best_valid_record(records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [record for record in records if _relative_gain(record) is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda record: float(_relative_gain(record)))


def _validation_test_gap(
    records: list[dict[str, Any]], final_candidate_id: str | None
) -> dict[str, Any]:
    valid_val = [
        record for record in records if record["split"] == "val" and _is_valid(record)
    ]
    valid_test = [
        record for record in records if record["split"] == "test" and _is_valid(record)
    ]
    if final_candidate_id is None:
        val_ids = {record["candidate_id"] for record in valid_val}
        test_ids = {record["candidate_id"] for record in valid_test}
        matched = sorted(val_ids & test_ids)
        if len(matched) == 1:
            final_candidate_id = matched[0]
        elif not matched:
            return {
                "value": None,
                "candidate_id": None,
                "val_relative_mae_gain": None,
                "test_relative_mae_gain": None,
                "status": "not_available_until_final_test",
            }
        else:
            return {
                "value": None,
                "candidate_id": None,
                "val_relative_mae_gain": None,
                "test_relative_mae_gain": None,
                "status": "ambiguous_final_candidate",
            }

    candidate_val = _best_valid_record(
        record
        for record in valid_val
        if record["candidate_id"] == final_candidate_id
    )
    candidate_tests = [
        record for record in valid_test if record["candidate_id"] == final_candidate_id
    ]
    if candidate_val is None or not candidate_tests:
        return {
            "value": None,
            "candidate_id": final_candidate_id,
            "val_relative_mae_gain": (
                _relative_gain(candidate_val) if candidate_val is not None else None
            ),
            "test_relative_mae_gain": None,
            "status": "not_available_until_final_test",
        }
    if len(candidate_tests) != 1:
        return {
            "value": None,
            "candidate_id": final_candidate_id,
            "val_relative_mae_gain": _relative_gain(candidate_val),
            "test_relative_mae_gain": None,
            "status": "invalid_multiple_test_certifications",
        }
    val_gain = _relative_gain(candidate_val)
    test_gain = _relative_gain(candidate_tests[0])
    assert val_gain is not None and test_gain is not None
    return {
        "value": float(val_gain - test_gain),
        "candidate_id": final_candidate_id,
        "val_relative_mae_gain": val_gain,
        "test_relative_mae_gain": test_gain,
        "status": "computed",
    }


def evaluate_records(
    records: list[dict[str, Any]], final_candidate_id: str | None = None
) -> dict[str, Any]:
    """按 eval_v1.1 汇总一批实验；test 认证不计入候选实验分母。"""
    val_records = [record for record in records if record["split"] == "val"]
    valid_val = [record for record in val_records if _is_valid(record)]
    vcr = _ratio_metric(len(valid_val), len(val_records))
    acr = _ratio_metric(
        sum(not record["manual_intervention"] for record in valid_val),
        len(val_records),
    )
    recoverable = [record for record in val_records if record["recoverable_failure"]]
    recovered = [record for record in recoverable if record["auto_recovery_succeeded"]]
    error_recovery = _ratio_metric(len(recovered), len(recoverable))

    best_record = _best_valid_record(valid_val)
    best_gain = _relative_gain(best_record) if best_record is not None else None
    total_gpu_seconds = float(sum(record["gpu_seconds"] for record in val_records))
    total_token_count = float(sum(record["token_count"] for record in val_records))
    illegal_test_access = [
        record["experiment_id"]
        for record in val_records
        if record["test_accessed"]
    ]
    per_experiment = []
    for record in records:
        gain = _relative_gain(record)
        per_experiment.append(
            {
                "experiment_id": record["experiment_id"],
                "candidate_id": record["candidate_id"],
                "split": record["split"],
                "valid_and_reproducible": _is_valid(record),
                "relative_mae_gain": gain,
                "gpu_seconds": record["gpu_seconds"],
                "token_count": record["token_count"],
            }
        )

    return {
        "evaluator_version": EVALUATOR_VERSION,
        "batch_summary": {
            "registry_record_count": len(records),
            "submitted_val_candidate_count": len(val_records),
            "valid_reproducible_val_count": len(valid_val),
            "total_val_gpu_seconds_including_failures_and_retries": total_gpu_seconds,
            "total_val_token_count_including_failures_and_retries": total_token_count,
            "best_val_experiment_id": (
                best_record["experiment_id"] if best_record is not None else None
            ),
            "best_val_candidate_id": (
                best_record["candidate_id"] if best_record is not None else None
            ),
            "best_val_relative_mae_gain": best_gain,
        },
        "metrics": {
            "vcr": vcr,
            "acr": acr,
            "error_recovery": error_recovery,
            "gain_per_gpu_hour": _efficiency_metric(
                best_gain, total_gpu_seconds / 3600.0
            ),
            "gain_per_100k_tokens": _efficiency_metric(
                best_gain, total_token_count / 100000.0
            ),
            "validation_test_gap": _validation_test_gap(
                records, final_candidate_id
            ),
        },
        "test_access_audit": {
            "illegal_access_detected": bool(illegal_test_access),
            "illegal_val_experiment_ids": illegal_test_access,
            "status": "failed" if illegal_test_access else "passed",
        },
        "per_experiment": per_experiment,
        "evaluation_status": (
            "invalid_test_access" if illegal_test_access else "completed"
        ),
    }


def evaluate_registry(
    registry_path: str | Path,
    *,
    final_candidate_id: str | None = None,
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    records = load_registry(registry_path)
    spec = Path(spec_path) if spec_path else DEFAULT_SPEC_PATH
    if not spec.is_file():
        raise AutoResearchEvaluationInvalid(f"找不到评测规范：{spec}")
    result = evaluate_records(records, final_candidate_id)
    result.update(
        {
            "spec_path": str(spec),
            "spec_digest": _sha256_file(spec),
            "registry_path": str(Path(registry_path)),
            "registry_sha256": _sha256_file(Path(registry_path)),
            "evaluator_script_sha256": _sha256_file(Path(__file__)),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return result


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def _base_record(**changes: Any) -> dict[str, Any]:
    record = {
        "experiment_id": "val-1",
        "candidate_id": "candidate-a",
        "split": "val",
        "run_completed": True,
        "evaluation_status": "completed",
        "reproducible": True,
        "manual_intervention": False,
        "recoverable_failure": False,
        "auto_recovery_succeeded": False,
        "gpu_seconds": 3600,
        "token_count": 100000,
        "test_accessed": False,
        "mae": 0.9,
        "persistence_mae": 1.0,
    }
    record.update(changes)
    return record


def run_self_test() -> None:
    records = [
        _base_record(),
        _base_record(
            experiment_id="val-2",
            candidate_id="candidate-b",
            recoverable_failure=True,
            auto_recovery_succeeded=True,
            mae=0.8,
        ),
        _base_record(
            experiment_id="test-1",
            candidate_id="candidate-b",
            split="test",
            test_accessed=True,
            gpu_seconds=120,
            token_count=1000,
            mae=0.85,
        ),
    ]
    result = evaluate_records(records, "candidate-b")
    assert result["metrics"]["vcr"]["value"] == 1.0
    assert result["metrics"]["acr"]["value"] == 1.0
    assert result["metrics"]["error_recovery"]["value"] == 1.0
    assert abs(result["metrics"]["gain_per_gpu_hour"]["value"] - 0.1) < 1e-12
    assert abs(result["metrics"]["gain_per_100k_tokens"]["value"] - 0.1) < 1e-12
    assert abs(result["metrics"]["validation_test_gap"]["value"] - 0.05) < 1e-12
    assert result["test_access_audit"]["status"] == "passed"

    no_failures = evaluate_records([_base_record()])
    assert no_failures["metrics"]["error_recovery"]["value"] is None
    assert no_failures["metrics"]["validation_test_gap"]["value"] is None

    leaked = evaluate_records([_base_record(test_accessed=True)])
    assert leaked["evaluation_status"] == "invalid_test_access"

    with tempfile.TemporaryDirectory() as temporary_directory:
        registry = Path(temporary_directory) / "experiment_registry.jsonl"
        registry.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        loaded = load_registry(registry)
        assert len(loaded) == 3
        invalid = _base_record(experiment_id="invalid", persistence_mae=0)
        registry.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
        try:
            load_registry(registry)
        except AutoResearchEvaluationInvalid:
            pass
        else:
            raise AssertionError("persistence_mae=0 应当被拒绝")
    print("self-test passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="eval_v1.1 第二层自主研究评测器")
    parser.add_argument("--registry", help="调度器生成的 experiment_registry.jsonl")
    parser.add_argument("--output", default="auto_research_metrics_v1.1.json")
    parser.add_argument(
        "--final-candidate-id",
        help="冻结并进行最终 test 认证的 candidate_id；没有正式 test 时省略",
    )
    parser.add_argument("--spec", default=str(DEFAULT_SPEC_PATH))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_test()
        return 0
    if not args.registry:
        parser.error("评测时必须提供 --registry")
    try:
        result = evaluate_registry(
            args.registry,
            final_candidate_id=args.final_candidate_id,
            spec_path=args.spec,
        )
    except AutoResearchEvaluationInvalid as error:
        invalid_result = {
            "evaluator_version": EVALUATOR_VERSION,
            "evaluation_status": "invalid",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        _write_json(Path(args.output), invalid_result)
        print(json.dumps(invalid_result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    _write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
