from __future__ import annotations

import json
from typing import Any


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _delta_pp(first: float | None, second: float | None) -> str:
    if first is None or second is None:
        return "N/A"
    return f"{(first - second) * 100:+.1f} pp"


def render_test_report(audit: dict[str, Any], metrics: dict[str, Any], model_run: dict[str, Any] | None = None) -> str:
    shortcut = audit["shortcut_baselines"]
    model_run_status = (model_run or {}).get("status", metrics["status"])
    if audit["errors"]:
        recommendation = "BLOCKED：先修复结构或资产错误。"
    elif audit["warnings"]:
        recommendation = "REVIEW_REQUIRED：先复核样本级警告。"
    else:
        recommendation = "PASS_FOR_EXPLORATORY_DRY_RUN：可以运行主流模型并报告描述性结果。"
    lines = [
        "# Dry Run 测试结果报告",
        "",
        "> 结果标签：`exploratory_dry_run`。本报告验证数据和评测链路，不构成正式 Pilot、模型排名或统计推断结论。",
        "",
        "## 1. 执行摘要",
        "",
        f"- 数据预检：**{audit['status']}**",
        f"- 模型运行：**{model_run_status}**；评分产物：**{metrics['status']}**",
        f"- FQ2 捷径风险：**{shortcut['overall_risk']}（KNOWN_LIMITATION，不阻断本轮调用）**",
        f"- 当前建议：**{recommendation}**",
        "",
        "## 2. 数据与题组",
        "",
        f"- {audit['samples']} 条样本、{audit['relation_groups']} 个完整题组；FQ1/FQ2 各 {audit['groups_by_type'].get('FQ1', 0)}/{audit['groups_by_type'].get('FQ2', 0)} 组。",
        f"- 实际引用图片：{audit['referenced_assets']}；物理图片：{audit['physical_assets']}。",
        f"- 质量状态：`{json.dumps(audit['quality_status'], ensure_ascii=False)}`。",
        f"- 结构错误：{len(audit['errors'])}；样本级警告：{len(audit['warnings'])}。",
        "",
        "## 3. 已知有效性限制",
        "",
        f"- FQ2 三条件图片数分布：`{json.dumps(audit['image_count_by_fq2_condition'], ensure_ascii=False)}`。",
        f"- 图片数量两两重叠系数：`{json.dumps(audit['image_count_overlap_coefficients'], ensure_ascii=False)}`。",
        "- count-only 与 meta-only 的完整结果保存在 `dataset_audit.json/.md`，只用于捷径诊断，不计入能力指标。",
    ]
    for limitation in audit.get("known_limitations", []):
        lines.append(f"- {limitation}")

    lines.extend(["", "## 4. 模型主要结果", ""])
    if not metrics.get("models"):
        run_message = (model_run or {}).get("message", "没有可评分的模型预测文件。")
        lines.extend([
            f"- 模型结果：`NOT_RUN`。{run_message}",
            "- 配置 API key 并完成 all-images 调用后，脚本才会填写以下模型指标。",
        ])
    else:
        lines.extend([
            "| 模型 | 逐题准确率 | 有效响应 | FQ1-GEM | NTS | DIS |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for model in metrics["models"]:
            result = model.get("results", {}).get("all-images")
            if not result:
                continue
            main = result["main_metrics"]
            coverage = result["coverage"]
            lines.append(
                f"| {model['model_id']} | {_fmt(result['item_accuracy'])} | "
                f"{coverage['valid_answers']}/{coverage['expected']} | {_fmt(main['FQ1_GEM']['value'])} | "
                f"{_fmt(main['NTS']['value'])} | {_fmt(main['DIS']['value'])} |"
            )

        for model in metrics["models"]:
            result = model.get("results", {}).get("all-images")
            if not result:
                continue
            lines.extend(["", f"### {model['model_id']} 条件分解", ""])
            frames = result["fq1_by_reference_frame"]
            conditions = result["fq2_by_view_condition"]
            lines.extend([
                "| 分解项 | n | 准确率 |",
                "|---|---:|---:|",
                f"| FQ1 camera | {frames['camera']['n']} | {_fmt(frames['camera']['accuracy'])} |",
                f"| FQ1 object-intrinsic | {frames['object_intrinsic']['n']} | {_fmt(frames['object_intrinsic']['accuracy'])} |",
                f"| FQ1 fixed-scene frame | {frames['world']['n']} | {_fmt(frames['world']['accuracy'])} |",
                f"| FQ2 one valid MSS | {conditions['one_valid_mss']['n']} | {_fmt(conditions['one_valid_mss']['accuracy'])} |",
                f"| FQ2 remove required view | {conditions['remove_required_view']['n']} | {_fmt(conditions['remove_required_view']['accuracy'])} |",
                f"| FQ2 hard distractor added | {conditions['hard_distractor_added']['n']} | {_fmt(conditions['hard_distractor_added']['accuracy'])} |",
            ])

    baseline_models = [
        model for model in metrics.get("models", [])
        if {"all-images", "text-only", "single-image"}.issubset(model.get("results", {}))
    ]
    lines.extend(["", "## 5. 输入模式 Baseline 对比", ""])
    if not baseline_models:
        lines.append("- text-only 或 single-image 尚未完成，当前无法比较真实多图依赖。")
    else:
        lines.extend([
            "| 模型 | 模式 | 逐题准确率 | 有效响应 | FQ1-GEM | NTS | DIS |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for model in baseline_models:
            for mode in ("all-images", "text-only", "single-image"):
                result = model["results"][mode]
                main = result["main_metrics"]
                coverage = result["coverage"]
                lines.append(
                    f"| {model['model_id']} | {mode} | {_fmt(result['item_accuracy'])} | "
                    f"{coverage['valid_answers']}/{coverage['expected']} | "
                    f"{_fmt(main['FQ1_GEM']['value'])} | {_fmt(main['NTS']['value'])} | "
                    f"{_fmt(main['DIS']['value'])} |"
                )
        lines.extend(["", "### 逐题准确率增益", "", "| 模型 | all-images − text-only | all-images − single-image |", "|---|---:|---:|"])
        for model in baseline_models:
            results = model["results"]
            lines.append(
                f"| {model['model_id']} | "
                f"{_delta_pp(results['all-images']['item_accuracy'], results['text-only']['item_accuracy'])} | "
                f"{_delta_pp(results['all-images']['item_accuracy'], results['single-image']['item_accuracy'])} |"
            )
        lines.extend([
            "",
            "- 正值表示完整多图输入优于对应 Baseline；负值表示该模型在本次小样本运行中未从完整图集获得逐题准确率收益。",
            "- text-only 的拒答、无答案或非法响应按预注册规则计为错误，并同时反映在有效响应率中。",
            "- single-image 由冻结种子和题组/图片哈希确定性选择，不针对模型或正确答案挑图。",
        ])

    lines.extend([
        "",
        "## 6. 解释边界",
        "",
        "- FQ1-GEM、NTS、DIS 分别报告，不合成为总分；三者合取条件和随机期望不同。",
        "- `relation_group_id` 是本轮行为观测单位；不计算置信区间、显著性检验、功效或最小可检测差异。",
        "- 本轮暂不要求人类评测；没有人类结果不会阻断 Dry Run。",
        "- FQ2 图片数量捷径尚未消除，因此模型结果只能用于链路验证和初步错误观察。",
        "",
        "## 7. 当前结论",
        "",
        f"{recommendation} 当前结果不得用于声称稳定模型排名、显著人机差距或正式数据集有效性。",
        "",
    ])
    return "\n".join(lines)
