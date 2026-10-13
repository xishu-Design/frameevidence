from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from .common import read_jsonl, safe_path_component, write_json, write_jsonl

SYSTEM_PROMPT = "你正在参加盲化四选一评测。只能输出 A、B、C 或 D，不要解释。"
SUPPORTED_PROVIDERS = {
    "openai_compatible",
    "openai_responses",
    "anthropic_messages",
    "gemini_generate_content",
}
SUPPORTED_MODES = {"all-images", "text-only", "single-image"}
SUPPORTED_PROFILES = {"dry_run_descriptive", "formal_release"}
SUPPORTED_SHOT_COUNTS = {0, 1, 2, 3}
FEW_SHOT_SPLIT_STATUS_RANK = {
    "PROVISIONAL_NOT_FOR_PAPER": 0,
    "FROZEN_CANDIDATE": 1,
    "PASS": 2,
}
DEFAULT_API_KEY_ENV = {
    "openai_responses": "OPENAI_API_KEY",
    "anthropic_messages": "ANTHROPIC_API_KEY",
    "gemini_generate_content": "GEMINI_API_KEY",
}
DEFAULT_BASE_URL = {
    "openai_responses": "https://api.openai.com/v1",
    "anthropic_messages": "https://api.anthropic.com/v1",
    "gemini_generate_content": "https://generativelanguage.googleapis.com/v1beta",
}


def parse_answer(raw: str | None) -> tuple[str | None, str]:
    if raw is None:
        return None, "empty"
    text = str(raw).strip()
    if not text:
        return None, "empty"
    if text in {"A", "B", "C", "D"}:
        return text, "exact_letter"
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict) and value.get("answer") in {"A", "B", "C", "D"}:
        return str(value["answer"]), "strict_json"
    matches = re.findall(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text.upper())
    if len(set(matches)) == 1:
        return matches[0], "single_standalone_letter"
    return None, "ambiguous_or_missing"


def _provider(model: dict[str, Any]) -> str:
    return str(model.get("provider") or "openai_compatible")


def _api_key_env(model: dict[str, Any]) -> str:
    provider = _provider(model)
    return str(model.get("api_key_env", DEFAULT_API_KEY_ENV.get(provider, "")))


def _base_url(model: dict[str, Any]) -> str:
    base_url_env = str(model.get("base_url_env", "")).strip()
    env_value = os.environ.get(base_url_env, "") if base_url_env else ""
    return str(env_value or model.get("base_url") or DEFAULT_BASE_URL.get(_provider(model), "")).rstrip("/")


def _model_name(model: dict[str, Any]) -> str:
    """Return the API model name, optionally overridden without rewriting JSON."""
    model_env = str(model.get("model_env", "")).strip()
    env_value = os.environ.get(model_env, "").strip() if model_env else ""
    return env_value or str(model.get("model", "")).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_run_provenance(
    run_dir: Path,
    *,
    evaluation_profile: str | None = None,
) -> dict[str, Any]:
    """Build a content-addressed identity for the prepared evaluation input.

    The model configuration alone is not a sufficient resume key: the same
    endpoint can be run against a different dataset, split, prompt template,
    or prepared-input cache.  This record deliberately hashes both prepared
    JSONL files and the semantic fields from ``run_manifest.json``.  Formal
    runs fail closed when any required provenance is absent or inconsistent.
    """

    run_dir = run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    prepared_path = run_dir / "prepared" / "prepared_inputs.jsonl"
    mapping_path = run_dir / "private" / "evaluation_mapping.jsonl"
    formal = evaluation_profile == "formal_release"

    if not prepared_path.is_file():
        raise FileNotFoundError(f"缺少 prepared 输入：{prepared_path}")
    if formal and not manifest_path.is_file():
        raise RuntimeError(f"正式评测缺少 run_manifest.json：{manifest_path}")
    if formal and not mapping_path.is_file():
        raise RuntimeError(f"正式评测缺少 evaluation_mapping.jsonl：{mapping_path}")

    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, dict):
            raise RuntimeError(f"run manifest 必须是 JSON 对象：{manifest_path}")
        manifest = loaded
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        selection = {}

    payload: dict[str, Any] = {
        "schema_version": "frameevidence-run-provenance-v1",
        "evaluation_profile": manifest.get("evaluation_profile") or evaluation_profile,
        "result_label": manifest.get("result_label"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "seed": manifest.get("seed"),
        "prompt_template_sha256": manifest.get("prompt_template_sha256"),
        "evaluation_split": selection.get("evaluation_split") or manifest.get("evaluation_split"),
        "split_manifest_sha256": selection.get("split_manifest_sha256") or manifest.get("split_manifest_sha256"),
        "allowed_group_ids_sha256": selection.get("allowed_group_ids_sha256") or manifest.get("allowed_group_ids_sha256"),
        "selected_group_count": selection.get("selected_group_count"),
        "source_samples": manifest.get("source_samples"),
        "public_items": manifest.get("public_items"),
        "ordering": manifest.get("ordering"),
        "single_image_rule": manifest.get("single_image_rule"),
        "prepared_inputs_sha256": _sha256_file(prepared_path),
        "evaluation_mapping_sha256": _sha256_file(mapping_path) if mapping_path.is_file() else None,
    }

    if formal:
        required = (
            "dataset_fingerprint",
            "seed",
            "prompt_template_sha256",
            "evaluation_split",
            "split_manifest_sha256",
            "allowed_group_ids_sha256",
            "selected_group_count",
            "source_samples",
            "public_items",
            "ordering",
            "single_image_rule",
            "evaluation_mapping_sha256",
        )
        missing = [name for name in required if payload.get(name) in (None, "")]
        if missing:
            raise RuntimeError(f"正式评测运行溯源缺少字段：{missing}")
        if payload["evaluation_profile"] != "formal_release":
            raise RuntimeError(
                "run manifest evaluation_profile 不是 formal_release："
                f"{payload['evaluation_profile']!r}"
            )
        copied_split = run_dir / "manifests" / "split_manifest.csv"
        if not copied_split.is_file():
            raise RuntimeError(f"正式评测缺少冻结 split 副本：{copied_split}")
        copied_split_sha256 = _sha256_file(copied_split)
        if copied_split_sha256 != payload["split_manifest_sha256"]:
            raise RuntimeError(
                "冻结 split 副本与 run manifest 不一致："
                f"{copied_split_sha256} != {payload['split_manifest_sha256']}"
            )
        payload["copied_split_manifest_sha256"] = copied_split_sha256

    payload["run_provenance_sha256"] = _canonical_sha256(payload)
    return payload


def _few_shot_status_variant_hashes(manifest_path: Path) -> dict[str, str]:
    """Map hashes of status-only manifest variants to their split status.

    The fixed-demo builder writes deterministic, two-space-indented JSON.  A
    split can later move from provisional to frozen/PASS without changing any
    demonstration shown to the model.  Reconstruct both common newline styles
    so that a Windows-generated selection can be resumed after a Linux rebuild
    (or vice versa).  A hash match proves that *only* ``split_status`` changed.
    """
    raw = manifest_path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    manifest = json.loads(raw.decode("utf-8-sig"))
    current_status = str(manifest.get("split_status", ""))
    if current_status not in FEW_SHOT_SPLIT_STATUS_RANK:
        return {}

    variants: dict[str, str] = {}
    for status in FEW_SHOT_SPLIT_STATUS_RANK:
        candidate = dict(manifest)
        candidate["split_status"] = status
        text = json.dumps(candidate, ensure_ascii=False, indent=2) + "\n"
        for newline in ("\n", "\r\n"):
            encoded = text.replace("\n", newline).encode("utf-8")
            if has_bom:
                encoded = b"\xef\xbb\xbf" + encoded
            variants[hashlib.sha256(encoded).hexdigest()] = status
    return variants


def _few_shot_manifest_path(model: dict[str, Any], config_dir: Path) -> Path | None:
    value = str(model.get("few_shot_manifest", "")).strip()
    if not value:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def _load_few_shot_context(model: dict[str, Any], config_dir: Path) -> dict[str, Any] | None:
    shot_count = int(model.get("shot_count", 0))
    if shot_count == 0:
        return None
    manifest_path = _few_shot_manifest_path(model, config_dir)
    if manifest_path is None or not manifest_path.is_file():
        raise ValueError(f"few-shot manifest 不存在：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if str(manifest.get("source_split", "")) != "train":
        raise ValueError("few-shot manifest 的 source_split 必须为 train")
    leakage_audit = manifest.get("leakage_audit")
    if not isinstance(leakage_audit, dict) or leakage_audit.get("status") != "PASS":
        raise ValueError("few-shot manifest 缺少 PASS 的 leakage_audit")
    pools = manifest.get("pools")
    if not isinstance(pools, dict):
        raise ValueError(f"few-shot manifest 缺少 pools：{manifest_path}")
    normalized: dict[str, list[dict[str, Any]]] = {}
    for question_type in ("FQ1", "FQ2"):
        examples = pools.get(question_type)
        if not isinstance(examples, list) or len(examples) < shot_count:
            raise ValueError(
                f"few-shot manifest 的 {question_type} 至少需要 {shot_count} 个示例"
            )
        normalized_examples: list[dict[str, Any]] = []
        for index, example in enumerate(examples[:shot_count], 1):
            if not isinstance(example, dict):
                raise ValueError(f"{question_type} 示例 {index} 不是对象")
            prompt = str(example.get("prompt", "")).strip()
            answer = str(example.get("answer", "")).strip()
            refs_by_mode = example.get("image_refs_by_mode")
            if not prompt or answer not in {"A", "B", "C", "D"}:
                raise ValueError(f"{question_type} 示例 {index} 的 prompt/answer 非法")
            if not isinstance(refs_by_mode, dict):
                raise ValueError(f"{question_type} 示例 {index} 缺少 image_refs_by_mode")
            image_paths_by_mode: dict[str, list[Path]] = {}
            for mode in SUPPORTED_MODES:
                refs = refs_by_mode.get(mode)
                if not isinstance(refs, list):
                    raise ValueError(f"{question_type} 示例 {index} 缺少 {mode} 图片列表")
                paths = [(manifest_path.parent / str(ref)).resolve() for ref in refs]
                missing = [str(path) for path in paths if not path.is_file()]
                if missing:
                    raise ValueError(
                        f"{question_type} 示例 {index} 的图片不存在：{missing[:2]}"
                    )
                image_paths_by_mode[mode] = paths
            normalized_examples.append({
                "sample_id": str(example.get("sample_id", "")),
                "relation_group_id": str(example.get("relation_group_id", "")),
                "prompt": prompt,
                "answer": answer,
                "image_paths_by_mode": image_paths_by_mode,
            })
        normalized[question_type] = normalized_examples
    return {
        "path": manifest_path,
        "sha256": _sha256_file(manifest_path),
        "protocol_id": str(manifest.get("protocol_id", "")),
        "source_split": str(manifest.get("source_split", "")),
        "split_status": str(manifest.get("split_status", "")),
        "pools": normalized,
    }


def _data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _base64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _response_text(provider: str, payload: dict[str, Any]) -> str:
    if provider == "openai_compatible":
        choices = payload.get("choices") or []
        if not choices:
            raise ValueError("响应缺少 choices")
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        raise ValueError("响应缺少 message.content")
    if provider == "openai_responses":
        if isinstance(payload.get("output_text"), str):
            return str(payload["output_text"])
        texts: list[str] = []
        for output in payload.get("output", []) or []:
            for part in output.get("content", []) or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(str(part["text"]))
        if texts:
            return "".join(texts)
        raise ValueError("响应缺少 output_text")
    if provider == "anthropic_messages":
        texts = [
            str(part.get("text", ""))
            for part in payload.get("content", []) or []
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        if texts:
            return "".join(texts)
        raise ValueError("响应缺少 content.text")
    if provider == "gemini_generate_content":
        candidates = payload.get("candidates") or []
        if not candidates:
            raise ValueError("响应缺少 candidates")
        parts = candidates[0].get("content", {}).get("parts", []) or []
        texts = [str(part.get("text", "")) for part in parts if isinstance(part, dict) and isinstance(part.get("text"), str)]
        if texts:
            return "".join(texts)
        raise ValueError("响应缺少 candidates.content.parts.text")
    raise ValueError(f"不支持的 provider：{provider}")


def _few_shot_examples_for_item(
    item: dict[str, Any],
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    shot_count = int(model.get("shot_count", 0))
    if shot_count == 0:
        return []
    context = model.get("_few_shot_context")
    if not isinstance(context, dict):
        raise ValueError("few-shot runtime context 未加载")
    question_type = str(item.get("_question_type", ""))
    if question_type not in {"FQ1", "FQ2"}:
        raise ValueError("few-shot 路由缺少目标题的 question_type")
    mode = str(item.get("evaluation_mode", ""))
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"few-shot 目标 evaluation_mode 非法：{mode}")
    examples = context["pools"][question_type][:shot_count]
    return [
        {
            **example,
            "image_paths": example["image_paths_by_mode"][mode],
        }
        for example in examples
    ]


def _openai_content(
    item: dict[str, Any],
    model: dict[str, Any],
    prepared_root: Path,
    *,
    responses_api: bool,
) -> list[dict[str, Any]]:
    """Build one MirrorQA-style prompt with fixed labelled demonstrations."""
    text_type = "input_text" if responses_api else "text"
    content: list[dict[str, Any]] = []
    examples = _few_shot_examples_for_item(item, model)
    if examples:
        content.append({
            "type": text_type,
            "text": (
                f"下面给出 {len(examples)} 个已标注示例。每个示例的图片顺序与题目中的图一、图二等编号一致。"
                "学习示例后，只回答最后一道目标题。\n"
            ),
        })
        for index, example in enumerate(examples, 1):
            content.append({
                "type": text_type,
                "text": f"示例 {index}\n{example['prompt']}",
            })
            for path in example["image_paths"]:
                if responses_api:
                    content.append({
                        "type": "input_image",
                        "image_url": _data_uri(path),
                        "detail": model.get("image_detail", "high"),
                    })
                else:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": _data_uri(path),
                            "detail": model.get("image_detail", "high"),
                        },
                    })
            content.append({
                "type": text_type,
                "text": f"示例 {index} 的正确输出：{example['answer']}\n",
            })
        content.append({"type": text_type, "text": "现在回答目标题："})

    content.append({"type": text_type, "text": item["prompt"]})
    for path in [prepared_root / str(value) for value in item.get("image_refs", [])]:
        if responses_api:
            content.append({
                "type": "input_image",
                "image_url": _data_uri(path),
                "detail": model.get("image_detail", "high"),
            })
        else:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": _data_uri(path),
                    "detail": model.get("image_detail", "high"),
                },
            })
    return content


def _request_spec(
    item: dict[str, Any],
    model: dict[str, Any],
    prepared_root: Path,
    api_key: str | None,
) -> tuple[str, dict[str, str], dict[str, Any], str]:
    provider = _provider(model)
    base_url = _base_url(model)
    max_tokens = int(model.get("max_tokens", 32))
    temperature = model.get("temperature", 0)
    headers: dict[str, str] = {"Content-Type": "application/json"}

    if provider == "openai_compatible":
        content = _openai_content(item, model, prepared_root, responses_api=False)
        body: dict[str, Any] = {
            "model": _model_name(model),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if model.get("reasoning_effort") is not None:
            body["reasoning_effort"] = str(model["reasoning_effort"])
        chat_template_kwargs = model.get("chat_template_kwargs")
        if chat_template_kwargs is not None:
            if not isinstance(chat_template_kwargs, dict):
                raise ValueError("chat_template_kwargs must be a JSON object")
            # Copy the mapping so request construction never mutates the frozen
            # model configuration.  vLLM exposes this field on its OpenAI-
            # compatible Chat Completions endpoint.
            body["chat_template_kwargs"] = dict(chat_template_kwargs)
        if model.get("seed") is not None:
            body["seed"] = model["seed"]
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        endpoint = base_url + str(model.get("chat_completions_path", "/chat/completions"))
    elif provider == "openai_responses":
        content = _openai_content(item, model, prepared_root, responses_api=True)
        body = {
            "model": _model_name(model),
            "instructions": SYSTEM_PROMPT,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if model.get("reasoning_effort") is not None:
            body["reasoning"] = {"effort": str(model["reasoning_effort"])}
        headers["Authorization"] = f"Bearer {api_key}"
        endpoint = base_url + str(model.get("responses_path", "/responses"))
    elif provider == "anthropic_messages":
        image_paths = [prepared_root / str(value) for value in item.get("image_refs", [])]
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": _base64_image(path)},
            }
            for path in image_paths
        ]
        content.append({"type": "text", "text": item["prompt"]})
        body = {
            "model": _model_name(model),
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        headers["x-api-key"] = str(api_key)
        headers["anthropic-version"] = str(model.get("anthropic_version", "2023-06-01"))
        endpoint = base_url + str(model.get("messages_path", "/messages"))
    elif provider == "gemini_generate_content":
        image_paths = [prepared_root / str(value) for value in item.get("image_refs", [])]
        parts: list[dict[str, Any]] = [
            {"inline_data": {"mime_type": "image/jpeg", "data": _base64_image(path)}}
            for path in image_paths
        ]
        parts.append({"text": item["prompt"]})
        generation_config: dict[str, Any] = {"maxOutputTokens": max_tokens}
        if temperature is not None:
            generation_config["temperature"] = temperature
        body = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        headers["x-goog-api-key"] = str(api_key)
        path_template = str(model.get("generate_content_path", "/models/{model}:generateContent"))
        endpoint = base_url + path_template.format(model=_model_name(model))
    else:
        raise ValueError(f"不支持的 provider：{provider}")

    for header, env_name in model.get("header_env", {}).items():
        value = os.environ.get(str(env_name))
        if value:
            headers[str(header)] = value
    return endpoint, headers, body, provider


def _request_one(
    item: dict[str, Any],
    model: dict[str, Any],
    prepared_root: Path,
    api_key: str | None,
) -> tuple[str, dict[str, Any], int]:
    endpoint, headers, body, provider = _request_spec(item, model, prepared_root, api_key)
    attempts = int(model.get("max_attempts", 3))
    timeout = float(model.get("timeout_seconds", 120))
    backoff = float(model.get("retry_backoff_seconds", 1))
    last_error: Exception | None = None
    started = time.perf_counter()
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(endpoint, headers=headers, json=body, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"HTTP {response.status_code}: {response.text[:500]}", response=response)
            if not response.ok:
                preview = response.text[:500]
                if api_key:
                    preview = preview.replace(api_key, "<redacted-api-key>")
                if endpoint:
                    preview = preview.replace(endpoint, "<redacted-endpoint>")
                raise RuntimeError(
                    f"HTTP {response.status_code}: {preview.replace(chr(10), ' ')}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                preview = response.text[:300].replace(api_key, "<redacted-api-key>") if api_key else response.text[:300]
                content_type = response.headers.get("content-type", "<missing>")
                raise RuntimeError(
                    f"HTTP {response.status_code} 返回非 JSON（Content-Type={content_type}）："
                    f"{preview.replace(chr(10), ' ')}"
                ) from exc
            latency_ms = round((time.perf_counter() - started) * 1000)
            usage = payload.get("usage") or payload.get("usageMetadata") or {}
            return _response_text(provider, payload), usage, latency_ms
        except Exception as exc:  # noqa: BLE001 - errors are recorded per item
            last_error = exc
            if attempt < attempts and backoff > 0:
                time.sleep(min(backoff * (2 ** (attempt - 1)), 8))
    raise RuntimeError(str(last_error))


def _load_existing(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.is_file() else []


def _select_group_subset(
    items: list[dict[str, Any]],
    mapping: list[dict[str, Any]],
    limit_groups: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    if limit_groups <= 0:
        raise ValueError("--limit-groups 必须大于 0")
    mapping_by_eval = {str(row["eval_id"]): row for row in mapping}
    group_order: list[str] = []
    group_type: dict[str, str] = {}
    for item in items:
        row = mapping_by_eval.get(str(item.get("eval_id")))
        if not row:
            continue
        group_id = str(row["relation_group_id"])
        if group_id not in group_type:
            group_order.append(group_id)
            group_type[group_id] = str(row["question_type"])
    if limit_groups > len(group_order):
        raise ValueError(f"--limit-groups={limit_groups} 超过可用完整题组数 {len(group_order)}")

    available_by_type = {
        question_type: [group for group in group_order if group_type[group] == question_type]
        for question_type in ("FQ1", "FQ2")
    }
    selected_groups: list[str] = []
    if all(available_by_type.values()):
        fq1_target = min(len(available_by_type["FQ1"]), (limit_groups + 1) // 2)
        fq2_target = min(len(available_by_type["FQ2"]), limit_groups - fq1_target)
        remaining = limit_groups - fq1_target - fq2_target
        if remaining:
            fq1_extra = min(remaining, len(available_by_type["FQ1"]) - fq1_target)
            fq1_target += fq1_extra
            remaining -= fq1_extra
            fq2_target += min(remaining, len(available_by_type["FQ2"]) - fq2_target)
        selected_set = set(available_by_type["FQ1"][:fq1_target] + available_by_type["FQ2"][:fq2_target])
        selected_groups = [group for group in group_order if group in selected_set]
    else:
        selected_groups = group_order[:limit_groups]

    selected_set = set(selected_groups)
    selected_items = [
        item
        for item in items
        if (
            str(item.get("eval_id")) in mapping_by_eval
            and str(mapping_by_eval[str(item["eval_id"])]["relation_group_id"]) in selected_set
        )
    ]
    breakdown = {
        question_type: sum(group_type[group] == question_type for group in selected_groups)
        for question_type in ("FQ1", "FQ2")
    }
    return selected_items, selected_groups, breakdown


def _evaluate_item(
    item: dict[str, Any],
    model: dict[str, Any],
    prepared_root: Path,
    api_key: str | None,
    model_id: str,
    preliminary: bool = True,
    result_label: str = "exploratory_dry_run",
) -> dict[str, Any]:
    try:
        raw, usage, latency_ms = _request_one(item, model, prepared_root, api_key)
        answer, parse_status = parse_answer(raw)
        return {
            "eval_id": item["eval_id"],
            "model_id": model_id,
            "provider": _provider(model),
            "shot_count": int(model.get("shot_count", 0)),
            "evaluation_mode": item["evaluation_mode"],
            "status": "OK" if answer else "PARSE_ERROR",
            "parsed_answer": answer,
            "parse_status": parse_status,
            "raw_response": raw,
            "latency_ms": latency_ms,
            "usage": usage,
            "preliminary": preliminary,
            "result_label": result_label,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "eval_id": item["eval_id"],
            "model_id": model_id,
            "provider": _provider(model),
            "shot_count": int(model.get("shot_count", 0)),
            "evaluation_mode": item["evaluation_mode"],
            "status": "CALL_ERROR",
            "parsed_answer": None,
            "parse_status": "call_error",
            "raw_response": "",
            "latency_ms": None,
            "usage": {},
            "error": str(exc),
            "preliminary": preliminary,
            "result_label": result_label,
        }


def _model_errors(
    model: dict[str, Any],
    require_env: bool = True,
    config_dir: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    provider = _provider(model)
    if provider not in SUPPORTED_PROVIDERS:
        errors.append(f"provider 不支持：{provider}")
    if not str(model.get("model_id", "")).strip():
        errors.append("缺少 model_id")
    if not _model_name(model):
        errors.append("缺少 model")
    base_url = _base_url(model)
    if not base_url.startswith(("http://", "https://")):
        errors.append("base_url 必须是 http(s) URL")
    modes = set(model.get("evaluation_modes", ["all-images"]))
    if not modes or not modes.issubset(SUPPORTED_MODES):
        errors.append(f"evaluation_modes 非法：{sorted(modes)}")
    concurrency = int(model.get("concurrency", 1))
    if concurrency < 1 or concurrency > 8:
        errors.append("concurrency 必须在 1--8 之间")
    chat_template_kwargs = model.get("chat_template_kwargs")
    if chat_template_kwargs is not None:
        if provider != "openai_compatible":
            errors.append("chat_template_kwargs 当前仅支持 openai_compatible")
        if not isinstance(chat_template_kwargs, dict):
            errors.append("chat_template_kwargs 必须是 JSON 对象")
    api_env = _api_key_env(model)
    if provider != "openai_compatible" and not api_env:
        errors.append("原生云端 provider 必须配置 api_key_env")
    if require_env and api_env and not os.environ.get(api_env):
        errors.append(f"缺少环境变量 {api_env}")
    try:
        shot_count = int(model.get("shot_count", 0))
    except (TypeError, ValueError):
        shot_count = -1
    if shot_count not in SUPPORTED_SHOT_COUNTS:
        errors.append(f"shot_count 必须属于 {sorted(SUPPORTED_SHOT_COUNTS)}")
    elif shot_count > 0:
        if provider not in {"openai_compatible", "openai_responses"}:
            errors.append("few-shot 当前仅支持 openai_compatible/openai_responses")
        if config_dir is None:
            if not str(model.get("few_shot_manifest", "")).strip():
                errors.append("shot_count > 0 时必须配置 few_shot_manifest")
        else:
            try:
                _load_few_shot_context(model, config_dir)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(str(exc))
    return errors


def check_model_config(
    config_path: Path,
    model_ids: set[str] | None = None,
    require_env: bool = True,
) -> dict[str, Any]:
    if not config_path.is_file():
        return {"status": "CONFIG_ERROR", "config": str(config_path), "errors": ["配置文件不存在"], "models": {}}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "CONFIG_ERROR", "config": str(config_path), "errors": [str(exc)], "models": {}}
    profile = str(config.get("evaluation_profile", "dry_run_descriptive"))
    top_errors = [] if profile in SUPPORTED_PROFILES else [
        f"evaluation_profile 必须属于 {sorted(SUPPORTED_PROFILES)}，实际为 {profile}"
    ]
    all_models = config.get("models", [])
    if not isinstance(all_models, list):
        return {"status": "CONFIG_ERROR", "config": str(config_path), "errors": top_errors + ["models 必须是数组"], "models": {}}
    ids = [str(model.get("model_id", "")) for model in all_models if isinstance(model, dict)]
    duplicates = sorted({value for value in ids if value and ids.count(value) > 1})
    if duplicates:
        top_errors.append(f"model_id 重复：{duplicates}")
    known_ids = set(ids)
    if model_ids and not model_ids.issubset(known_ids):
        top_errors.append(f"请求了不存在的模型：{sorted(model_ids - known_ids)}")
    selected = [
        model for model in all_models
        if isinstance(model, dict)
        and bool(model.get("enabled", True))
        and (not model_ids or str(model.get("model_id")) in model_ids)
    ]
    model_results: dict[str, Any] = {}
    for model in selected:
        model_id = str(model.get("model_id", "<missing>"))
        errors = _model_errors(
            model,
            require_env=require_env,
            config_dir=config_path.resolve().parent,
        )
        model_results[model_id] = {
            "status": "PASS" if not errors else "CONFIG_ERROR",
            "provider": _provider(model),
            "model": _model_name(model),
            "api_key_env": _api_key_env(model) or None,
            "api_key_present": bool(_api_key_env(model) and os.environ.get(_api_key_env(model))),
            "base_url_env": model.get("base_url_env"),
            "evaluation_modes": model.get("evaluation_modes", ["all-images"]),
            "shot_count": int(model.get("shot_count", 0)),
            "few_shot_manifest": str(
                _few_shot_manifest_path(model, config_path.resolve().parent)
            ) if int(model.get("shot_count", 0)) > 0 else None,
            "errors": errors,
        }
    if not selected:
        top_errors.append("没有启用且被选中的模型")
    status = "PASS" if not top_errors and all(value["status"] == "PASS" for value in model_results.values()) else "CONFIG_ERROR"
    return {
        "status": status,
        "evaluation_profile": profile,
        "config": str(config_path.resolve()),
        "errors": top_errors,
        "models": model_results,
    }


def _protocol_fingerprint(
    model: dict[str, Any],
    *,
    few_shot_manifest_sha256: str | None = None,
    run_provenance_sha256: str | None = None,
) -> str:
    context = model.get("_few_shot_context")
    manifest_sha256 = (
        few_shot_manifest_sha256
        if few_shot_manifest_sha256 is not None
        else context.get("sha256") if isinstance(context, dict) else None
    )
    payload = {
        "provider": _provider(model),
        "base_url": _base_url(model),
        "model": _model_name(model),
        "system_prompt": SYSTEM_PROMPT,
        "shot_count": int(model.get("shot_count", 0)),
        "few_shot_manifest_sha256": manifest_sha256,
        "few_shot_protocol_id": (
            context.get("protocol_id") if isinstance(context, dict) else None
        ),
        "evaluation_modes": sorted(model.get("evaluation_modes", ["all-images"])),
        "temperature": model.get("temperature", 0),
        "reasoning_effort": model.get("reasoning_effort"),
        "max_tokens": int(model.get("max_tokens", 32)),
        "image_detail": model.get("image_detail", "high"),
    }
    # Preserve legacy fingerprints when this optional setting is absent, while
    # binding every explicitly configured chat-template mode to a new protocol.
    if model.get("chat_template_kwargs") is not None:
        payload["chat_template_kwargs"] = model.get("chat_template_kwargs")
    # Omit this key for legacy-fingerprint reconstruction.  New fingerprints
    # always include it, binding a model protocol to one exact prepared run.
    if run_provenance_sha256 is not None:
        payload["run_provenance_sha256"] = run_provenance_sha256
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_models(
    run_dir: Path,
    config_path: Path,
    audit: dict[str, Any],
    model_ids: set[str] | None = None,
    modes: set[str] | None = None,
    limit: int | None = None,
    retry_failed: bool = False,
    limit_groups: int | None = None,
    evaluation_profile: str | None = None,
    result_label: str | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    profile = evaluation_profile or str(config.get("evaluation_profile", "dry_run_descriptive"))
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"不支持的 evaluation_profile：{profile}")
    label = result_label or ("formal_release" if profile == "formal_release" else "exploratory_dry_run")
    preliminary = profile != "formal_release"
    requested_modes = modes or {"all-images"}
    if not requested_modes.issubset(SUPPORTED_MODES):
        raise ValueError(f"不支持的评测模式：{sorted(requested_modes - SUPPORTED_MODES)}")
    if limit is not None and limit <= 0:
        raise ValueError("--limit 必须大于 0")
    if limit_groups is not None and limit_groups <= 0:
        raise ValueError("--limit-groups 必须大于 0")
    if limit is not None and limit_groups is not None:
        raise ValueError("--limit 与 --limit-groups 不能同时使用")
    models = [
        model for model in config.get("models", [])
        if isinstance(model, dict)
        and bool(model.get("enabled", True))
        and (not model_ids or str(model.get("model_id")) in model_ids)
    ]
    summary: dict[str, Any] = {
        "status": "NOT_RUN",
        "evaluation_profile": profile,
        "result_label": label,
        "shortcut_risk": audit.get("shortcut_baselines", {}).get("overall_risk", "NOT_RUN"),
        "shortcut_policy": "warn",
        "known_limitations": audit.get("known_limitations", []),
        "requested_modes": sorted(requested_modes),
        "limit": limit,
        "limit_groups": limit_groups,
        "retry_failed": retry_failed,
        "models": {},
    }
    if not models:
        summary["message"] = "没有启用且被选中的模型；未调用任何 API。"
        write_json(run_dir / "model_run_summary.json", summary)
        return summary

    items = read_jsonl(run_dir / "prepared" / "prepared_inputs.jsonl")
    item_eval_ids = [str(item.get("eval_id", "")) for item in items]
    if not all(item_eval_ids) or len(set(item_eval_ids)) != len(item_eval_ids):
        raise RuntimeError("prepared_inputs.jsonl 含空或重复 eval_id，拒绝运行")
    prepared_by_eval = {str(item["eval_id"]): item for item in items}
    run_provenance = build_run_provenance(
        run_dir,
        evaluation_profile=profile,
    )
    run_provenance_sha256 = str(run_provenance["run_provenance_sha256"])
    uses_few_shot = any(int(model.get("shot_count", 0)) > 0 for model in models)
    private_mapping = (
        read_jsonl(run_dir / "private" / "evaluation_mapping.jsonl")
        if limit_groups is not None or uses_few_shot
        else []
    )
    private_by_eval = {
        str(row.get("eval_id")): row
        for row in private_mapping
        if row.get("eval_id")
    }
    prepared_root = run_dir / "prepared"
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    any_success = False
    any_error = False
    for model in models:
        model_id = str(model.get("model_id", ""))
        config_errors = _model_errors(
            model,
            require_env=True,
            config_dir=config_path.resolve().parent,
        )
        if config_errors:
            summary["models"][model_id or "<missing>"] = {"status": "CONFIG_ERROR", "errors": config_errors}
            any_error = True
            continue
        runtime_model = dict(model)
        runtime_model["_few_shot_context"] = _load_few_shot_context(
            model,
            config_path.resolve().parent,
        )
        shot_count = int(runtime_model.get("shot_count", 0))
        protocol_sha256 = _protocol_fingerprint(
            runtime_model,
            run_provenance_sha256=run_provenance_sha256,
        )
        legacy_protocol_sha256 = _protocol_fingerprint(runtime_model)
        safe_id = safe_path_component(model_id)
        output_path = predictions_dir / f"{safe_id}.jsonl"
        selection_path = predictions_dir / f"{safe_id}.selection.json"
        existing_rows = _load_existing(output_path)
        previous_selection: dict[str, Any] | None = None
        protocol_migration: dict[str, Any] | None = None
        if existing_rows:
            if not selection_path.is_file():
                raise RuntimeError(
                    f"{model_id} 已有预测但缺少协议选择文件；"
                    "无法证明数据、split 与提示词来源一致，请使用新的 model_id 或移走旧预测。"
                )
            else:
                previous_selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
                if not isinstance(previous_selection, dict):
                    raise RuntimeError(f"{model_id} 的协议选择文件必须是 JSON 对象")
                previous_ids = previous_selection.get("selected_eval_ids")
                if not isinstance(previous_ids, list):
                    raise RuntimeError(f"{model_id} 的旧选择文件缺少 selected_eval_ids")
                previous_ids = [str(value) for value in previous_ids]
                if not all(previous_ids) or len(set(previous_ids)) != len(previous_ids):
                    raise RuntimeError(f"{model_id} 的旧选择文件含空或重复 eval_id")
                existing_ids = [str(row.get("eval_id", "")) for row in existing_rows]
                if not all(existing_ids) or len(set(existing_ids)) != len(existing_ids):
                    raise RuntimeError(f"{model_id} 的旧预测含空或重复 eval_id")
                if not set(existing_ids).issubset(set(previous_ids)):
                    raise RuntimeError(f"{model_id} 的旧预测不属于旧选择清单")
                for row in existing_rows:
                    eval_id = str(row["eval_id"])
                    prepared = prepared_by_eval.get(eval_id)
                    if prepared is None:
                        raise RuntimeError(f"{model_id} 的旧预测 eval_id 不属于当前 prepared：{eval_id}")
                    if str(row.get("model_id", "")) != model_id:
                        raise RuntimeError(f"{model_id} 的旧预测含错误 model_id：{eval_id}")
                    if str(row.get("evaluation_mode", "")) != str(prepared.get("evaluation_mode", "")):
                        raise RuntimeError(f"{model_id} 的旧预测 mode 与当前 prepared 不一致：{eval_id}")
                for eval_id in previous_ids:
                    if eval_id not in prepared_by_eval:
                        raise RuntimeError(f"{model_id} 的旧选择 eval_id 不属于当前 prepared：{eval_id}")
                expected_selection_fields = {
                    "model_id": model_id,
                    "api_model": _model_name(runtime_model),
                    "shot_count": shot_count,
                }
                for field, expected in expected_selection_fields.items():
                    if previous_selection.get(field) != expected:
                        raise RuntimeError(
                            f"{model_id} 的旧选择 {field}={previous_selection.get(field)!r}，"
                            f"当前为 {expected!r}"
                        )
                previous_protocol = previous_selection.get("protocol_sha256")
                previous_run_provenance = previous_selection.get("run_provenance_sha256")
                if previous_protocol == protocol_sha256:
                    if previous_run_provenance != run_provenance_sha256:
                        raise RuntimeError(
                            f"{model_id} 的协议指纹虽相同但运行溯源缺失或不一致；拒绝复用。"
                        )
                elif previous_protocol:
                    context = runtime_model.get("_few_shot_context")
                    previous_manifest_sha256 = str(
                        previous_selection.get("few_shot_manifest_sha256", "")
                    )
                    previous_protocol_id = str(
                        previous_selection.get("few_shot_protocol_id", "")
                    )
                    current_protocol_id = (
                        str(context.get("protocol_id", ""))
                        if isinstance(context, dict)
                        else ""
                    )
                    status_variants = (
                        _few_shot_status_variant_hashes(Path(context["path"]))
                        if isinstance(context, dict) and context.get("path")
                        else {}
                    )
                    previous_status = status_variants.get(previous_manifest_sha256)
                    current_status = (
                        str(context.get("split_status", ""))
                        if isinstance(context, dict)
                        else ""
                    )
                    previous_fingerprint = (
                        _protocol_fingerprint(
                            runtime_model,
                            few_shot_manifest_sha256=previous_manifest_sha256,
                            run_provenance_sha256=(
                                run_provenance_sha256
                                if previous_run_provenance == run_provenance_sha256
                                else None
                            ),
                        )
                        if previous_manifest_sha256
                        else ""
                    )
                    status_only_upgrade = (
                        previous_status in FEW_SHOT_SPLIT_STATUS_RANK
                        and current_status in FEW_SHOT_SPLIT_STATUS_RANK
                        and FEW_SHOT_SPLIT_STATUS_RANK[previous_status]
                        <= FEW_SHOT_SPLIT_STATUS_RANK[current_status]
                        and previous_protocol_id == current_protocol_id
                        and previous_fingerprint == previous_protocol
                    )
                    legacy_provenance_upgrade = (
                        previous_run_provenance in (None, "")
                        and previous_protocol == legacy_protocol_sha256
                    )
                    if not status_only_upgrade and not legacy_provenance_upgrade:
                        raise RuntimeError(
                            f"{model_id} 的协议指纹已变化；请使用新的 model_id，不能复用旧预测。"
                        )
                    if status_only_upgrade:
                        protocol_migration = {
                            "reason": "few_shot_split_status_only_upgrade",
                            "from_split_status": previous_status,
                            "to_split_status": current_status,
                            "from_manifest_sha256": previous_manifest_sha256,
                            "to_manifest_sha256": str(context.get("sha256", "")),
                            "from_protocol_sha256": str(previous_protocol),
                            "to_protocol_sha256": protocol_sha256,
                            "run_provenance_sha256": run_provenance_sha256,
                        }
                    else:
                        protocol_migration = {
                            "reason": "legacy_protocol_bound_to_verified_run_provenance",
                            "from_protocol_sha256": str(previous_protocol),
                            "to_protocol_sha256": protocol_sha256,
                            "run_provenance_sha256": run_provenance_sha256,
                        }
                else:
                    raise RuntimeError(
                        f"{model_id} 的旧预测没有协议指纹；请使用新的 model_id。"
                    )
        latest = {str(row.get("eval_id")): row for row in existing_rows if row.get("eval_id")}
        if retry_failed:
            latest = {
                eval_id: row
                for eval_id, row in latest.items()
                if row.get("status") == "OK"
                and row.get("parsed_answer") in {"A", "B", "C", "D"}
            }
        rows = list(latest.values())
        completed = set(latest)
        api_env = _api_key_env(runtime_model)
        api_key = os.environ.get(api_env) if api_env else None
        allowed_modes = set(runtime_model.get("evaluation_modes", ["all-images"])) & requested_modes
        selected: list[dict[str, Any]] = []
        for item in items:
            if item.get("evaluation_mode") not in allowed_modes:
                continue
            enriched = dict(item)
            if shot_count > 0:
                mapping_row = private_by_eval.get(str(item.get("eval_id")))
                if not mapping_row:
                    raise RuntimeError(f"few-shot 路由找不到 eval_id={item.get('eval_id')} 的私有映射")
                enriched["_question_type"] = str(mapping_row.get("question_type", ""))
            selected.append(enriched)
        selected_groups: list[str] = []
        group_breakdown: dict[str, int] = {}
        if limit_groups is not None:
            selected, selected_groups, group_breakdown = _select_group_subset(
                selected,
                private_mapping,
                limit_groups,
            )
        if limit is not None:
            selected = selected[:limit]
        selected_eval_ids = [str(item["eval_id"]) for item in selected]
        context = runtime_model.get("_few_shot_context")
        protocol_migrations: list[dict[str, Any]] = []
        if isinstance(previous_selection, dict):
            prior_migrations = previous_selection.get("protocol_migrations", [])
            if isinstance(prior_migrations, list):
                protocol_migrations = [
                    value for value in prior_migrations if isinstance(value, dict)
                ]
        if protocol_migration and protocol_migration not in protocol_migrations:
            protocol_migrations.append(protocol_migration)
        selection_record = {
            "schema_version": "frameevidence-prediction-selection-v2",
            "model_id": model_id,
            "api_model": _model_name(runtime_model),
            "shot_count": shot_count,
            "protocol_sha256": protocol_sha256,
            "run_provenance_sha256": run_provenance_sha256,
            "run_provenance": run_provenance,
            "few_shot_protocol_id": context.get("protocol_id") if isinstance(context, dict) else None,
            "few_shot_manifest": str(context.get("path")) if isinstance(context, dict) else None,
            "few_shot_manifest_sha256": context.get("sha256") if isinstance(context, dict) else None,
            "few_shot_split_status": context.get("split_status") if isinstance(context, dict) else None,
            "evaluation_modes": sorted(allowed_modes),
            "limit": limit,
            "limit_groups": limit_groups,
            "selected_items": len(selected_eval_ids),
            "selected_eval_ids": selected_eval_ids,
            "selected_relation_group_ids": selected_groups,
            "selected_group_breakdown": group_breakdown,
        }
        if protocol_migrations:
            selection_record["protocol_migrations"] = protocol_migrations
        write_json(
            selection_path,
            selection_record,
        )
        min_interval = float(runtime_model.get("minimum_interval_seconds", 0))
        concurrency = int(runtime_model.get("concurrency", 1))
        pending = [item for item in selected if item["eval_id"] not in completed]
        model_success = 0
        model_errors = 0
        if concurrency == 1:
            for item in pending:
                if min_interval > 0:
                    time.sleep(min(min_interval, 60))
                row = _evaluate_item(
                    item,
                    runtime_model,
                    prepared_root,
                    api_key,
                    model_id,
                    preliminary=preliminary,
                    result_label=label,
                )
                row["protocol_sha256"] = protocol_sha256
                row["run_provenance_sha256"] = run_provenance_sha256
                answer = row.get("parsed_answer")
                model_success += int(answer in {"A", "B", "C", "D"})
                model_errors += int(answer not in {"A", "B", "C", "D"})
                rows.append(row)
                completed.add(str(row["eval_id"]))
                write_jsonl(output_path, rows)
        else:
            with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=safe_id) as executor:
                futures = {
                    executor.submit(
                        _evaluate_item,
                        item,
                        runtime_model,
                        prepared_root,
                        api_key,
                        model_id,
                        preliminary,
                        label,
                    ): item["eval_id"]
                    for item in pending
                }
                for future in as_completed(futures):
                    row = future.result()
                    row["protocol_sha256"] = protocol_sha256
                    row["run_provenance_sha256"] = run_provenance_sha256
                    answer = row.get("parsed_answer")
                    model_success += int(answer in {"A", "B", "C", "D"})
                    model_errors += int(answer not in {"A", "B", "C", "D"})
                    rows.append(row)
                    completed.add(str(row["eval_id"]))
                    write_jsonl(output_path, rows)
        total_valid = sum(
            row.get("status") == "OK"
            and row.get("parsed_answer") in {"A", "B", "C", "D"}
            for row in rows
        )
        total_errors = len(rows) - total_valid
        checkpoint_path = predictions_dir / f"{safe_id}.checkpoint.json"
        checkpoint = {
            "schema_version": "frameevidence-prediction-checkpoint-v1",
            "model_id": model_id,
            "api_model": _model_name(runtime_model),
            "provider": _provider(runtime_model),
            "protocol_sha256": protocol_sha256,
            "run_provenance_sha256": run_provenance_sha256,
            "selection_file": str(selection_path.resolve()),
            "selection_sha256": _sha256_file(selection_path),
            "prediction_file": str(output_path.resolve()),
            "prediction_sha256": _sha256_file(output_path),
            "records": len(rows),
            "unique_eval_ids": len({str(row.get("eval_id", "")) for row in rows}),
            "valid_records": total_valid,
        }
        write_json(checkpoint_path, checkpoint)
        summary["models"][model_id] = {
            "status": "PASS" if total_errors == 0 else "REVIEW_REQUIRED",
            "provider": _provider(runtime_model),
            "api_model": _model_name(runtime_model),
            "shot_count": shot_count,
            "protocol_sha256": protocol_sha256,
            "few_shot_manifest_sha256": context.get("sha256") if isinstance(context, dict) else None,
            "protocol_migration": protocol_migration,
            "concurrency": concurrency,
            "selected_items": len(selected),
            "selected_groups": len(selected_groups) if limit_groups is not None else None,
            "selected_group_breakdown": group_breakdown if limit_groups is not None else None,
            "completed_items": sum(item["eval_id"] in completed for item in selected),
            "new_valid": model_success,
            "new_errors": model_errors,
            "total_records": len(rows),
            "total_valid": total_valid,
            "total_errors": total_errors,
            "output": str(output_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
        }
        any_success = any_success or bool(rows)
        any_error = any_error or bool(total_errors)
    summary["status"] = "REVIEW_REQUIRED" if any_error else "PASS" if any_success else "NOT_RUN"
    write_json(run_dir / "model_run_summary.json", summary)
    return summary
