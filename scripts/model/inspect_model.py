#!/usr/bin/env python3
"""静态读取 Hugging Face 风格模型目录，不导入 remote code。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

JSON_FILES = (
    'config.json',
    'generation_config.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'adapter_config.json',
)
INDEX_GLOBS = ('*.safetensors.index.json', '*.bin.index.json')
WEIGHT_GLOBS = ('*.safetensors', '*.bin', '*.pt', '*.pth')
NESTED_MODEL_CONFIG_KEYS = ('text_config', 'llm_config', 'language_config', 'model_config')
ARCHITECTURE_KEYS = (
    'vocab_size',
    'hidden_size',
    'num_hidden_layers',
    'num_attention_heads',
    'num_key_value_heads',
    'head_dim',
    'intermediate_size',
    'dense_intermediate_size',
    'shared_intermediate_size',
    'hidden_act',
    'rms_norm_eps',
    'max_position_embeddings',
    'rope_theta',
    'rope_scaling',
    'rotary_dim',
    'partial_rotary_factor',
    'sliding_window',
    'tie_word_embeddings',
    'use_qk_norm',
    'qk_norm_type',
    'use_gemma_norm',
    'attention_output_gate',
    'num_local_experts',
    'num_experts_per_tok',
    'n_shared_experts',
    'scoring_func',
    'use_routing_bias',
    'routed_scaling_factor',
    'num_mtp_modules',
    'num_nextn_predict_layers',
    'sparse_attention_config',
)


def load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding='utf-8')), None
    except Exception as exc:  # noqa: BLE001
        return None, f'{type(exc).__name__}: {exc}'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    return parser.parse_args()


def config_sections(config: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(config, dict):
        return {}

    sections = {'root': config}
    for key in NESTED_MODEL_CONFIG_KEYS:
        value = config.get(key)
        if isinstance(value, dict):
            sections[key] = value
    return sections


def collect_architecture(config: Any) -> tuple[dict[str, Any], str | None, list[str]]:
    sections = config_sections(config)
    for section_name in (*NESTED_MODEL_CONFIG_KEYS, 'root'):
        section = sections.get(section_name)
        if not section:
            continue
        values = {key: section[key] for key in ARCHITECTURE_KEYS if key in section}
        if values:
            return values, section_name, sorted(sections)
    return {}, None, sorted(sections)


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        print(f'模型目录不存在: {model_path}', file=sys.stderr)
        return 2

    files: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name in JSON_FILES:
        path = model_path / name
        if path.exists():
            data, error = load_json(path)
            if error:
                errors[name] = error
            else:
                files[name] = data

    index_files: dict[str, Any] = {}
    for pattern in INDEX_GLOBS:
        for path in sorted(model_path.glob(pattern)):
            data, error = load_json(path)
            if error:
                errors[path.name] = error
            else:
                weight_map = data.get('weight_map', {}) if isinstance(data, dict) else {}
                index_files[path.name] = {
                    'metadata': data.get('metadata', {}) if isinstance(data, dict) else {},
                    'weight_count': len(weight_map),
                    'shards': sorted(set(weight_map.values())),
                    'sample_weights': list(weight_map.keys())[:100],
                }

    weights: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for pattern in WEIGHT_GLOBS:
        for path in sorted(model_path.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            weights.append({'name': path.name, 'size_bytes': path.stat().st_size})

    config = files.get('config.json', {})
    architecture, architecture_source, available_config_sections = collect_architecture(config)
    summary = {
        'model_path': str(model_path),
        'config_present': 'config.json' in files,
        'identity': {
            'architectures': config.get('architectures') if isinstance(config, dict) else None,
            'model_type': config.get('model_type') if isinstance(config, dict) else None,
            'torch_dtype': config.get('torch_dtype') if isinstance(config, dict) else None,
            'transformers_version': config.get('transformers_version') if isinstance(config, dict) else None,
        },
        'architecture_source': architecture_source,
        'available_config_sections': available_config_sections,
        'architecture': architecture,
        'json_files': files,
        'index_files': index_files,
        'weight_files': weights,
        'errors': errors,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    if 'config.json' not in files:
        print('未找到可解析的 config.json', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
