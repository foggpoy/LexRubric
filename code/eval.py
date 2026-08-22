import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

import requests


# =========================
# config
# =========================
BASE_DIR = Path(__file__).parent
PROMPT_TEMPLATE_PATH = BASE_DIR / "prompt.txt"

# DashScope OpenAI-compatible endpoint
DASHSCOPE_CHAT_URL = "base_url"
DASHSCOPE_API_KEY = "api_key"

# sglang endpoint（set provider=sglang）
SGLANG_URL = "http://127.0.0.1:4679"

# judge model (via DASHSCOPE_API_KEY)
JUDGE_MODEL = "qwen3.6-27b"
JUDGE_ENABLE_THINKING = False
JUDGE_MAX_TOKENS = 4096
JUDGE_TEMPERATURE = 0.0

# Concurrency and Retry
RESPONSE_WORKERS = 5
RUBRIC_WORKERS = 10
API_TIMEOUT = 3600
MAX_RETRIES = 3

# provider:
# - dashscope_chat: API (OpenAI template)
# - dashscope_gemini_native: Gemini contents/generationConfig
# - sglang: local openai-style /v1/chat/completions
MODELS = [
    {
        "name": "gpt-5.2",
        "provider": "dashscope_chat",
        "enable_thinking": None,
        "res_max_tokens": 16384,
        "temperature": 0.6,
    }
]

# task config：
# - legacy_pair: evaluate「falvzixun + sifakaoshi」
ACTIVE_TASK = "legacy_pair"
TASK_PRESETS = {
    "legacy_pair": {
        "output_root": BASE_DIR / "model_evaluation_results",
        "summary_key": "category",
        "datasets": [
            {
                "name": "法律咨询",
                "path": BASE_DIR.parent / "data" / "falvzixun.json",
                "id_field": "ID",
                "question_field": "question",
                "answer_field": "answer",
                "rubrics_field": "rubrics",
                "rubric_criterion_field": "criterion",
                "rubric_point_field": "point",
                "rubric_dimension_field": "dimension",
            },
            {
                "name": "司法考试",
                "path": BASE_DIR.parent / "data" / "sifakaoshi.json",
                "id_field": "ID",
                "question_field": "question",
                "answer_field": "answer",
                "rubrics_field": "rubrics",
                "rubric_criterion_field": "criterion",
                "rubric_point_field": "point",
                "rubric_dimension_field": "dimension",
            },
        ],
    },
}

# =========================
# config end
# =========================


print_lock = Lock()


class RetryableError(Exception):
    pass


def safe_print(*args, **kwargs):
    with print_lock:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def load_prompt_template() -> str:
    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def normalize_rubric(rubric: dict, dataset_cfg: dict) -> dict:
    return {
        "criterion": rubric.get(dataset_cfg["rubric_criterion_field"], ""),
        "point": rubric.get(dataset_cfg["rubric_point_field"], 0),
        "dimension": rubric.get(dataset_cfg["rubric_dimension_field"], ""),
    }


def load_dataset(dataset_cfg: dict) -> list:
    file_path = Path(dataset_cfg["path"])
    if not file_path.exists():
        safe_print(f"[Warning] 文件不存在: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    normalized = []
    for item in raw_data:
        data_id = item.get(dataset_cfg["id_field"])
        if data_id is None:
            continue
        normalized.append(
            {
                "ID": data_id,
                "question": item.get(dataset_cfg["question_field"], ""),
                "answer": item.get(dataset_cfg["answer_field"], "")
                if dataset_cfg.get("answer_field")
                else None,
                "rubrics": [
                    normalize_rubric(r, dataset_cfg)
                    for r in item.get(dataset_cfg["rubrics_field"], [])
                ],
            }
        )
    return normalized


def extract_text_content(result: dict) -> str:
    # 1) OpenAI template
    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            if content.strip():
                return content.strip()
        elif isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        texts.append(text)
            if texts:
                return "\n".join(texts).strip()

    # 2) Anthropic style: content=[{type:"text", text:"..."}]
    content_blocks = result.get("content")
    if isinstance(content_blocks, list):
        texts = []
        for block in content_blocks:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
        if texts:
            return "\n".join(texts).strip()

    # 3) Gemini: candidates[].content.parts[].text
    candidates = result.get("candidates")
    if isinstance(candidates, list) and candidates:
        cand_content = candidates[0].get("content", {})
        parts = cand_content.get("parts", [])
        if isinstance(parts, list):
            texts = []
            for part in parts:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        texts.append(text)
            if texts:
                return "\n".join(texts).strip()

    return ""


def call_dashscope_chat(
    model: str,
    messages: list,
    enable_thinking=None,
    max_tokens: int = 16384,
    temperature: float = 0.6,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                DASHSCOPE_CHAT_URL,
                json=payload,
                headers=headers,
                timeout=API_TIMEOUT,
            )
            if response.status_code != 200:
                try:
                    err_json = response.json()
                    err_msg = err_json.get("error") or err_json
                except Exception:
                    err_msg = response.text
                if response.status_code in (429, 500, 502, 503, 504):
                    raise RetryableError(f"HTTP {response.status_code}: {err_msg}")
                return f"[API HTTP {response.status_code}: {err_msg}]"

            result = response.json()
            content = extract_text_content(result)
            if not content:
                raise RetryableError(
                    f"Empty/unsupported response format: keys={list(result.keys())}"
                )
            return content
        except RetryableError as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return f"[API max retries exceeded: {e}]"
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return "[API timeout]"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(1.0 * attempt)
            else:
                return f"[API error: {e}]"
    return "[API error: unknown]"


def call_dashscope_gemini(
    model: str, question: str, max_tokens: int = 16384, temperature: float = 0.6
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
    }
    payload = {
        "model": model,
        "contents": [
            {
                "role": "user",
                "parts": [{"text": question}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                DASHSCOPE_CHAT_URL,
                json=payload,
                headers=headers,
                timeout=API_TIMEOUT,
            )
            if response.status_code != 200:
                try:
                    err_json = response.json()
                    err_msg = err_json.get("error") or err_json
                except Exception:
                    err_msg = response.text
                if response.status_code in (429, 500, 502, 503, 504):
                    raise RetryableError(f"HTTP {response.status_code}: {err_msg}")
                return f"[API HTTP {response.status_code}: {err_msg}]"

            result = response.json()
            content = extract_text_content(result)
            if not content:
                raise RetryableError(
                    f"Empty/unsupported response format: keys={list(result.keys())}"
                )
            return content
        except RetryableError as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return f"[API max retries exceeded: {e}]"
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return "[API timeout]"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(1.0 * attempt)
            else:
                return f"[API error: {e}]"
    return "[API error: unknown]"


def call_sglang(model: str, question: str, max_tokens: int = 16384, temperature: float = 0.6) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {"Content-Type": "application/json"}
    chat_url = f"{SGLANG_URL}/v1/chat/completions"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                chat_url,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT,
            )
            if response.status_code != 200:
                msg = response.text
                if response.status_code in (429, 500, 502, 503, 504):
                    raise RetryableError(f"HTTP {response.status_code}: {msg}")
                return f"[sglang HTTP {response.status_code}: {msg}]"

            result = response.json()
            content = extract_text_content(result)
            if not content:
                raise RetryableError(
                    f"Empty/unsupported response format: keys={list(result.keys())}"
                )
            return content
        except RetryableError as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return f"[sglang max retries exceeded: {e}]"
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                return "[sglang timeout]"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(1.0 * attempt)
            else:
                return f"[sglang error: {e}]"
    return "[sglang error: unknown]"


def is_api_error(response: str) -> bool:
    error_prefixes = [
        "[API HTTP",
        "[API max retries",
        "[API timeout]",
        "[API error:",
        "[sglang HTTP",
        "[sglang max retries",
        "[sglang timeout]",
        "[sglang error:",
    ]
    return any(response.startswith(prefix) for prefix in error_prefixes)


def parse_json_response(response: str):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    pattern = r"```json\s*(.*?)\s*```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    pattern2 = r"```\s*(.*?)\s*```"
    match2 = re.search(pattern2, response, re.DOTALL)
    if match2:
        try:
            return json.loads(match2.group(1))
        except json.JSONDecodeError:
            pass

    return None


def extract_response_after_think(response: str) -> str:
    if "</think>" in response:
        parts = response.split("</think>", 1)
        if len(parts) > 1:
            return parts[1].strip()
    return response


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def generate_model_response(model_cfg: dict, question: str, data_id) -> dict:
    provider = model_cfg["provider"]
    model_name = model_cfg["name"]
    res_max_tokens = int(model_cfg.get("res_max_tokens", 16384))
    temperature = float(model_cfg.get("temperature", 0.6))

    if provider == "dashscope_chat":
        messages = [{"role": "user", "content": question}]
        response = call_dashscope_chat(
            model=model_name,
            messages=messages,
            enable_thinking=model_cfg.get("enable_thinking"),
            max_tokens=res_max_tokens,
            temperature=temperature,
        )
    elif provider == "dashscope_gemini_native":
        response = call_dashscope_gemini(
            model=model_name,
            question=question,
            max_tokens=res_max_tokens,
            temperature=temperature,
        )
    elif provider == "sglang":
        response = call_sglang(
            model=model_name,
            question=question,
            max_tokens=res_max_tokens,
            temperature=temperature,
        )
    else:
        response = f"[API error: unsupported provider={provider}]"

    return {
        "data_id": data_id,
        "model_response": response,
        "error": is_api_error(response),
    }


def evaluate_single_criterion(
    prompt_template: str, conversation: str, criterion: dict, criterion_idx: int
) -> dict:
    criterion_content = criterion.get("criterion", "")
    point = criterion.get("point", 0)
    dimension = criterion.get("dimension", "")
    direction = "不是" if point >= 0 else "是"

    prompt = (
        prompt_template.replace("{conversation}", conversation)
        .replace("{rubric_item}", criterion_content)
        .replace("{direction}", direction)
    )

    messages = [{"role": "user", "content": prompt}]
    max_parse_retries = 3

    for retry in range(max_parse_retries):
        response = call_dashscope_chat(
            model=JUDGE_MODEL,
            messages=messages,
            enable_thinking=JUDGE_ENABLE_THINKING,
            max_tokens=JUDGE_MAX_TOKENS,
            temperature=JUDGE_TEMPERATURE,
        )

        if is_api_error(response):
            return {
                "criterion_idx": criterion_idx,
                "criterion": criterion_content,
                "point": point,
                "dimension": dimension,
                "direction": direction,
                "criteria_met": False,
                "explanation": response,
                "raw_response": response,
                "parse_error": True,
            }

        parsed = parse_json_response(response)
        if parsed is not None and "criteria_met" in parsed:
            return {
                "criterion_idx": criterion_idx,
                "criterion": criterion_content,
                "point": point,
                "dimension": dimension,
                "direction": direction,
                "criteria_met": to_bool(parsed.get("criteria_met")),
                "explanation": parsed.get("explanation", ""),
                "raw_response": response,
                "parse_error": False,
            }

        if retry < max_parse_retries - 1:
            pass

    return {
        "criterion_idx": criterion_idx,
        "criterion": criterion_content,
        "point": point,
        "dimension": dimension,
        "direction": direction,
        "criteria_met": False,
        "explanation": "JSON解析失败",
        "raw_response": response,
        "parse_error": True,
    }


def evaluate_response_with_rubrics(
    question: str, model_response: str, rubrics: list, prompt_template: str
) -> tuple:
    filtered_response = extract_response_after_think(model_response)
    conversation = f"user: {question}\nassistant: {filtered_response}"

    criterion_results = []
    with ThreadPoolExecutor(max_workers=RUBRIC_WORKERS) as executor:
        futures = {
            executor.submit(
                evaluate_single_criterion, prompt_template, conversation, criterion, idx
            ): idx
            for idx, criterion in enumerate(rubrics)
        }
        for future in as_completed(futures):
            criterion_results.append(future.result())

    criterion_results.sort(key=lambda x: x["criterion_idx"])
    total_score = sum(r["point"] for r in criterion_results if r["criteria_met"])
    return total_score, criterion_results, filtered_response


def get_completed_ids(output_dir: Path) -> set:
    completed = set()
    if not output_dir.exists():
        return completed
    for json_file in output_dir.glob("*.json"):
        if json_file.name.startswith("_"):
            continue
        completed.add(json_file.stem)
    return completed


def save_summary(output_dir: Path, dataset_name: str, model_name: str, summary_key: str):
    results = []
    for json_file in output_dir.glob("*.json"):
        if json_file.name.startswith("_"):
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.append(
                    {
                        "ID": data.get("ID"),
                        "total_score": data.get("total_score", 0),
                        "error": data.get("error", False),
                    }
                )
        except Exception:
            pass

    summary = {
        "model": model_name,
        summary_key: dataset_name,
        "total": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "avg_score": sum(r["total_score"] for r in results) / len(results) if results else 0,
        "results": sorted(results, key=lambda x: x["ID"]),
    }
    with open(output_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def process_model_dataset(
    model_cfg: dict,
    dataset_cfg: dict,
    prompt_template: str,
    output_root: Path,
    summary_key: str,
):
    model_name = model_cfg["name"]
    dataset_name = dataset_cfg["name"]
    output_dir = output_root / model_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    data_list = load_dataset(dataset_cfg)
    if not data_list:
        safe_print(f"  [{dataset_name}] 无可用数据，跳过")
        return

    completed_ids = get_completed_ids(output_dir)
    pending_data = [d for d in data_list if str(d["ID"]) not in completed_ids]

    if not pending_data:
        safe_print(f"  [{dataset_name}] 全部已完成，跳过")
        return

    safe_print(f"  [{dataset_name}] 待处理: {len(pending_data)} 条 (已完成: {len(completed_ids)})")
    safe_print("    生成模型回复...")

    responses = {}
    with ThreadPoolExecutor(max_workers=RESPONSE_WORKERS) as executor:
        futures = {
            executor.submit(generate_model_response, model_cfg, item["question"], item["ID"]): item["ID"]
            for item in pending_data
        }
        for future in as_completed(futures):
            result = future.result()
            responses[result["data_id"]] = result

    safe_print(f"    模型回复生成完成: {len(responses)} 条")
    safe_print("    评估中...")

    for idx, item in enumerate(pending_data):
        data_id = item["ID"]
        question = item["question"]
        rubrics = item["rubrics"]

        response_info = responses.get(data_id, {})
        model_response = response_info.get("model_response", "")
        if response_info.get("error", False):
            result = {
                "ID": data_id,
                "question": question,
                "model_response": model_response,
                "filtered_response": "",
                "total_score": 0,
                "criterion_evaluations": [],
                "error": True,
                "error_message": model_response,
            }
            if item.get("answer") is not None:
                result["answer"] = item.get("answer", "")
        else:
            total_score, criterion_evals, filtered_response = evaluate_response_with_rubrics(
                question, model_response, rubrics, prompt_template
            )
            result = {
                "ID": data_id,
                "question": question,
                "model_response": model_response,
                "filtered_response": filtered_response,
                "total_score": total_score,
                "criterion_evaluations": criterion_evals,
                "error": False,
            }
            if item.get("answer") is not None:
                result["answer"] = item.get("answer", "")

        with open(output_dir / f"{data_id}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if (idx + 1) % 10 == 0 or idx == len(pending_data) - 1:
            safe_print(f"    进度: {idx + 1}/{len(pending_data)}")

    save_summary(output_dir, dataset_name, model_name, summary_key)


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    if ACTIVE_TASK not in TASK_PRESETS:
        raise ValueError(
            f"ACTIVE_TASK={ACTIVE_TASK} 不存在，可选: {list(TASK_PRESETS.keys())}"
        )
    task_cfg = TASK_PRESETS[ACTIVE_TASK]
    output_root = Path(task_cfg["output_root"])
    datasets = task_cfg["datasets"]
    summary_key = task_cfg["summary_key"]

    safe_print("=" * 80)
    safe_print("统一评测")
    safe_print(f"时间: {datetime.now().isoformat()}")
    safe_print(f"任务预设: {ACTIVE_TASK}")
    safe_print("=" * 80)
    output_root.mkdir(parents=True, exist_ok=True)

    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "YOUR_DASHSCOPE_API_KEY":
        safe_print("[Warning] DASHSCOPE_API_KEY 未配置，DashScope 请求会失败。")

    prompt_template = load_prompt_template()
    safe_print("Prompt 模板已加载")
    safe_print(f"模型数: {len(MODELS)} | 数据集数: {len(datasets)}")

    for model_cfg in MODELS:
        safe_print(f"\n{'='*60}")
        safe_print(f"模型: {model_cfg['name']}")
        safe_print(f"provider: {model_cfg['provider']}")
        safe_print(f"res_max_tokens: {model_cfg.get('res_max_tokens', 16384)}")
        safe_print(f"{'='*60}")
        for dataset_cfg in datasets:
            process_model_dataset(
                model_cfg, dataset_cfg, prompt_template, output_root, summary_key
            )

    safe_print(f"\n{'='*80}")
    safe_print("评测完成")
    safe_print(f"结果保存在: {output_root}")
    safe_print("=" * 80)


if __name__ == "__main__":
    main()
