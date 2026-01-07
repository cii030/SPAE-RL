import math
import json
import argparse
from typing import Dict, Any, List, Set, Optional

import requests
from transformers import AutoTokenizer


def find_all_tokens_endwith_substring(tokenizer, substring: str) -> Set[int]:
    """
    Find token ids whose token string ends with a given substring.

    This is tokenizer-dependent. For some BPE vocabularies, '.ĊĊ' may represent '.\\n\\n'.
    """
    target_tokens: Set[int] = set()
    vocab = tokenizer.get_vocab()
    for token_str, token_id in vocab.items():
        if token_str.endswith(substring):
            target_tokens.add(token_id)
    return target_tokens


def entropy_from_logprob_dict(logprobs_dict: Dict[str, float]) -> float:
    """
    Approximate entropy from a (truncated) dict of token->logprob.
    This follows the original behavior: exponentiate provided logprobs and compute entropy
    without renormalizing across the support.
    """
    if not logprobs_dict:
        return 0.0
    try:
        probs = [math.exp(lp) for lp in logprobs_dict.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        return entropy
    except Exception:
        return 0.0


def post_completions(
    api_url: str,
    model_name: str,
    prompt: str,
    n: int,
    temperature: float,
    max_tokens: int,
    logprobs: Optional[int],
    api_key: Optional[str],
) -> Dict[str, Any]:
    """
    Call vLLM OpenAI-compatible /v1/completions.
    """
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "n": n,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs

    resp = requests.post(api_url, headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def vllm_infer(
    model_name: str,
    tokenizer,
    input_obj: Dict[str, Any],
    outpath: str,
    port: int,
    api_key: Optional[str] = "token-123456",
):
    """
    A probing example following the original code logic.

    Prints are preserved in the original style:
      - It still uses print(..., file=log_file) for the same log contents as the original code.
    """
    api_url = f"http://localhost:{port}/v1/completions"
    log_file = open(outpath, "a", encoding="utf-8")

    reach_gt_solution = False
    res = {
        "solving_tokens": 0,
        "checking_tokens": 0,
        "right_to_wrong": False,
        "istrue": False,
    }

    delta_step_potential_list: List[float] = []
    step_potential_list: List[Any] = []
    step_text_list: List[str] = []

    conf_list: List[Any] = []
    acc_list: List[Any] = []

    # -----------------------------
    # 1) Main generation
    # -----------------------------
    messages = [{"role": "user", "content": input_obj["question"]}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    all_step_outputs_json = post_completions(
        api_url=api_url,
        model_name=model_name,
        prompt=formatted_prompt,
        n=1,
        temperature=0.6,
        max_tokens=32768,
        logprobs=None,
        api_key=api_key,
    )

    output_samples = all_step_outputs_json["choices"][0]
    answer_text = output_samples["text"]

    # --- keep the original log prints exactly ---
    print("The complete answer is:", file=log_file)
    print(answer_text, file=log_file)

    istrue = str(input_obj["answer"]) in answer_text[-300:]
    res["istrue"] = istrue

    token_id_list = tokenizer.encode(answer_text, add_special_tokens=False)

    # -----------------------------
    # 2) Find step boundaries in <think> ... </think>
    # -----------------------------
    found_points: List[Dict[str, Any]] = []

    target_ids = find_all_tokens_endwith_substring(tokenizer, ".ĊĊ")

    think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")
    search_end_idx = len(token_id_list)
    try:
        search_end_idx = token_id_list.index(think_end_token_id)
    except ValueError:
        pass

    for i in range(search_end_idx):
        current_token_id = token_id_list[i]
        if current_token_id == think_end_token_id:
            break
        if current_token_id in target_ids:
            found_points.append({"index": i, "prob": 0})

    prev_step_potential = 0.0
    prev_point_pos = 0

    # -----------------------------
    # 3) Probe after each step boundary
    # -----------------------------
    for i, point in enumerate(found_points):
        acc = 0.0
        conf = 0.0
        acc_list_k: List[float] = []
        conf_list_k: List[float] = []

        print(f"Step count: {i}，current tokens: {point['index']}", file=log_file)
        restart_pos = point["index"]

        truncated_token_id_list = token_id_list[: restart_pos + 1]
        truncated_text = tokenizer.decode(truncated_token_id_list)

        step_token_id_list = token_id_list[prev_point_pos + 1 : restart_pos + 1]
        step_text = tokenizer.decode(step_token_id_list)
        step_text_list.append(step_text)
        prev_point_pos = restart_pos

        probe_prompt = "**Final Answer**\n\\boxed{"
        current_prompt = formatted_prompt + truncated_text + probe_prompt

        print(step_text, file=log_file)

        all_step_outputs_json = post_completions(
            api_url=api_url,
            model_name=model_name,
            prompt=current_prompt,
            n=5,
            temperature=0.6,
            max_tokens=10,
            logprobs=20,
            api_key=api_key,
        )

        TARGET_TOKEN_STR = str(input_obj["answer"])
        target_token_ids = tokenizer.encode(TARGET_TOKEN_STR, add_special_tokens=False)

        print("answer token:", target_token_ids, file=log_file)

        num_samples = len(all_step_outputs_json["choices"])

        for j, step_output in enumerate(all_step_outputs_json["choices"]):
            print(f"Sampling: {j}", file=log_file)

            valid_sample_count = 0
            mean_acc = 0.0
            mean_conf = 0.0

            logprobs_obj = step_output.get("logprobs", {})
            if logprobs_obj is None:
                logprobs_obj = {}

            logprobs_list_of_dicts = logprobs_obj.get("top_logprobs", [])
            if logprobs_list_of_dicts is None:
                logprobs_list_of_dicts = []

            for k, target_token_id in enumerate(target_token_ids):
                if k >= len(logprobs_list_of_dicts):
                    break

                current_step_logprobs = logprobs_list_of_dicts[k]
                if current_step_logprobs is None:
                    current_step_logprobs = {}

                target_token = tokenizer.convert_ids_to_tokens(target_token_id)
                logprob_val = current_step_logprobs.get(str(target_token))

                if logprob_val is not None:
                    mean_acc += float(logprob_val)
                else:
                    mean_acc += -10

                mean_conf += entropy_from_logprob_dict(current_step_logprobs)
                valid_sample_count += 1
            mean_conf = math.exp(-mean_conf)
            mean_acc = math.exp(mean_acc)

            if valid_sample_count > 0:
                conf += (mean_conf / valid_sample_count)
                acc += (mean_acc / valid_sample_count)

            answer_text_probe = step_output.get("text", "")
            print(answer_text_probe, file=log_file)

            acc_list_k.append(mean_acc / valid_sample_count if valid_sample_count else -10.0)
            conf_list_k.append(mean_conf / valid_sample_count if valid_sample_count else 0.0)

        acc /= max(num_samples, 1)
        conf /= max(num_samples, 1)
        acc_list.append(acc_list_k)
        conf_list.append(conf_list_k)

        # Step Potential (same formula as your original code)
        step_potential = 1.5 * acc * conf + 0.5 * acc - conf

        # Difference starts from step 2 (i==0 => 0)
        if i == 0:
            delta_step_potential = 0.0
        else:
            delta_step_potential = step_potential - prev_step_potential

        step_potential_list.append((point["index"], step_potential))
        delta_step_potential_list.append(delta_step_potential)

        print(
            f"Step: {i}, average acc:{acc}, average conf:{conf}, potential value:{step_potential}, delta potential value:{delta_step_potential}",
            file=log_file,
        )

        if istrue and step_potential > 0.9 and (reach_gt_solution is False):
            reach_gt_solution = True
            res["solving_tokens"] = point["index"]
            res["checking_tokens"] = search_end_idx - point["index"]

        if (not istrue) and step_potential > 0.9:
            res["right_to_wrong"] = True

        prev_step_potential = step_potential

    res["acc"] = acc_list
    res["conf"] = conf_list

    print(res, file=log_file)
    print(step_text_list, file=log_file)

    delta_step_potential_list.sort(reverse=True)
    print(delta_step_potential_list, file=log_file)
    print(step_potential_list, file=log_file)

    log_file.close()
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name served by vLLM")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", type=str, default="token-123456")
    parser.add_argument("--outpath", type=str, default="probe_log.txt")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--answer", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    input_obj = {"question": args.question, "answer": args.answer}
    res = vllm_infer(
        model_name=args.model,
        tokenizer=tokenizer,
        input_obj=input_obj,
        outpath=args.outpath,
        port=args.port,
        api_key=args.api_key,
    )

    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
