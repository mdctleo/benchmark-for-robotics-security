import json
from pathlib import Path

import torch
from vllm import LLM as VLLMEngine, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class VLM:
    """Local vLLM chat wrapper. Text prompts with optional image input, JSON output."""

    def __init__(
        self,
        model_name,
        system_prompt,
        max_new_tokens=500,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=42,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = int(top_k)
        self.seed = seed

        print(f"Model name: {self.model_name}", flush=True)

        tensor_parallel_size = max(torch.cuda.device_count(), 1)
        self.model = VLLMEngine(
            model=self.model_name,
            max_model_len=16384,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            seed=self.seed,
            gpu_memory_utilization=0.9,
            limit_mm_per_prompt={"image": 2},
            allowed_local_media_path=str(Path.cwd()),
        )

    def _is_qwen3_family(self):
        return "Qwen3" in self.model_name

    def _vllm_chat_template_kwargs(self, enable_thinking=True):
        if self._is_qwen3_family():
            return {"enable_thinking": enable_thinking}
        return None

    def _extract_thinking_trace(self, request_output):
        """Split the output text at </think>; returns (thinking_trace, response_text)."""
        if not request_output.outputs:
            return "", ""
        output_text = request_output.outputs[0].text or ""
        end_tag = "</think>"
        end_idx = output_text.find(end_tag)

        if end_idx == -1:
            return "", output_text.strip()

        thinking_trace = output_text[:end_idx].strip()
        response_text = output_text[end_idx + len(end_tag):].strip()
        return thinking_trace, response_text

    def _parse_json_response(self, output_text, thinking_trace=None):
        try:
            result = json.loads(output_text)
            if thinking_trace is not None:
                result["thinking_trace"] = thinking_trace
            return result
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON response", "raw_output": output_text}

    def _run_chat(self, conversations, sampling_params, enable_thinking=True):
        if not conversations:
            return []

        return self.model.chat(
            messages=conversations,
            sampling_params=sampling_params,
            use_tqdm=True,
            chat_template_kwargs=self._vllm_chat_template_kwargs(enable_thinking=enable_thinking),
        )

    def set_new_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt

    def _build_user_content(self, prompt, image_paths=None):
        content = [{"type": "text", "text": prompt}]
        for image_path in image_paths or []:
            resolved_image_path = Path(image_path).expanduser().resolve()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": resolved_image_path.as_uri()},
                }
            )
        return content

    def _build_message(self, prompt, image_paths=None):
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": self._build_user_content(prompt, image_paths=image_paths),
            },
        ]

    def query(self, prompts, image_paths=None, response_schema=None):
        """Run one chat per prompt. `image_paths` is None or one path per prompt."""
        if image_paths is None:
            image_paths = [None] * len(prompts)
        if len(image_paths) != len(prompts):
            raise ValueError("The number of image paths must match the number of prompts.")

        messages = []
        for prompt, image in zip(prompts, image_paths):
            messages.append(self._build_message(prompt, image_paths=[image] if image else None))

        return self.query_local_vlm(messages, response_schema=response_schema)

    def query_local_vlm(self, messages, response_schema=None):
        generation_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            max_tokens=self.max_new_tokens,
            structured_outputs=StructuredOutputsParams(json=response_schema) if response_schema else None,
        )

        request_outputs = self._run_chat(messages, generation_params, enable_thinking=False)

        parsed_outputs = []
        for request_output in request_outputs:
            if request_output is None:
                parsed_outputs.append({"error": "Empty response"})
            else:
                thinking_trace, response_text = self._extract_thinking_trace(request_output)
                parsed_outputs.append(self._parse_json_response(response_text, thinking_trace=thinking_trace))

        return parsed_outputs
