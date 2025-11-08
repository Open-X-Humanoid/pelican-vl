import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.load_video import read_video_pyav_base64
from vllm import LLM, SamplingParams
import logging
import time

try:
    from qwen_vl_utils import process_vision_info
    from qwen_vl_utils import vision_process
    from qwen_vl_utils import extract_vision_info, fetch_image
    from torchvision import io, transforms
    from torchvision.transforms import InterpolationMode
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl")
class Qwen3_VL(lmms):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_flash_attention_2: Optional[bool] = True,

        min_pixels: int = 128 * 32 * 32,
        max_pixels: int = 768 * 32 * 32,
        total_pixels: int = 128000 * 32 * 32 * 0.9,
        max_num_frames: int = None,
        fps: Optional[float] = None,

        system_prompt: Optional[str] = None,
        reasoning_prompt: Optional[str] = None,

        use_vllm: Optional[bool] = False,
        gpu_memory_utilization: Optional[float] = 0.8,
        tensor_parallel_size: Optional[int] = 1,
        enforce_eager: Optional[bool] = True,

        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = 0.01,
        repetition_penalty: Optional[float] = None,

        max_new_tokens: Optional[int] = None,
        max_model_len: Optional[int] = 96000,

        modality="video",
        **kwargs,
    ) -> None:
        super().__init__()
        if max_num_frames is not None:
            vision_process.FPS_MAX_FRAMES = max_num_frames
        if fps is not None:
            vision_process.FPS = fps
        self.use_vllm = use_vllm
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.system_prompt = system_prompt
        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len

        self.modality = modality
        self.default_gen_kwargs = {
            "top_p": None,
            "top_k": None,
            "temperature": 0.01,
            "repetition_penalty": None,
            "max_new_tokens": 128,
        }
        self.user_generation_kwargs = {}
        if top_p is not None:
            self.user_generation_kwargs["top_p"] = top_p
        if top_k is not None:
            self.user_generation_kwargs["top_k"] = top_k
        if temperature is not None:
            self.user_generation_kwargs["temperature"] = temperature
        if repetition_penalty is not None:
            self.user_generation_kwargs["repetition_penalty"] = repetition_penalty
        if max_new_tokens is not None:
            self.user_generation_kwargs["max_new_tokens"] = max_new_tokens

        accelerator = Accelerator()
        if not use_vllm:
            if accelerator.num_processes > 1:
                self._device = torch.device(f"cuda:{accelerator.local_process_index}")
                self.device_map = f"cuda:{accelerator.local_process_index}"
            else:
                self._device = torch.device(device)
                self.device_map = device_map if device_map else device

        if use_flash_attention_2 and not use_vllm:
            print("Using flash attention 2")
            eval_logger.info("Using flash attention 2")
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained,
                dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        elif use_vllm:
            print("Using vLLM")
            eval_logger.info("Using vLLM")
            self._model = LLM(
                model=pretrained,
                mm_encoder_tp_mode="data",
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                enforce_eager=enforce_eager,
                max_model_len=max_model_len,
            )
        else:
            print("Using default")
            eval_logger.info("Using default")
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(pretrained, dtype="auto", device_map=self.device_map).eval()

        self.processor = AutoProcessor.from_pretrained(pretrained)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self.batch_size_per_gpu = int(batch_size)

        if accelerator.num_processes > 1 and not use_vllm:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")

    def prepare_inputs_for_vllm(self, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # qwen_vl_utils 0.0.14+ reqired
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs

        return {
            'prompt': text,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs
        }

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            contexts = contexts.replace("<video>", "").replace("<image>", "")
            if self.reasoning_prompt:
                contexts = contexts.strip() + self.reasoning_prompt
            text_content = {"type": "text", "text": contexts.replace("<video>", "")}

            system_message = []
            if self.system_prompt is not None:
                system_message = [{"role": "system", "content": self.system_prompt}]

            from datasets import concatenate_datasets
            split = split.split(",")
            ds = concatenate_datasets([self.task_dict[task][s] for s in split])
            visuals = doc_to_visual(ds[doc_id])
            if self.modality == "video":
                video_contents = [
                    {
                        "type": "video", "video": visual,
                        "total_pixels": self.total_pixels,
                        "max_pixels": self.max_pixels, "min_pixels": self.min_pixels
                    }
                    for visual in visuals
                ]
                messages = system_message + [
                    {
                        "role": "user",
                        "content": video_contents + [text_content],
                    }
                ]
            else:
                image_contents = [
                    {
                        "type": "image", "image": visual,
                        "max_pixels": self.max_pixels, "min_pixels": self.min_pixels
                    }
                    for visual in visuals
                ]
                messages = system_message + [
                    {
                        "role": "user",
                        "content": image_contents + [text_content],
                    }
                ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            # Update with provided kwargs
            current_gen_kwargs = {**self.default_gen_kwargs, **gen_kwargs, **self.user_generation_kwargs}
            do_sample = True if current_gen_kwargs["temperature"] > 0 else False,
            current_gen_kwargs.update({"do_sample": do_sample,})
            if self.use_vllm:
                sampling_params = SamplingParams()
                top_p = current_gen_kwargs["top_p"] if current_gen_kwargs.get("top_p", None) is not None else sampling_params.top_p
                top_k = current_gen_kwargs["top_k"] if current_gen_kwargs.get("top_k", None) is not None else sampling_params.top_k
                temperature = current_gen_kwargs["temperature"] if current_gen_kwargs.get("temperature", None) is not None else sampling_params.temperature
                repetition_penalty = current_gen_kwargs["repetition_penalty"] if current_gen_kwargs.get("repetition_penalty", None) is not None else sampling_params.repetition_penalty
                max_tokens = current_gen_kwargs["max_new_tokens"] if current_gen_kwargs.get("max_new_tokens", None) is not None else sampling_params.max_tokens
                sampling_params = SamplingParams(
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                )
                inputs = [self.prepare_inputs_for_vllm(message) for message in [messages]]
                outputs = self.model.generate(inputs, sampling_params=sampling_params)
                response = outputs[0].outputs[0].text
            else:
                inputs = inputs.to(self.model.device)
                current_gen_kwargs.pop("until")
                generated_ids = self.model.generate(**inputs, **current_gen_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            res.append(response)

            pbar.update(1)
        pbar.close()
        return res
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")