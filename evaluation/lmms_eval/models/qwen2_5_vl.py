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
    Qwen2_5_VLForConditionalGeneration,
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
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen2_5_vl")
class Qwen2_5_VL(lmms):
    """
    Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        modality: str = "image",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",

        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        use_flash_attention_2: Optional[bool] = False,

        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        fps: Optional[float] = 2,
        total_pixels: Optional[int] = None,
        max_num_frames: int = 32,

        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,

        use_vllm: Optional[bool] = False,
        gpu_memory_utilization: Optional[float] = 0.8,
        tensor_parallel_size: Optional[int] = 1,
        pipeline_parallel_size: Optional[int] = 1,
        enforce_eager: Optional[bool] = True,

        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.use_vllm = use_vllm
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        assert modality in ["image", "video"]
        vision_process.FPS = fps
        self.total_pixels = total_pixels

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        if use_flash_attention_2:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        elif use_vllm:
            self._model = LLM(
                model=pretrained,
                # limit_mm_per_prompt={"image": 4, "video": 4},
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                enforce_eager=enforce_eager,
            )
        else:
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map).eval()

        self.modality = modality

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
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

        self.default_gen_kwargs = {
            "top_p": None,
            "top_k": None,
            "temperature": 0.0,
            "repetition_penalty": 1.05,
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

        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.interleave_visuals = interleave_visuals

        # self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

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
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

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
    def max_length(self):
        return self._max_length

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

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]

            from datasets import concatenate_datasets
            split = split[0].split(",")
            ds = concatenate_datasets([self.task_dict[task][s] for s in split])

            # visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visual_list = [doc_to_visual[0](ds[ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tokenizer.decode(self.eot_token_id)]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                for visual in visual_list[i]:
                    if self.modality == "video":  # Video file
                        visual_dict = {
                            "type": "video", "video": visual,
                            "max_pixels": self.max_pixels, "min_pixels": self.min_pixels
                        }
                        if self.total_pixels is not None:
                            visual_dict["total_pixels"] = self.total_pixels
                        processed_visuals.append(visual_dict)
                    elif self.modality == "image":  # Handle both single and multiple images
                        processed_visuals.append(
                            {
                                "type": "image", "image": visual,
                                "max_pixels": self.max_pixels, "min_pixels": self.min_pixels
                            }
                        )

                if self.interleave_visuals is False:
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})

                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
            image_inputs, video_inputs, video_kwargs = process_vision_info(batched_messages, return_video_kwargs=True)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                video_inputs[0] = video_inputs[0][indices]

            # Update with provided kwargs
            current_gen_kwargs = {**self.default_gen_kwargs, **gen_kwargs, **self.user_generation_kwargs}

            pad_token_id = self.tokenizer.pad_token_id

            if self.use_vllm:
                mm_data = {}
                if image_inputs is not None:
                    mm_data["image"] = image_inputs
                if video_inputs is not None:
                    mm_data["video"] = video_inputs

                llm_inputs = {
                    "prompt": texts[0],
                    "multi_modal_data": mm_data,
                    "video_kwargs": video_kwargs,
                }
                sampling_params = SamplingParams()
                current_gen_kwargs = {k: v for k, v in current_gen_kwargs.items() if v is not None}
                temperature = current_gen_kwargs.get("temperature", sampling_params.temperature)
                top_p = current_gen_kwargs.get("top_p", sampling_params.top_p)
                top_k = current_gen_kwargs.get("top_k", sampling_params.top_k)
                repetition_penalty = current_gen_kwargs.get("repetition_penalty", sampling_params.repetition_penalty)
                max_tokens = current_gen_kwargs["max_new_tokens"]
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                )
                outputs = self._model.generate([llm_inputs], sampling_params=sampling_params)
                answers = [outputs[0].outputs[0].text]
            else:
                inputs = self.processor(
                    text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
                )

                if self.device_map == "auto":
                    inputs = inputs.to("cuda")
                else:
                    inputs = inputs.to(self.device)
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if current_gen_kwargs["temperature"] > 0 else False,
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    top_k=current_gen_kwargs["top_k"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )

                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
