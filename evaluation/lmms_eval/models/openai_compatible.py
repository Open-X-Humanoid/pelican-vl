import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import json
import os
import time
from io import BytesIO
from typing import List, Tuple, Union

import numpy as np
import requests as url_requests
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
import portalocker

try:
    from decord import VideoReader, cpu
except ImportError:
    pass

from dotenv import find_dotenv, load_dotenv
from loguru import logger as eval_logger
from openai import AzureOpenAI, OpenAI
from PIL import Image

load_dotenv(verbose=True)


@register_model("openai_compatible")
class OpenAICompatible(lmms):
    def __init__(
        self,
        model_version: str = "grok-2-latest",
        timeout: int = 300,
        max_retries: int = 5,
        max_size_in_mb: int = 20,
        continual_mode: bool = False,
        response_persistent_folder: str = None,
        azure_openai: bool = False,
        fps: int = 2,
        max_num_frames: int = 32,
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
        thinking_budget: int = 2048,
        max_workers=1,
        **kwargs,
    ) -> None:
        print(locals())
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_size_in_mb = max_size_in_mb
        self.continual_mode = continual_mode
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.thinking_budget = thinking_budget
        self.max_workers = max_workers

        if self.continual_mode:
            if response_persistent_folder is None:
                raise ValueError("Continual mode requires a persistent path for the response. Please provide a valid path.")
            os.makedirs(response_persistent_folder, exist_ok=True)
            self.response_persistent_folder = response_persistent_folder
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

            if os.path.exists(self.response_persistent_file):
                with portalocker.Lock(self.response_persistent_file, 'r', timeout=50, flags=portalocker.LOCK_SH) as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        self.client = (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"), timeout=self.timeout)
            if not azure_openai
            else AzureOpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"), azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"), api_version=os.getenv("AZURE_OPENAI_API_VERSION"))
        )

    # Function to encode the image
    def encode_image(self, image: Union[Image.Image, str]):
        max_size = self.max_size_in_mb * 1024 * 1024  # 20MB in bytes
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.copy()

        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()

        # If image is too large, resize it while maintaining aspect ratio
        while len(byte_data) > max_size and img.size[0] > 100 and img.size[1] > 100:
            new_size = (int(img.size[0] * 0.75), int(img.size[1] * 0.75))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()

        base64_str = base64.b64encode(byte_data).decode("utf-8")
        return base64_str

    # Function to encode the video
    def encode_video(self, video_path, max_num_frames):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num, video_fps = len(vr), vr.get_avg_fps()
        for_get_frames_num = min(
            max_num_frames, int(total_frame_num / video_fps * self.fps), total_frame_num
        )
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, for_get_frames_num, dtype=int)

        # Ensure the last frame is included
        if total_frame_num - 1 not in uniform_sampled_frames:
            uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="PNG")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)

        return base64_frames

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    def generate_uuid(self, task, split, doc_id, gen_kwargs):
        temperature = gen_kwargs["temperature"]
        top_p = gen_kwargs["top_p"]
        max_new_tokens = gen_kwargs.get("max_new_tokens", self.max_new_tokens)
        api_uuid = (
            f"fps-{self.fps}###"
            f"max_num_frames-{self.max_num_frames}###"
            f"max_size_in_mb-{self.max_size_in_mb}###"
            f"temperature-{temperature}###"
            f"top_p-{top_p}###"
            f"max_new_tokens-{max_new_tokens}"
        )
        example_uuid = f"task-{task}###split-{split}###doc_id-{doc_id}"
        return f"{api_uuid}######{example_uuid}"

    def generate_until(self, requests) -> List[str]:
        res = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        lock = threading.Lock()

        def process_single(idx, reg):
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = reg.args

            gen_kwargs.setdefault("max_new_tokens", 8192)
            gen_kwargs.setdefault("temperature", 0)
            gen_kwargs.setdefault("top_p", 1.0)
            if self.max_new_tokens is not None:
                gen_kwargs["max_new_tokens"] = self.max_new_tokens
            if self.temperature is not None:
                gen_kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                gen_kwargs["top_p"] = self.top_p

            if self.continual_mode and self.cache_mode == "resume":
                uuid = self.generate_uuid(task, split, doc_id, gen_kwargs)
                if uuid in self.response_cache:
                    return self.response_cache[uuid]

            from datasets import concatenate_datasets
            split = split.split(",")
            ds = concatenate_datasets([self.task_dict[task][s] for s in split])
            visuals = [doc_to_visual(ds[doc_id])]
            modality = "text"
            max_num_frames = self.max_num_frames
            imgs = []
            if None not in visuals:
                visuals = self.flatten(visuals)
                for visual in visuals:
                    if isinstance(visual, str):
                        if any(visual.endswith(ext) for ext in [".mp4", ".avi", ".mov", ".flv", ".wmv"]):
                            modality = "video"
                            imgs.extend(self.encode_video(visual, max_num_frames))
                        elif any(visual.endswith(ext) for ext in
                                 [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]):
                            modality = "image"
                            imgs.append(self.encode_image(visual))
                    elif isinstance(visual, Image.Image):
                        imgs.append(self.encode_image(visual))

            payload = {
                "model": self.model_version,
                "messages": [{"role": "user", "content": [{"type": "text", "text": contexts}]}],
                "max_tokens": gen_kwargs["max_new_tokens"],
                "temperature": gen_kwargs["temperature"],
            }
            for img in imgs:
                payload["messages"][0]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                )

            if "gemini-2.5" in self.model_version:
                payload["extra_body"] = {
                    "extra_body":{
                        "google": {"thinking_config": {"thinking_budget": self.thinking_budget, "include_thoughts": True}}
                    }
                }

            response_text = ""
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(**payload)
                    response_text = response.choices[0].message.content
                    if response_text is None:
                        payload["max_tokens"] *= 2
                        raise ValueError("Response text is None")
                    break
                except Exception as e:
                    error_msg = str(e)
                    eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed with error: {error_msg}")
                    if "413" in error_msg and modality == "video":
                        max_num_frames //= 2
                        imgs = []
                        for visual in visuals:
                            imgs.extend(self.encode_video(visual, max_num_frames))
                        payload["messages"][0]["content"] = (
                                [{"type": "text", "text": contexts}]
                                + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}} for img
                                   in imgs]
                        )
                        eval_logger.info(f"Video is too long for the api, reducing max_num_frames to {max_num_frames}")
                    if attempt == self.max_retries - 1:
                        eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                        response_text = ""
                    else:
                        time.sleep(10)
            if self.continual_mode:
                uuid = self.generate_uuid(task, split, doc_id, gen_kwargs)
                with lock:
                    self.response_cache[uuid] = response_text

                    if not os.path.exists(self.response_persistent_file):
                        with open(self.response_persistent_file, "w") as f:
                            json.dump({}, f)

                    try:
                        with portalocker.Lock(self.response_persistent_file, mode="r+", flags=portalocker.LOCK_EX,
                                              timeout=30) as f:
                            try:
                                f.seek(0)
                                cache = json.load(f)
                            except json.JSONDecodeError:
                                cache = {}

                            cache[uuid] = response_text
                            f.seek(0)
                            json.dump(cache, f)
                            f.truncate()
                            f.flush()
                            os.fsync(f.fileno())
                    except Exception as e:
                        print(f"[Cache write failed] {e}")
            with lock:
                pbar.update(1)

            return response_text

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(process_single, i, reg): i for i, reg in enumerate(requests)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    res[idx] = future.result()
                except Exception as e:
                    res[idx] = ""
                    print(f"[Thread-{idx}] failed: {e}")

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for OpenAI compatible models")

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("TODO: Implement loglikelihood for OpenAI compatible models")
