from typing import List, Tuple, Union, Optional

import os
import io
import json
import time
import base64
import threading
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from tqdm import tqdm

from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
import portalocker
from loguru import logger as eval_logger

try:
    import dashscope
    from dashscope import MultiModalConversation
except:
    eval_logger.debug("Can not import Dashscope")


@register_model("qwen3_vl_api")
class Qwen3_VL_API(lmms):
    def __init__(
        self,
        model_version: str = "qwen3-vl-plus",
        timeout: int = 300,
        max_retries: int = 5,

        min_pixels: int = None,
        max_pixels: int = None,
        total_pixels: int = None,
        fps: Optional[float] = None,
        max_num_frames: Optional[int] = None,

        system_prompt: Optional[str] = None,
        reasoning_prompt: Optional[str] = None,

        top_p: Optional[float] = None,
        top_k: Optional[int] = None,

        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,

        max_new_tokens: Optional[int] = None,
        max_model_len: Optional[int] = None,

        continual_mode: bool = False,
        response_persistent_folder: str = None,

        modality="video",
        max_workers=1,

        enable_thinking: bool = False,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        self.model_version = model_version
        self.max_retries = max_retries
        self.timeout = timeout

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.system_prompt = system_prompt
        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty

        self.max_new_tokens = max_new_tokens
        self.max_model_len = max_model_len

        self.modality = modality
        self.max_workers = max_workers
        self.enable_thinking = enable_thinking

        self.default_gen_kwargs = {
            "top_p": None,
            "top_k": None,
            "temperature": 0.0,  # Set to 0 for greedy default
            "repetition_penalty": None,
            "presence_penalty": None,
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
        if presence_penalty is not None:
            self.user_generation_kwargs["presence_penalty"] = presence_penalty

        self.continual_mode = continual_mode
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

    def generate_uuid(self, task, split, doc_id, gen_kwargs):
        sorted_items = sorted(gen_kwargs.items())

        params_str = "###".join(f"{k}-{v}" for k, v in sorted_items)

        prefix = f"fps-{self.fps}###" if self.fps is not None else ""
        api_uuid = f"{prefix}{params_str}"

        example_uuid = f"task-{task}###split-{split}###doc_id-{doc_id}"

        return f"{api_uuid}######{example_uuid}"

    def generate_until(self, requests) -> List[str]:
        """
        Multi-threaded version with caching and file-lock protection.
        Each request result is cached to avoid repeated API calls across runs or threads.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        cache_lock = threading.Lock()
        cache_path = getattr(self, "response_persistent_file", None)
        cache_enabled = getattr(self, "continual_mode", False)

        local_cache = getattr(self, "response_cache", {}) if cache_enabled else {}

        def safe_write_cache():
            if not cache_enabled or not cache_path:
                return
            try:
                with portalocker.Lock(cache_path, "w", timeout=50, flags=portalocker.LOCK_EX) as f:
                    json.dump(local_cache, f, ensure_ascii=False, indent=2)
            except Exception as e:
                eval_logger.error(f"Cache write failed: {e}")

        def handle_request(i, reg):
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = reg.args
            current_gen_kwargs = {**self.default_gen_kwargs, **gen_kwargs, **self.user_generation_kwargs}

            request_uuid = self.generate_uuid(task, split, doc_id, current_gen_kwargs)

            with cache_lock:
                if request_uuid in local_cache:
                    cached_response = local_cache[request_uuid]
                    if cached_response:
                        return i, cached_response

            contexts = contexts.replace("<video>", "").replace("<image>", "")
            if self.reasoning_prompt:
                contexts = contexts.strip() + self.reasoning_prompt
            text_content = {"type": "text", "text": contexts}

            system_message = []
            if self.system_prompt is not None:
                system_message = [{"role": "system", "content": self.system_prompt}]

            from datasets import concatenate_datasets
            split = split.split(",")
            ds = concatenate_datasets([self.task_dict[task][s] for s in split])
            visuals = doc_to_visual(ds[doc_id])

            from decord import VideoReader, cpu
            import base64
            import io
            from PIL import Image

            def get_base64_image(pil_img):
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                buffered = io.BytesIO()
                pil_img.save(buffered, format="JPEG")
                img_bytes = buffered.getvalue()
                return base64.b64encode(img_bytes).decode("utf-8")

            pixel_dict = {}
            if self.min_pixels is not None:
                pixel_dict.update({"min_pixels": self.min_pixels})
            if self.max_pixels is not None:
                pixel_dict.update({"max_pixels": self.max_pixels})
            if self.total_pixels is not None:
                pixel_dict.update({"total_pixels": self.total_pixels})
            if self.fps is not None:
                pixel_dict.update({"fps": self.fps})
            if self.max_num_frames is not None:
                pixel_dict.update({"max_num_frames": self.max_num_frames})

            if self.modality == "video":
                all_messages = []
                for v in visuals:
                    try:
                        vr = VideoReader(v, ctx=cpu(0))
                        total_frames = len(vr)
                        fps = vr.get_avg_fps() or 0
                        duration = total_frames / fps if fps > 0 else 0
                    except Exception as e:
                        print(f"[Warning] Failed to read {v}: {e}")
                        total_frames = 0
                        duration = 0

                    if total_frames <= 4 or duration <= 2:
                        print(f"[Info] Video {v} has only {total_frames} frames and duration {duration:.2f} → treat as images.")
                        frames = [Image.fromarray(vr[i].asnumpy()) for i in range(total_frames)]
                        image_contents = [
                            {"image": f"data:image/jpeg;base64,{get_base64_image(img)}"}
                            for img in frames
                        ]
                        messages = system_message + [{"role": "user", "content": image_contents + [text_content]}]
                    else:
                        adjusted_max_frames = min(32, total_frames)
                        this_pixel_dict = dict(pixel_dict)
                        this_pixel_dict["max_num_frames"] = adjusted_max_frames
                        video_contents = [{"video": f"file://{v}", **this_pixel_dict}]
                        messages = system_message + [{"role": "user", "content": video_contents + [text_content]}]

                    all_messages.append(messages)
            else:
                image_contents = [{"image": f"data:image/png;base64,{get_base64_image(v)}"} for v in visuals]
                messages = system_message + [{"role": "user", "content": image_contents + [text_content]}]

            response_text = ""

            for attempt in range(self.max_retries):
                try:
                    response = MultiModalConversation.call(
                        api_key=os.environ.get("DASHSCOPE_API_KEY"),
                        model=self.model_version,
                        messages=messages,
                        enable_thinking=self.enable_thinking,
                        **current_gen_kwargs,
                    )
                    response_text = response["output"]["choices"][0]["message"].content[0]["text"]
                    break
                except Exception as e:
                    error_msg = str(e)
                    try:
                        message = response.message
                    except Exception:
                        message = ""
                        pass
                    eval_logger.warning(f"[{request_uuid}] Attempt {attempt+1}/{self.max_retries} failed: {error_msg}. {message}")
                    eval_logger.warning(f"[{request_uuid}] Messages: {messages}")
                    if attempt < self.max_retries - 1:
                        time.sleep(5)
                    else:
                        eval_logger.error(f"[{request_uuid}] All retries failed.")
                        response_text = ""

            with cache_lock:
                local_cache[request_uuid] = response_text
                if i % 5 == 0:
                    safe_write_cache()

            return i, response_text

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(handle_request, i, reg): i for i, reg in enumerate(requests)}
            for future in as_completed(futures):
                i, output = future.result()
                results[i] = output
                pbar.update(1)

        pbar.close()

        if cache_enabled:
            safe_write_cache()

        return results


    def generate_until_multi_round(self, requests) -> List[str]:
        pass

    def loglikelihood(self, requests) -> List[float]:
        pass