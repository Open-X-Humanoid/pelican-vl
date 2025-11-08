import importlib
import os
import sys

import hf_transfer
from loguru import logger

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    "internvl3": "InternVL3",
    "openai_compatible": "OpenAICompatible",
    "qwen3_vl_api": "Qwen3_VL_API",
    "qwen3_vl_moe": "Qwen3_VL_MOE",
    "qwen3_vl": "Qwen3_VL",
    "qwen2_5_vl": "Qwen2_5_VL",
}


def get_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    if "." not in model_class:
        model_class = f"lmms_eval.models.{model_name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise


if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            AVAILABLE_MODELS[model_name] = f"{plugin}.models.{model_name}.{model_class}"
