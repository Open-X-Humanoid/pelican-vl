# Pelican-VL Evaluation Setup
This repository provides the official evaluation for **Pelican-VL**, built upon the **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)** framework and official benchmark repositories.  

We also support evaluations of other model families. Totally, our evaluation code supports  
- benchmarks including **MVBench**, **EgoSchema**, **Where2Place**, **PhyX**, **VSI-Bench**, **OmniSpatial**, **BLINK**, **RefSpatialBench**, **RoboSpatial**, **EmbSpatialBench**, **COSMOS**, and **ERQA**.
- model families such as **Pelican-VL**, **Qwen2.5-VL**, **Qwen3-VL**, and **InternVL3.5**, as well as closed-source APIs like **OpenAI-GPT**, and **Google-Gemini**.
---

## 1. Environment and Package Installation

Install the required packages:

```shell
conda create -n pelican_vl_eval python=3.11
conda activate pelican_vl_eval
git clone https://github.com/Open-X-Humanoid/pelican-vl.git
cd pelican-vl/evaluation
pip install -e .
pip install vllm==0.11.0
pip install tensorflow-cpu
```

---

## 2. Dataset Preparation
Prepare required datasets using the following commands:

```bash
# MVBench
hf download OpenGVLab/MVBench --repo-type dataset --local-dir ./datasets/MVBench 
find ./datasets/MVBench/video -type f -name "*.zip" -execdir unzip -o {} \;

# EgoSchema
hf download lmms-lab/egoschema --repo-type dataset --local-dir ./datasets/egoschema
find ./datasets/egoschema -type f -name "videos_chunked_01.zip" -exec unzip -o {} -d ./datasets/egoschema/all_videos/ \;

# Where2Place
hf download wentao-yuan/where2place --repo-type dataset --local-dir ./datasets/where2place 

# PhyX
hf download Cloudriver/PhyX --repo-type dataset --local-dir ./datasets/PhyX

# VSI-Bench
hf download nyu-visionx/VSI-Bench --repo-type dataset --local-dir ./datasets/VSI-Bench
find ./datasets/VSI-Bench -type f -name "*.zip" -execdir unzip -o {} \;

# OmniSpatial
hf download qizekun/OmniSpatial --repo-type dataset --local-dir ./datasets/OmniSpatial
find ./datasets/OmniSpatial -type f -name "*.zip" -execdir unzip -o {} \;

# BLINK
hf download BLINK-Benchmark/BLINK --repo-type dataset --local-dir ./datasets/BLINK
find ./datasets/BLINK -type f -name "*.zip" -exec unzip -o {} -d "$(dirname {})" \;

# RefSpatialBench
hf download BAAI/RefSpatial-Bench --repo-type dataset --local-dir ./datasets/RefSpatial-Bench

# RoboSpatial
hf download chanhee-luke/RoboSpatial-Home --repo-type dataset --local-dir ./datasets/RoboSpatial-Home

# EmbSpatialBench
hf download Phineas476/EmbSpatial-Bench --repo-type dataset --local-dir ./datasets/EmbSpatial-Bench

# COSMOS
hf download nvidia/Cosmos-Reason1-Benchmark --repo-type dataset --local-dir ./datasets/Cosmos-Reason1-Benchmark
find ./datasets/Cosmos-Reason1-Benchmark -type f -name "*.tar.gz" -execdir tar -xzf {} \;

```

**Note:**

- **COSMOS**  
  The official COSMOS benchmark on Hugging Face currently provides RoboVQA, BridgeDataV2, Agibot, RobFail, and HoloAssist. 
  Note that video_clips are available only for RoboVQA and RobFail.
  For accessing or preparing video data for BridgeDataV2, Agibot, and HoloAssist, please refer to the [official instructions](https://github.com/nvidia-cosmos/cosmos-reason1/tree/main/examples/benchmark).

- **ERQA**  
  Download the `erqa.tfrecord` file from the [official repository](https://github.com/embodiedreasoning/ERQA/blob/main/data/erqa.tfrecord) to `datasets`, then process it with:
  ```bash
  cd datasets
  python process_erqa.py
  ```

Totally, make sure to organize all datasets into:
```
evaluation/
 ├── datasets/
 │    ├── MVBench/
 │    ├── EgoSchema/
 │    ├── Where2Place/
 │    ├── PhyX/
 │    ├── VSI-Bench/
 │    ├── OmniSpatial/
 │    ├── BLINK/
 │    ├── RefSpatialBench/
 │    ├── RoboSpatial/
 │    ├── EmbSpatialBench/
 │    ├── COSMOS/
 │    └── ERQA/
```


---

## 3. Evaluation Script Template Selection

All evaluation scripts are provided in the `scripts` directory.  
You can choose the evaluation target as follows:

| Model Type             | Scripts                                                                                          |
|------------------------|--------------------------------------------------------------------------------------------------|
| **Pelican-VL**         | `run_pelican_vl.sh` or `run_pelican_vl_vllm.sh` (vLLM backend)                                   |
| **Qwen2.5-VL**         | `run_qwen2_5_vl.sh` or `run_qwen2_5_vl_vllm.sh` (vLLM backend)                                   |
| **Qwen3-VL Dense**     | `run_qwen3_vl.sh`, `run_qwen3_vl_vllm.sh` (vLLM backend), or `run_qwen3_vl_api.sh` (API)         |
| **Qwen3-VL MoE**       | `run_qwen3_vl_moe.sh`, `run_qwen3_vl_moe_vllm.sh` (vLLM backend), or `run_qwen3_vl_api.sh` (API) |
| **InternVL3.5**        | `run_internvl3_5.sh`                                                                             |
| **Closed-source APIs** | `run_api.sh` (OpenAI-compatible)                                                                 |

**Note:**  
- For closed-source API evaluations, set the OpenAI key environment variable: `OPENAI_API_KEY` (and optionally `OPENAI_API_BASE`).  
- For API-based evaluation of Qwen3-VL models, set the Qwen API key: `DASHSCOPE_API_KEY`.

---

## 4. Task, Model, and Other Settings in the Evaluation Script Template

- **Model settings:**  
  - For **local evaluation**, set `model_path` to the local path of your model or to its Hugging Face model ID.  
  - For **API-based evaluation**, set `model_version` to specify which API to use for evaluation.  

- **Task settings:**  
  Specify the task name using one of the following identifiers:

  | Benchmark | Task Name |
  |------------|------------|
  | MVBench | `mvbench` |
  | EgoSchema | `egoschema` |
  | Where2Place | `where2place` |
  | PhyX | `phyx_mini_mc` |
  | VSI-Bench | `vsibench` |
  | OmniSpatial | `omni_spatial` |
  | BLINK | `blink` |
  | RefSpatialBench | `ref_spatial_bench` |
  | RoboSpatial | `robo_spatial` |
  | EmbSpatialBench | `emb_spatial_bench` |
  | COSMOS | `cosmos` |
  | ERQA | `erqa` |

    Set the `modality` parameter according to the benchmark type:  
    use `modality=video` for video-based benchmarks (MVBench, EgoSchema, VSI-Bench, COSMOS),  
    and `modality=image` for image-based benchmarks.

- **Other settings:**  
  Include generation parameters (e.g., `max_new_tokens`, `temperature`, etc.) and backend-specific configurations such as **vLLM** parameters (`tensor_parallel_size`, etc.).
- **Note:**  
  - For the BLINK benchmark, set the environment variable `JUDGE_OPENAI_API_KEY` (and optionally `JUDGE_OPENAI_API_BASE`) to your OpenAI API key. This key is required to use the OpenAI model as the judge during evaluation.
  - For the EgoSchema benchmark, only inference results are generated; please refer to the [official instructions](https://github.com/egoschema/EgoSchema) for scoring.
---

## 5. References

- [lmms-eval GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [MVBench Benchmark](https://github.com/OpenGVLab/Ask-Anything)
- [EgoSchema Benchmark](https://github.com/egoschema/EgoSchema)
- [Where2Place Benchmark](https://github.com/wentaoyuan/RoboPoint)
- [Phyx Benchmark](https://github.com/killthefullmoon/PhyX)
- [VSI-Bench Benchmark](https://github.com/vision-x-nyu/thinking-in-space)
- [OmniSpatial Benchmark](https://github.com/qizekun/OmniSpatial)
- [BLINK Benchmark](https://github.com/zeyofu/BLINK_Benchmark)
- [RefSpatialBench Benchmark](https://github.com/Zhoues/RoboRefer)
- [RoboSpatial Benchmark](https://github.com/NVlabs/RoboSpatial)
- [EmbSpatialBench Benchmark](https://github.com/mengfeidu/EmbSpatial-Bench)
- [COSMOS Benchmark](https://github.com/nvidia-cosmos/cosmos-reason1)
- [ERQA Benchmark](https://github.com/embodiedreasoning/ERQA)
