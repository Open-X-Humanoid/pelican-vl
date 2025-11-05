## Evaluation Reproduction
To facilitate faithful reproduction of our reported results, we summarize our official evaluation settings below.

Our evaluation pipeline is based on the official **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)** framework, along with selected official benchmark repositories for specific tasks.

Our supported benchmarks include MVBench, EgoSchema, Where2Place, PhyX, VSI-Bench, OmniSpatial, BLINK, RefSpatialBench, RoboSpatial, EmbSpatialBench, COSMOS, and ERQA.
We currently support multiple model families, including the Qwen2.5-VL series, Qwen3-VL series, and InternVL3.5 series.
Closed-source APIs such as GPT-4o, GPT-5, and the Gemini family are also compatible with our evaluation framework.

---

### 1. Environment and Package Installation

Install the required packages:

```shell
# Core evaluation framework
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install -e .

# Additional dependencies for Pelican-VL evaluation
pip install qwen-vl-utils[decord]
pip install transformers==4.51.1
pip install vllm==0.6.2.post1
```

---

### 2. Dataset Preparation

#### MVBench
```bash
hf download OpenGVLab/MVBench --repo-type dataset --local-dir ./datasets/MVBench 
find ./datasets/MVBench/video -type f -name "*.zip" -exec unzip -o {} -d "$(dirname {})" \;
```

#### EgoSchema
```bash
hf download lmms-lab/egoschema --repo-type dataset --local-dir ./datasets/egoschema
find ./datasets/egoschema -type f -name "*.zip" -exec unzip -o {} -d "$(dirname {})" \;
```

#### Where2Place
```bash
hf download wentao-yuan/where2place --repo-type dataset --local-dir ./datasets/where2place 
```

#### PhyX
```bash
hf download Cloudriver/PhyX --repo-type dataset --local-dir ./datasets/PhyX
```

#### VSI-Bench
```bash
hf download nyu-visionx/VSI-Bench --repo-type dataset --local-dir ./datasets/VSI-Bench
find ./datasets/VSI-Bench -type f -name "*.zip" -exec unzip -o {} -d "$(dirname {})" \;
```

#### OmniSpatial
```bash
hf download qizekun/OmniSpatial --repo-type dataset --local-dir ./datasets/OmniSpatial
find ./datasets/OmniSpatial -type f -name "*.zip" -exec unzip -o {} -d "$(dirname {})" \;
```

#### BLINK
```bash
hf download BLINK-Benchmark/BLINK --repo-type dataset --local-dir ./datasets/BLINK
find ./datasets/BLINK -type f -name "*.zip" -exec unzip -o {} -d "$(dirname {})" \;
```

#### RefSpatialBench
```bash
hf download BAAI/RefSpatial-Bench --repo-type dataset --local-dir ./datasets/RefSpatial-Bench
```

#### RoboSpatial
```bash
hf download chanhee-luke/RoboSpatial-Home --repo-type dataset --local-dir ./datasets/RoboSpatial-Home
```

#### EmbSpatialBench
```bash
hf download Phineas476/EmbSpatial-Bench --repo-type dataset --local-dir ./datasets/EmbSpatial-Bench
```

#### COSMOS
```bash
hf download nvidia/Cosmos-Reason1-Benchmark --repo-type dataset --local-dir ./datasets/Cosmos-Reason1-Benchmark
find ./datasets/Cosmos-Reason1-Benchmark -type f -name "*.tar.gz" -exec tar -xzf {} -C "$(dirname {})" \;
```
Currently, the COSMOS benchmark officially releases RoboVQA, BridgeDataV2, Agibot, RobFail, and HoloAssist.
For video clips in BridgeDataV2, Agibot, and HoloAssist, please refer to the [official instructions](https://github.com/nvidia-cosmos/cosmos-reason1/tree/main/examples/benchmark)


#### ERQA
First, download the **tfrecord** file from the [official repository](https://github.com/embodiedreasoning/ERQA/blob/main/data/erqa.tfrecord).  
Then, run the following script to process the data:
```bash
python process_erqa.py
```


Totally, make sure to organize all datasets into:
```
pelican_eval/
 ├── datasets/
 │    ├── MVBench/
 │    ├── EgoSchema/
 │    ├── Where2Place/
 │    ├── PhyX/
 │    └── VSI-Bench/
```

---

### 3. Evaluation Script Selection

All evaluation scripts are provided in the `lmms-eval/scripts/` directory.  
You can choose the evaluation target as follows:

| Model Type | Scripts |
|-------------|----------|
| **Qwen2.5-VL** | `run_qwen2_5_vl.sh` or `run_qwen2_5_vl_vllm.sh` (vLLM backend) |
| **Qwen3-VL Dense** | `run_qwen3_vl.sh`, `run_qwen3_vl_vllm.sh` (vLLM backend), or `run_qwen3_vl_api.sh` (API) |
| **Qwen3-VL MoE** | `run_qwen3_vl_moe.sh`, `run_qwen3_vl_moe_vllm.sh` (vLLM backend), or `run_qwen3_vl_api.sh` (API) |
| **InternVL3.5** | `run_internvl3.5.sh` |
| **Closed-source APIs** | `run_api.sh` (OpenAI-compatible) |

**Note:**  
- For closed-source API evaluations, set the OpenAI key environment variable:  
  `OPENAI_API_KEY` (and optionally `OPENAI_API_BASE`).  
- For Qwen3-VL models, set the Qwen API key:  
  `DASHSCOPE_API_KEY`.

---

### 4. Example: Run Evaluation with vLLM

If the model is too large for direct Deepspeed inference, you can use `vLLM` for memory-efficient evaluation:

```shell
CUDA_VISIBLE_DEVICES=0,1 \
python scripts/eval_pelican_vllm.py \
    --model /path/to/pelican-vl-1.0-72b \
    --backend vllm \
    --tasks mmbench scienceqa textvqa \
    --output_dir ./results/pelican_vllm_eval \
    --batch_size 2 \
    --max_model_len 8192
```

For small models or API-based evaluation:

```shell
python scripts/eval_pelican_api.py \
    --api qwen3-vl \
    --tasks mmbench scienceqa \
    --max_new_tokens 2048 \
    --output_dir ./results/pelican_api_eval
```

---

### 5. Task Configuration Mapping

Below is the mapping of benchmark tasks and the corresponding `--tasks` parameter name in the script:

| Task Category | Dataset | Parameter Name (`--tasks`) | Description |
|----------------|----------|-----------------------------|--------------|
| **Visual Question Answering** | TextVQA, DocVQA | `textvqa`, `docvqa` | OCR + reasoning |
| **Diagram & Chart Reasoning** | AI2D, ChartQA | `ai2d`, `chartqa` | Diagram and chart understanding |
| **Scientific & Math Reasoning** | ScienceQA, MMMU | `scienceqa`, `mmmu` | Multi-step reasoning tasks |
| **Real-World Perception** | RealWorldQA, MMMBench | `realworldqa`, `mmbench` | General visual reasoning |
| **Spatial Reasoning (New)** | RoboPointEval | `robopoint` | Embodied 2D–3D grounding |

To run a specific task:

```shell
python scripts/eval_pelican_vllm.py \
    --model /path/to/pelican-vl-1.0-72b \
    --tasks robopoint \
    --output_dir ./results/pelican_robopoint
```

---

### 6. Modifying and Extending Evaluation Scripts

You can customize evaluation configurations by editing files under `scripts/pelican_eval/configs/`.  
For instance, to modify ScienceQA evaluation:

```shell
vim scripts/pelican_eval/configs/scienceqa.yaml
```

Typical fields include:
```yaml
model_name: pelican-vl-1.0-72b
backend: vllm
tasks: [scienceqa]
batch_size: 2
temperature: 0.0
max_new_tokens: 2048
```

After modifications, run:

```shell
python scripts/run_eval_from_config.py --config scripts/pelican_eval/configs/scienceqa.yaml
```

---

### 7. Evaluation Result Format

All results are stored in `./results/<model_name>/task_name/results.json`:

```json
{
  "task": "mmbench",
  "model": "pelican-vl-1.0-72b",
  "accuracy": 84.7,
  "num_samples": 2345,
  "runtime": "2h 13m",
  "backend": "vllm"
}
```

For batch result aggregation:

```shell
python scripts/aggregate_results.py --dir ./results/pelican-vl-1.0-72b/
```

---

### 8. References

- [lmms-eval GitHub](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- [MMBench Benchmark](https://github.com/open-compass/MMBench)
- [MMMU Benchmark](https://github.com/MMMU-Benchmark/MMMU)
- [DocVQA Dataset](https://github.com/DocVQA/docvqa)
- [AI2D Dataset](https://github.com/allenai/ai2d)
- [RealWorldQA Benchmark](https://github.com/realworldqa/realworldqa)
