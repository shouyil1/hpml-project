# Agentic AI Pipeline Optimization for IT Operations

**Course:** COMSW6998 High Performance Machine Learning, Columbia University  

**Team Members:** 
* Shouyi Li 
* Kexin Wang 
* Moxi Yuan 
* Linshuo Li 

**Mentor:** 
* Ruchi Mahindru

---

## Project Description

Agentic AI systems are increasingly capable of automating complex IT operations like Site Reliability Engineering (SRE) and software maintenance. However, existing benchmarks primarily focus on **accuracy**, without critical operational constraints like **latency, throughput, and cost**.

This project bridges that gap by optimizing the inference pipeline for autonomous agents. We utilize **vLLM** on commodity hardware to accelerate **ITBench SRE Agents** and **SWE-agents**. By leveraging system-level optimizations such as **PagedAttention, Prefix Caching, Chunked Prefill, CUDA Graphs, FlashAttention, Continuous Batching, and Quantization (AWQ)**, we achieved a **25x reduction in total task runtime** and an **18x reduction in execution cost** compared to standard baselines and commercial APIs.

## Repository Outline

This repository is structured as a monorepo containing the optimized agent frameworks and evaluation scenarios.

```text
.
├── ITBench-SRE-Agent/      # The SRE agent logic based on CrewAI
├── ITBench-Scenarios/      # Kubernetes-based incident scenarios for evaluation
└── SWE-agent/              # Modified SWE-agent code for vLLM integration
````

  * **ITBench-SRE-Agent:** Contains the source code for the SRE agent that interacts with Kubernetes clusters to diagnose and resolve incidents.
  * **ITBench-Scenarios:** Scripts to spin up local Kubernetes (Kind) clusters and inject faults (e.g., CrashLoopBackOff incidents).
  * **SWE-agent:** The official SWE-agent repository, modified to support local vLLM inference endpoints.

-----

## Example Commands to Execute

We deployed our experiments on a **GCP g2-standard-12 instance (1 NVIDIA L4 GPU, 24GB VRAM)**. Below are the commands to reproduce our setups.

### 1\. Starting the Inference Server (vLLM)

We evaluated three configurations. The most optimal setup was vLLM Full.

**vLLM Server (Example Command):**

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model "Qwen/Qwen2.5-14B-Instruct-AWQ" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --host 0.0.0.0 \
  --port 8000
```

*Note: To run the "Serial" baseline, add the flag `--max-num-seqs 1`.*

### 2\. Running the SRE Agent (ITBench)

The SRE agent connects to the vLLM server to solve a Kubernetes incident.

**Step A: Start the Incident Environment**

```bash
cd ITBench-Scenarios/sre
source venv/bin/activate
# Start Incident #1 (e.g., CrashLoopBackOff)
INCIDENT_NUMBER=1 make start_incident
```

**Step B: Run the Agent (Example Command)**

```bash
# In a separate terminal
docker run --network host \
  --mount type=bind,src="$(pwd)",target=/app/lumyn \
  -e KUBECONFIG=/app/lumyn/config \
  -e OPENAI_API_BASE="http://localhost:8000/v1" \
  -it itbench-sre-agent \
  crewai run
```

### 3\. Running the SWE-agent

For software engineering tasks, we used the SWE-agent framework connected to our vLLM endpoint.

```bash
# 1. Start the vLLM container (Example Command)
docker run -d --runtime nvidia --gpus all \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-Coder-14B-Instruct-AWQ \
    --quantization awq

# 2. Run the Benchmark (SWE-bench Lite)
sweagent run-batch \
  --config config/vllm_qwen.yaml \
  --instances.type swe_bench \
  --instances.subset lite \
  --num_workers 1
```

-----

## Results and Observations

We evaluated the performance impact of system-level optimizations on long-context agent workflows.

### 1\. End-to-End Performance (ITBench SRE)

Switching from a naive baseline to vLLM resulted in a massive reduction in total runtime.

| Configuration | Optimization Added | Latency (ms/tok) | Speed (tok/s) | Total Runtime |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (HF)** | None | \~121.0 | \~8.2 | \~10,800 s (\~3 hrs) |
| **vLLM Serial** | PagedAttention | 40.1 | 24.9 | **422 s (\~7 mins)** |
| **vLLM Full** | + Continuous Batching | 43.0 | 23.3 | 446 s |
| **vLLM + Spec** | + Speculative Decoding | 43.9 | 23.0 | 443 s |

*Figure 1: Comparison of total runtime across configurations. Note the 25x speedup provided by vLLM.*

### 2\. Cost Efficiency

Self-hosting on an L4 GPU significantly reduces the "Capability Tax" of using commercial APIs for prefill-heavy workloads.

| Metric | Value |
| :--- | :--- |
| Total Prompt Tokens | 648,971 (98.7% of traffic) |
| Total Gen Tokens | 8,301 |
| **Est. Cost (Self-Hosted L4)** | **\~$0.09** |
| **Est. Cost (GPT-4o API)** | **~$1.70** |

### 3\. Key Observations

  * **The "Prefill Dominance":** 98.7% of the workload is reading logs. vLLM's **Prefix Caching** allows the agent to skip re-processing these logs at every step, driving the 25x speedup.
  * **Memory is King:** The naive baseline crashed at \~8k tokens due to memory fragmentation. **PagedAttention** reduced fragmentation to \<4%, allowing safe context handling up to 28k tokens.
  * **Batch Size of One:** Standard optimizations like Continuous Batching provided no gain because autonomous agents operate sequentially (Thought → Action → Observation), leaving the GPU idle between steps.

-----

## W\&B Waiver

*Regarding Weights & Biases (W\&B): As per discussion with the instructor, we have thoroughly documented our experiments via system logs, timing metrics, and the tables provided above. Therefore, a W\&B project board is not included.*

-----

## Acknowledgements

This project builds upon the following open-source repositories:

  * [ITBench](https://www.google.com/search?q=https://github.com/ITBench)
  * [SWE-agent](https://github.com/princeton-nlp/SWE-agent)
  * [vLLM](https://github.com/vllm-project/vllm)

