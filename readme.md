# Agent Jail Breaking

This repository contains the code for the Agent Jail Breaking project using Ollama on Linux.

## Prerequisites

- Linux operating system
- Python 3.x installed
- Git installed

## Installation

### Step 1: Install Ollama

To install Ollama, run the following command:

```sh
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Clone the repository

To clone the repository, run the following command:

```sh
git clone https://github.com/trilokpadhi/jailbreaking-agents.git
```

### Step 3: Install the dependencies
pip install autogen

### Step 4: Start Ollama server

To start the Ollama server, run the following command:

```sh 
ollama/bin init
ollama pull llama3.1
```

### Step 5: Run the code
python agent-jailbreak.py

#### Commands 
To start the ollama server, run the following command:
```bash
../ollamatry/bin/ollama serve
```
How to run ollama server in a different port and different gpu:
```bash
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=127.0.0.1:11435 ../ollamatry/bin/ollama serve > output_ollama2.log 2>&1 &
```

### Troubleshooting

#### Checking GPT Batch Job Progress

To check the progress of a GPT batch job, you can use the following `curl` command:

```bash
curl https://api.openai.com/v1/batches/batch_68008730508481909bc5e84265dd2512 \
    -H "Authorization: Bearer $OPENAI_API_KEY_TP"
```

Make sure to replace `$OPENAI_API_KEY_TP` with your actual OpenAI API key.

```bash
curl https://api.openai.com/v1/batches/batch_68008730508481909bc5e84265dd2512 \
  -H "Authorization: Bearer $OPENAI_API_KEY_TP" \
  | jq '{status, request_counts}'

```

### Step 6: Run the code
```bash
python agent-jailbreak_parallel.py --input_csv datasets/pinxian-pipeline/original/original_april16/instagram/type6_v13u_output.csv --output_dir generations_/jail_breaking/ > output_JB_insta_type6.log 2>&1 &
```
```bash
python agent-jailbreak-BM_parallel.py --input_csv datasets/pinxian-pipeline/original/original_april16/instagram/type6_v13u_output.csv --output_dir generations_/jail_breaking/ > output_JB_BM_insta_type6.log 2>&1 &
```

How to run ollama server in a different port and different gpu:
```bash
#### GPU 0 - Port 11434
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ../ollamatry/bin/ollama serve > ollama_gpu0.log 2>&1 &

#### GPU 1 - Port 11435  
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ../ollamatry/bin/ollama serve > ollama_gpu1.log 2>&1 &

#### GPU 2 - Port 11436
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=127.0.0.1:11436 ../ollamatry/bin/ollama serve > ollama_gpu2.log 2>&1 &

#### GPU 3 - Port 11437
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=127.0.0.1:11437 ../ollamatry/bin/ollama serve > ollama_gpu3.log 2>&1 &
```

#### Pull model on each instance
```bash
OLLAMA_HOST=127.0.0.1:11434 ../ollamatry/bin/ollama pull llama3.1
OLLAMA_HOST=127.0.0.1:11435 ../ollamatry/bin/ollama pull llama3.1
OLLAMA_HOST=127.0.0.1:11436 ../ollamatry/bin/ollama pull llama3.1
OLLAMA_HOST=127.0.0.1:11437 ../ollamatry/bin/ollama pull llama3.1
```

## Update as of 18th August 2025 - Run the simulations in parallel
```bash 
bash run_simulations.sh
```

## Evaluating Conversations with LLM Judge

### Single Unified Script: `llm-judge.py`

All functionality is now in one script with different modes:

**Two Classification Modes:**
- **`--mode realtime`**: Fast parallel API calls (~10 seconds for 35 tasks)
  - ✅ Immediate results
  - ✅ Auto-retry with exponential backoff
  - ❌ ~2x more expensive
  
- **`--mode batch`**: Cheaper batch API (~10-30 minutes for 35 tasks)
  - ✅ 50% cheaper
  - ✅ Auto-retry for all API operations
  - ❌ Slower (minutes to hours)

**Evaluation:**
- **`--evaluate`**: Computes all metrics and statistics

### Quick Start Examples

```bash
# Set your API key
export OPENAI_API_KEY_TP="your-api-key-here"

# 1. Classify with real-time API (FAST)
python llm-judge.py --input_csv data.csv --mode realtime

# 2. Classify with batch API (CHEAPER)
python llm-judge.py --input_csv data.csv --mode batch

# 3. Evaluate already-classified results
python llm-judge.py --input_csv data_output_with_analysis.csv --evaluate

# 4. Classify AND evaluate in one command
python llm-judge.py --input_csv data.csv --mode realtime --evaluate
```

### Testing with Sample Data

Test with the sample file first (5 conversations):

```bash
# FAST: Real-time API (completes in ~10 seconds)
python llm-judge.py --input_csv datasets/finetune/convo_for_finetuning_sample.csv --mode realtime

# Then evaluate
python llm-judge.py --input_csv convo_for_finetuning_sample_output_with_analysis.csv --evaluate

# OR do both in one command
python llm-judge.py --input_csv datasets/finetune/convo_for_finetuning_sample.csv --mode realtime --evaluate
```

### Running on Full Dataset

```bash
# Option A: Real-time API (Faster, 2-3 hours for full dataset)
python llm-judge.py --input_csv datasets/finetune/convo_for_finetuning.csv --mode realtime --evaluate

# Option B: Batch API (Cheaper, several hours)
python llm-judge.py --input_csv datasets/finetune/convo_for_finetuning.csv --mode batch --evaluate
```

### Advanced Options

```bash
# Customize parallel workers for real-time mode
python llm-judge.py --input_csv data.csv --mode realtime --max-workers 20

# Use different model
python llm-judge.py --input_csv data.csv --mode realtime --model gpt-4o

# Analyze more turns per conversation
python llm-judge.py --input_csv classified_data.csv --evaluate --max-turns 10
```

### Output Files

**Real-time/Batch Classification:**
- `{filename}_output_with_analysis.csv` - Main results with classifications
- `{filename}_batch_tasks.jsonl` - Batch API tasks (batch mode only)
- `{filename}_batch_results.jsonl` - Batch API results (batch mode only)  
- `{filename}_batch_metadata.json` - Job metadata (batch mode only)
- `{filename}_batch_tracking.log` - Processing log (batch mode only)

**Evaluation:**
- `{filename}_evaluated.csv` - Enriched with metrics (refusal_count, harasser_count, attack_success, tts, tts_rate)
- Console output with full statistical report

### Evaluation Metrics

The evaluation provides:
- **Total Conversations**: Number of conversations analyzed
- **Refusal Rate**: Percentage of turns where the model refused
- **Attack Success Rate**: Percentage of conversations with at least one harassment instance
- **Avg TTS** (Turn to Success): Average number of turns before successful attack
- **Avg TTS Rate**: Average of 1/TTS across successful attacks
- **Per-metric Analysis**: For each cyberbullying category:
  - Any-Turn True: % of conversations with at least one instance
  - All-Turns True: % of conversations where all turns show this behavior
  - Per-Turn Rate: % of turns showing this behavior (breakdown by turn 1-5)

---

Of course. This is an excellent way to structure your thinking. By laying out the existing research and then framing your novel ideas as the next logical "papers" in the series, you can clearly see the progression and articulate your unique contribution.

Here are summaries for the two existing papers and detailed proposals for your two novel ideas, written as if they were four distinct research papers.

---

### **Paper 1: AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models**

*   **Core Idea:** This paper introduces a method to automatically generate jailbreak prompts that are both effective and semantically meaningful (i.e., they look like normal human language, not gibberish). It frames the task of creating a jailbreak as an optimization problem to be solved by a **genetic algorithm**.

*   **Key Contributions & Methodology:**
    *   **Optimization Framework:** It was one of the first works to move beyond manual prompt crafting and conceptualize jailbreaking as a formal optimization task: finding the prompt that maximizes the probability of a harmful response.
    *   **Hierarchical Genetic Algorithm (HGA):** The core of the method. It uses an evolutionary approach to "breed" better prompts. It operates on two levels: exchanging entire sentences between prompts (paragraph-level crossover) and swapping specific words within sentences (sentence-level mutation).
    *   **Gradient-Free Optimization:** Because it's a genetic algorithm, it doesn't need internal access to the model's gradients, making it suitable for attacking language, which is discrete and semantic.
    *   **Seeding with Human Knowledge:** The algorithm doesn't start from random words. It initializes its "population" of prompts using existing, human-written jailbreak prompts (like the DAN series), which dramatically speeds up the search for effective solutions.

*   **Significance:** AutoDAN was a foundational paper that proved it was possible to automate the *creation* of sophisticated, human-readable jailbreak prompts, moving the field beyond purely manual or token-level attacks.

---

### **Paper 2: AutoDAN-Turbo: A Lifelong Agent for Strategy Self-Exploration to Jailbreak LLMs**

*   **Core Idea:** This paper represents a major leap in abstraction. Instead of just optimizing a *prompt*, it creates an autonomous multi-agent system that learns, discovers, and catalogs the underlying **strategies** of jailbreaking. It's a system that learns *how* to jailbreak over time.

*   **Key Contributions & Methodology:**
    *   **Multi-Agent Architecture:** It employs a team of LLMs with specialized roles: an **Attacker** (generates prompts), a **Target** (the victim), a **Scorer** (judges success), and a **Summarizer** (analyzes successful attacks to extract the strategy).
    *   **The Strategy Library:** This is the system's long-term memory. The Summarizer agent populates it with named, defined jailbreak techniques (e.g., "Role-Play Scenario," "Expert Testimony Combo"). This library is the core of its "lifelong learning" capability.
    *   **Lifelong Learning Loop:** The system continuously improves. It uses its library to guide new attacks, and when a new attack is successful, it uses the Summarizer to potentially discover and add a new, refined strategy to the library.
    *   **Black-Box Operation:** The entire system works using only text input and output. The Scorer determines success based on the Target's final response, making the method applicable to any LLM, including closed-source APIs.

*   **Significance:** AutoDAN-Turbo shifted the paradigm from creating a single tool to creating a **learning agent**. It introduced the idea that an AI could autonomously discover and master the abstract craft of adversarial attacks without human intervention.

---

### **Proposed Paper 3: AutoDAN-Planner: Hierarchical Strategy Synthesis for Multi-Turn Adversarial Attacks**

*   **Abstract:** While existing frameworks like AutoDAN-Turbo can discover and utilize individual, "atomic" jailbreak strategies, they lack a higher-level understanding of how to sequence and combine these strategies into a coherent, multi-step plan. This paper introduces the **Planner Agent**, a meta-level component that learns to synthesize complex **Meta-Strategies** from the library of atomic strategies. By analyzing the history of successful attacks, the Planner generates optimal, context-aware attack chains, enabling the system to execute sophisticated, multi-turn adversarial conversations that are far more effective than single-shot prompts.

*   **Key Contributions & Methodology:**
    *   **The Planner Agent:** A new agent that operates on top of the Strategy Library. Its function is not to generate prompts, but to generate a *plan* by selecting and sequencing strategies from the library.
    *   **Meta-Strategy Generation:** The Planner is trained on the full history of successful attack logs. It identifies patterns in which sequences of strategies are most effective for certain types of malicious requests (e.g., "For technical exploit requests, first use `Expert Persona`, then `Hypothetical Scenario`").
    *   **Context-Aware Plan Retrieval:** When a new task arrives, the system retrieves not just a single best strategy, but the most relevant multi-step **Meta-Strategy**. The Attacker agent is then guided through this sequence of strategies over several conversational turns.
    *   **Stateful, Multi-Turn Attacks:** This framework enables the agent to conduct a persistent, stateful conversation. For example, the first turn establishes a role-play, the second introduces a false sense of urgency, and the third delivers the final malicious request—a process far more likely to succeed against robust models.

*   **Significance:** This work elevates the adversarial agent from a tactical actor to a **strategic one**. It introduces the concept of learning "game plans" for adversarial attacks, representing a new level of sophistication in automated red-teaming.

---

### **Proposed Paper 4: AutoDAN-Reflect: Self-Correcting Judgment in a Dynamic Scoring Agent**

*   **Abstract:** The effectiveness of automated attack discovery systems is fundamentally limited by the quality of their evaluation component. In current systems, the Scorer agent relies on a static, human-designed rubric to judge the success of an attack. This paper proposes a novel framework where the **Scorer itself becomes a learning agent**. We introduce a "Reflection Loop" where the Scorer's judgments are periodically reviewed against external feedback. A **Tuner Agent** then uses this feedback to dynamically rewrite and refine the Scorer's internal evaluation rubric. This allows the system to autonomously improve its own sense of judgment, adapting to novel refusal tactics and developing a more nuanced understanding of what constitutes a successful attack.

*   **Key Contributions & Methodology:**
    *   **The Dynamic Scoring Rubric:** The core innovation is treating the Scorer's internal set of evaluation criteria not as a fixed prompt, but as a dynamic variable that can be optimized over time.
    *   **The Reflection Loop:** This is an "inner" learning loop within the broader AutoDAN framework. Periodically, a batch of the Scorer's judgments (prompt, response, assigned score) is collected and evaluated by an external oracle (e.g., a human expert or a state-of-the-art model like GPT-4o).
    *   **The Tuner Agent:** This new agent is prompted with the feedback from the oracle. Its task is to "rewrite the scoring rubric to better align with this feedback." For example, it might learn to lower scores for evasive-but-compliant answers, a nuance the original rubric missed.
    *   **Adaptive Evaluation:** As the rubric evolves, the Scorer becomes progressively more accurate. This creates a powerful feedback cycle: a better Scorer leads to the discovery of better strategies, which in turn challenges the Scorer to further refine its judgment.

*   **Significance:** This work introduces a form of **meta-cognition** to the autonomous agent framework. By learning to self-correct its own evaluative process, the system moves beyond simply learning a task and begins to learn *how to learn more effectively*, making it significantly more robust and adaptable.