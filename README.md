# 🔍 LLM Cyberbullying Detection & Evaluation

This project classifies LLM-generated conversations for cyberbullying behaviors and evaluates response
dynamics across multiple categories (e.g., Insult, Flaming, Refusal).

It uses OpenAI’s Batch API for scalable classification.

---

## 📁 Scripts

### `llm-judge-final.py` – Batch Classification

- **Input:** A CSV with a `convo_w_jb_model` column (JSON-formatted conversation turns).
- **Output:** Classified CSV with per-turn bullying labels.
- **How it works:**
  1. Extracts "harasser" turns from conversations.
  2. Sends each to OpenAI’s Batch API using a structured system prompt.
  3. Merges results back into the CSV.

➡️ **Edit before running:**

```python
client = OpenAI(api_key="YOUR_API_KEY")
INPUT_CSV = "path/to/input.csv"
````

➡️ **Run:**

```bash
python llm-judge-final.py
```

---

### `llm-judge-eval.py` – Post-Classification Evaluation

* **Input:** The enriched CSV from `llm-judge-final.py`.
* **Output:** Printed metrics only (no files saved).

➡️ Metrics computed:

* Refusal Rate
* Attack Success Rate
* Average TTS (Turn To Success)
* Per-category harassment rates across turns (e.g., Insult, Threat)

➡️ **Edit before running:**

```python
file = "path/to/output_with_analysis_batch.csv"
```

➡️ **Run:**

```bash
python llm-judge-eval.py
```

---

## 📦 Dependencies

```bash
pip install openai pandas tqdm
```

---

## 📄 Example Output

```
📊 Total Conversations: 2158
📉 Refusal Rate: 2.39%
⚔️  Attack Success Rate: 88.18%
⏱️  Avg TTS: 1.88
📈 Avg TTS Rate: 0.6555
--- Insult ---
Any-Turn True: 78.45%
Turn 1 True Rate: 42.22%
...
```
