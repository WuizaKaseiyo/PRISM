# ACE: Agentic Context Engineering — Detailed Paper Summary
 
**Venue:** ICLR 2026

---

## 1. Problem Formulation

The paper addresses the challenge of **context adaptation** for LLMs — improving model behavior by modifying inputs (instructions, strategies, evidence) rather than updating model weights. This is relevant across system prompts, agent memory, and factual evidence used in LLM applications.

Two fundamental limitations of existing context adaptation methods are identified:

- **Brevity Bias:** Prompt optimizers tend to collapse toward short, generic instructions, discarding domain-specific heuristics, tool-use guidelines, and common failure modes. For example, GEPA highlights brevity as a strength, but this abstraction can omit critical details needed for agents and knowledge-intensive tasks. Iterative optimization often produces near-identical generic prompts, narrowing the search space and propagating recurring errors.

- **Context Collapse:** When an LLM is tasked with fully rewriting accumulated context at each adaptation step, it tends to compress it into much shorter, less informative summaries. The paper demonstrates this empirically on AppWorld: at step 60, context contained 18,282 tokens with 66.7% accuracy, but at the next step it collapsed to 122 tokens with accuracy dropping to 57.1% — worse than the 63.7% baseline without any adaptation.

The core argument is that contexts should function as **comprehensive, evolving playbooks** — detailed, inclusive, and rich with domain insights — rather than concise summaries. LLMs, unlike humans, are more effective when provided with long, detailed contexts and can distill relevance autonomously at inference time.

---

## 2. Method: ACE (Agentic Context Engineering)

ACE is a framework for scalable context adaptation in both **offline** (system prompt optimization) and **online** (test-time memory adaptation) settings. It treats contexts as evolving playbooks that accumulate, refine, and organize strategies over time.

### 2.1 Architecture: Three Specialized Roles

Building on the agentic design of Dynamic Cheatsheet, ACE introduces a structured division of labor across three components:

1. **Generator:** Produces reasoning trajectories for new queries. It receives the current context playbook and a query, then generates a solution trajectory. During execution, the Generator highlights which context bullets were useful or misleading, providing feedback for subsequent stages.

2. **Reflector:** Critiques the Generator's traces to extract concrete lessons. It analyzes what went wrong (or right), identifies root causes of errors, and proposes corrective insights. The Reflector can optionally iterate through multiple refinement rounds (up to 5). Critically, the Reflector is a **separate component** from the Curator — this separation of evaluation/insight extraction from curation is a key design contribution that improves context quality.

3. **Curator:** Synthesizes the Reflector's lessons into compact **delta entries** (small sets of candidate bullets). These are merged deterministically into the existing context by lightweight, non-LLM logic. The Curator identifies only new insights missing from the current playbook, avoids redundancy, and formats additions as structured JSON operations.

### 2.2 Incremental Delta Updates

A core design principle is representing context as a collection of **structured, itemized bullets** rather than a monolithic prompt. Each bullet consists of:

- **Metadata:** A unique identifier and counters tracking how often it was marked helpful or harmful.
- **Content:** A small unit capturing a reusable strategy, domain concept, or common failure mode.

Instead of regenerating contexts in full, ACE produces compact **delta contexts** — small sets of candidate bullets distilled by the Reflector and integrated by the Curator. This enables three key properties:

- **Localization:** Only relevant bullets are updated.
- **Fine-grained retrieval:** The Generator can focus on the most pertinent knowledge.
- **Incremental adaptation:** Efficient merging, pruning, and de-duplication during inference.

Because updates are itemized and localized, multiple deltas can be merged in parallel, enabling batched adaptation at scale.

### 2.3 Grow-and-Refine Mechanism

Beyond incremental growth, ACE ensures contexts remain compact and relevant through periodic or lazy refinement:

- Bullets with new identifiers are **appended**.
- Existing bullets are **updated in place** (e.g., incrementing helpful/harmful counters).
- A **de-duplication step** prunes redundancy by comparing bullets via semantic embeddings.

Refinement can be performed **proactively** (after each delta) or **lazily** (only when the context window is exceeded), depending on application requirements for latency and accuracy.

### 2.4 Multi-Epoch Adaptation

ACE supports revisiting the same queries across multiple epochs to progressively strengthen the context. The maximum number of epochs in offline adaptation is set to 5.

### 2.5 Supervision Flexibility

ACE can operate both with and without ground-truth labels. When labels are unavailable, it leverages **natural execution feedback** (e.g., code execution success/failure, environment signals) to guide the Reflector and Curator.

### 2.6 Context Organization

The generated playbook is organized into structured sections, including:

- **Strategies and Hard Rules** — domain-specific heuristics and policies.
- **Useful Code Snippets and Templates** — reusable code patterns.
- **Troubleshooting and Pitfalls** — known failure modes and workarounds.
- **APIs to Use for Specific Information** — guidance on which APIs to call.
- **Verification Checklist** — post-execution validation steps.
- **Formulas and Calculations** — domain-specific computation rules.

---

## 3. Experiments

### 3.1 Tasks and Datasets

**Agent Benchmark:**
- **AppWorld:** A suite of autonomous agent tasks involving API understanding, code generation, and environment interaction. It provides a realistic execution environment with common applications (email, file system, etc.) and tasks at two difficulty levels (normal and challenge). The best system on the public leaderboard at submission time achieved only 60.3% average accuracy.

**Domain-Specific Benchmarks (Financial Analysis):**
- **FiNER:** Financial Numeric Entity Recognition requiring labeling tokens in XBRL documents with one of 139 fine-grained entity types.
- **Formula:** Extracting values from structured XBRL filings and performing computations to answer financial queries (numerical reasoning).

### 3.2 Baselines

- **Base LLM:** DeepSeek-V3.1 with default prompts, no context engineering.
- **ICL (In-Context Learning):** Few-shot/many-shot demonstrations in the input prompt.
- **MIPROv2:** Bayesian prompt optimizer jointly optimizing instructions and demonstrations.
- **GEPA:** Genetic-Pareto reflective prompt evolution using execution traces.
- **Dynamic Cheatsheet (DC):** Test-time adaptive external memory of reusable strategies (cumulative mode).

### 3.3 Key Results

**AppWorld Agent Benchmark:**

| Method | GT Labels | Average Accuracy |
|--------|-----------|-----------------|
| ReAct (Base) | — | 42.4% |
| ReAct + ICL | ✓ | 46.0% |
| ReAct + GEPA | ✓ | 46.4% |
| ReAct + DC (CU) | ✗ | 51.9% |
| ReAct + ACE (offline) | ✓ | 59.4% |
| ReAct + ACE (offline, no GT) | ✗ | 57.2% |
| ReAct + ACE (online) | ✗ | 59.5% |

- ACE outperforms baselines by an average of **+10.6%** on agents.
- Without ground-truth labels, ACE still achieves +14.8% over the ReAct baseline.
- On the AppWorld leaderboard, ACE (59.4%) matches IBM CUGA (60.3%, powered by GPT-4.1) despite using a smaller open-source model (DeepSeek-V3.1). With online adaptation, ACE surpasses IBM CUGA by 8.4% TGC on the harder test-challenge split.

**Financial Analysis Benchmarks:**

| Method | GT Labels | FiNER Acc | Formula Acc | Average |
|--------|-----------|-----------|-------------|---------|
| Base LLM | — | 70.7% | 67.5% | 69.1% |
| ICL | ✓ | 72.3% | 67.0% | 69.6% |
| MIPROv2 | ✓ | 72.4% | 69.5% | 70.9% |
| GEPA | ✓ | 73.5% | 71.5% | 72.5% |
| ACE (offline) | ✓ | 78.3% | 85.5% | 81.9% |
| ACE (online) | ✓ | 76.7% | 76.5% | 76.6% |

- ACE delivers an average gain of **+8.6%** over strong baselines on financial analysis.
- Offline ACE with GT labels surpasses the best baseline (GEPA) by +9.4% on average.

### 3.4 Ablation Studies (AppWorld)

| Configuration | Average Accuracy |
|---------------|-----------------|
| ACE w/o Reflector or multi-epoch | 55.1% |
| ACE w/o multi-epoch | 56.8% |
| Full ACE | 59.4% |
| ACE online (no warmup) | 56.1% |
| ACE online + offline warmup | 59.5% |

- The Reflector contributes +1.7% over its absence.
- Multi-epoch refinement contributes an additional +2.6%.
- Offline warmup before online adaptation adds +3.4%.

### 3.5 Cost and Speed Analysis

**Offline (AppWorld):**
- ACE achieves **82.3% reduction** in adaptation latency vs. GEPA (9,517s vs. 53,898s).
- ACE requires **75.1% fewer rollouts** (357 vs. 1,434).

**Online (FiNER):**
- ACE achieves **91.5% reduction** in adaptation latency vs. DC (5,503s vs. 65,104s).
- ACE reduces token dollar cost by **83.6%** ($2.9 vs. $17.7).

On average, ACE achieves **86.9% lower adaptation latency** than existing adaptive methods.

---

## 4. Contributions

1. **Identifies two fundamental failure modes** of existing context adaptation: brevity bias and context collapse, with empirical demonstration of context collapse on AppWorld.

2. **Proposes ACE**, a framework treating contexts as evolving playbooks with three specialized roles (Generator, Reflector, Curator) that accumulate, refine, and organize strategies through a modular workflow.

3. **Introduces incremental delta updates** that replace monolithic context rewrites with localized, structured edits, preventing context collapse and reducing latency/cost.

4. **Introduces the grow-and-refine mechanism** that balances context expansion with redundancy control via semantic de-duplication.

5. **Demonstrates label-free adaptation:** ACE can self-improve using only natural execution feedback (code execution outcomes, environment signals) without requiring ground-truth labels.

6. **Achieves state-of-the-art results** on AppWorld (matching the top-ranked production agent using a smaller open-source model) and financial analysis benchmarks, with significant cost and latency reductions.

7. **Supports both offline and online adaptation**, enabling flexible deployment across system prompt optimization and test-time memory adaptation.

---

## 5. Limitations

1. **Dependence on Reflector quality:** ACE relies on a reasonably strong Reflector to extract meaningful insights. If the Reflector fails to produce useful lessons, the context may become noisy or harmful. In domains where no model can extract useful insights, the resulting context will lack them.

2. **Feedback signal dependency:** Without reliable feedback signals (ground-truth labels or execution outcomes), both ACE and other adaptive methods like DC may degrade. The constructed context can be polluted by spurious or misleading signals, as observed in experiments where removing GT labels caused performance drops on some tasks.

3. **Not universally beneficial:** Not all applications require rich contexts. Tasks like HotPotQA benefit more from concise, high-level instructions than long contexts. Games with fixed strategies (e.g., Game of 24) may only need a single reusable rule, making additional context redundant.

4. **Best suited for specific settings:** ACE is most beneficial in settings demanding detailed domain knowledge, complex tool use, or environment-specific strategies that go beyond what is embedded in model weights or simple system instructions.

5. **Longer contexts at inference:** Although ACE produces longer contexts than methods like GEPA, the authors argue this does not translate to linearly higher cost due to modern KV cache reuse, compression, and offloading techniques — but this remains an infrastructure-dependent consideration.