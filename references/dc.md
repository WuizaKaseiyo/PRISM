# Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory — Detailed Paper Summary

**Paper:** Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory  
**Authors:** Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, James Zou  
**Affiliations:** Stanford University, Together AI  
**Venue:** arXiv preprint, April 2025

---

## 1. Problem Formulation

The paper addresses a fundamental limitation of current LLMs at inference time: **each input query is processed in isolation**, without retaining insights from previous attempts. Models approach each new problem de novo, often re-deriving the same insights and re-committing the same errors. This contrasts sharply with human cognition, which is built on incremental learning — continuously internalizing new experiences and solutions into a persistent mental model.

The key challenges are:

- **No persistent memory across queries:** Deployed LLMs are fixed and retain no explicit or implicit memory of past questions, successes, or mistakes during inference.
- **Naive history approaches fail:** Simply appending entire conversation histories leads to context window overflow, dilutes crucial insights, and hampers retrieval efficiency.
- **Fine-tuning is impractical for adaptation:** Directly updating model weights at test time is computationally expensive and often infeasible for black-box commercial APIs (e.g., GPT-4, Claude).

The paper frames the solution as an **online learning** problem: given a sequence of inputs (x₁, x₂, ..., xₙ) sampled from an unknown test distribution D_test, the goal is to enable the model to improve its performance over time by maintaining and curating an external memory that evolves with each query — all without any gradient-based parameter updates.

---

## 2. Method: Dynamic Cheatsheet (DC)

DC endows a black-box LLM with a persistent, evolving, non-parametric external memory at inference time. The memory stores reusable strategies, solution sketches, code snippets, and problem-solving insights. The framework operates entirely without modifying model weights and requires no ground-truth labels or human feedback.

### 2.1 Core Building Blocks

The DC framework consists of two core modules that can operate on the same LM (prompted differently) or on separate LMs:

**Solution Generator (Gen):**  
At step i, the Generator receives the new query xᵢ and the current memory state Mᵢ, and produces a candidate solution:

> ỹᵢ = Gen(xᵢ, Mᵢ)

The memory Mᵢ conditions the model to reuse or adapt previously stored solutions, insights, techniques, or heuristics.

**Memory Curator (Cur):**  
After the Generator produces its answer ỹᵢ, the Curator updates the memory:

> Mᵢ₊₁ = Cur(Mᵢ, xᵢ, ỹᵢ)

The Curator considers three factors: (i) the usefulness and generalizability of the newly produced answer — if correct or insightful, it is distilled into a form suitable for later reference; (ii) refinement or removal of existing memory entries — if an existing entry was incorrect or superseded by a better strategy, the Curator may remove or update it; and (iii) clarity and compactness of the entire memory — entries are consolidated to retain succinct, high-impact references and heuristics.

Critically, the Curator does **not** have access to ground-truth labels. It must assess the correctness and efficiency of solutions by itself before updating the memory.

### 2.2 DC-Cu (Cumulative) Variant

Under DC-Cu, the system first performs solution generation based on the current memory (Gen), and then updates the memory (Cur), cumulatively expanding and refining memory items. DC-Cu does **not** contain a retrieval component. The workflow is:

1. Generate solution: ỹᵢ = Gen(xᵢ, Mᵢ₋₁)
2. Update memory: Mᵢ = Cur(Mᵢ₋₁, xᵢ, ỹᵢ)

Two potential drawbacks of DC-Cu: (a) it updates memory **after** processing a query, so the model cannot incorporate insights from the current query while reasoning; and (b) it does not store or revisit past input-output pairs unless explicitly retained.

### 2.3 DC-RS (Retrieval & Synthesis) Variant

DC-RS addresses DC-Cu's limitations by modifying the sequence of memory updates and introducing a retrieval mechanism (Retr). The retrieval mechanism ranks historical inputs based on cosine similarity (using OpenAI's text-embedding-3-small model) with the current query, selecting the top-k (k=3) most relevant past examples along with their generated solutions.

The DC-RS workflow is:

1. **Retrieve:** Rᵢ = Retr(xᵢ, {(xⱼ, ỹⱼ)}_{j<i}, k) — retrieve top-k similar past input-output pairs
2. **Curate/Update memory:** Mᵢ = Cur(Mᵢ₋₁, xᵢ, Rᵢ) — refine memory **before** generating a response, using retrieved examples
3. **Generate:** ỹᵢ = Gen(xᵢ, Mᵢ) — produce the solution with the updated memory

The key difference from DC-Cu is that DC-RS refines memory **before** responding and can retrieve prior cases when needed, enhancing adaptability and reasoning efficiency.

### 2.4 Memory Structure

The memory is organized as a structured repository with clearly defined sections:

- **Reusable Code Snippets and Solution Strategies** — well-documented code and worked-out solutions
- **General Problem-Solving Heuristics** — meta-reasoning strategies
- **Optimization Techniques & Edge Cases**
- **Specialized Knowledge & Theorems**

Each memory item includes a description, an example, and a usage counter that tracks how often the strategy has been successfully applied. Memory items are tagged with references to the problems that contributed to them (e.g., Q1, Q2, Q6). The cheatsheet is capped at approximately 2,000–2,500 words.

### 2.5 Self-Curation Properties

The memory is **self-curated**, meaning the model itself decides what to store, discard, and refine. The curation focuses on concise, transferable snippets rather than entire transcripts, preventing context ballooning. The Curator is instructed to preserve only high-value strategies, discard redundant or problem-specific details, and ensure previously effective solutions remain accessible while incorporating new methods.

---

## 3. Experiments

### 3.1 Tasks and Datasets

The evaluation focuses on challenging tasks where state-of-the-art LLMs still face limitations, prioritizing tasks that demand multi-step reasoning, heuristic search, and strategic adaptation:

- **AIME 2020–2025:** American Invitational Mathematics Examination problems (algebra, combinatorics, number theory, geometry, probability). Three subsets: AIME 2024 (30 questions), AIME 2025 (30 questions), AIME 2020–2024 (133 questions).
- **GPQA-Diamond:** 198 expert-validated graduate-level questions across natural sciences (biology, chemistry, physics).
- **Game of 24:** 100 arithmetic puzzles requiring combining four digits using +, −, ×, ÷ to equal 24.
- **Math Equation Balancer:** 250 arithmetic expressions requiring insertion of correct operators.
- **MMLU-Pro (Engineering and Physics):** 250 questions sampled from each of engineering (969 total) and physics (1,299 total) subsets.

### 3.2 Models

- **Large models:** GPT-4o, Claude 3.5 Sonnet
- **Smaller models:** GPT-4o-mini, Claude 3.5 Haiku
- **Reasoning model:** DeepSeek R1

### 3.3 Baselines

- **BL (Baseline):** Standard one-off inference with minimal instructions.
- **DC-∅ (Empty Memory):** Uses the DC generator prompt with structured instructions for problem-solving and code generation, but keeps memory always empty. A strong baseline isolating the effect of memory curation.
- **FH (Full History):** Appends the entire conversation history without curation or truncation.
- **DR (Dynamic Retrieval):** Retrieves the most similar past interactions and pastes them verbatim into the prompt, but no curation.

### 3.4 Key Results

**Claude 3.5 Sonnet:**

| Task | BL | DC-∅ | DR | DC-Cu | DC-RS |
|------|-----|------|-----|-------|-------|
| AIME 2024 | 23.3% | 36.7% | 43.3% | **50.0%** | 46.7% |
| AIME 2025 | 6.7% | 23.3% | 23.3% | **36.7%** | 30.0% |
| AIME 2020–24 | 6.7% | 30.1% | 39.1% | 38.4% | **40.6%** |
| Game of 24 | 12.0% | 10.0% | 11.0% | 14.0% | 14.0% |
| GPQA-Diamond | 59.6% | 60.1% | 63.6% | 61.1% | **68.7%** |
| Math Eqn. Balancer | 44.8% | 56.4% | 60.4% | **100%** | 97.8% |
| MMLU Pro Eng. | 61.2% | 57.2% | 65.2% | 66.8% | **67.6%** |
| MMLU Pro Physics | 74.0% | 75.6% | 80.4% | 77.6% | **82.0%** |

**GPT-4o:**

| Task | BL | DC-∅ | DR | DC-Cu | DC-RS |
|------|-----|------|-----|-------|-------|
| AIME 2024 | 20.0% | 36.7% | 26.7% | 36.7% | **40.0%** |
| AIME 2025 | 6.7% | 10.0% | 10.0% | 16.7% | **20.0%** |
| Game of 24 | 10.0% | 19.0% | 6.0% | 93.0% | **99.0%** |
| GPQA-Diamond | 57.1% | 57.1% | 55.1% | **58.1%** | 57.1% |
| Math Eqn. Balancer | 50.0% | 88.0% | **100%** | **100%** | 99.2% |

**Full History vs. DC (AIME):**

| Task | Model | BL | FH | Best DC |
|------|-------|-----|-----|---------|
| AIME 2024 | Claude 3.5 Sonnet | 23.3% | 26.7% | 50.0% (DC-Cu) |
| AIME 2025 | Claude 3.5 Sonnet | 6.7% | 6.7% | 36.7% (DC-Cu) |
| AIME 2024 | GPT-4o | 20.0% | 13.3% | 40.0% (DC-RS) |
| AIME 2025 | GPT-4o | 6.7% | 3.3% | 20.0% (DC-RS) |

Full-history appending either matched or degraded baseline performance, while DC's selective curation produced substantial gains.

**Smaller Models (Claude 3.5 Haiku, GPT-4o-mini):**

Smaller models showed more limited and inconsistent gains. Claude 3.5 Haiku achieved moderate improvements (AIME 2024: 10.0% → 36.7% under DC-Cu; GPQA-Diamond: 43.4% → 49.0% under DC-RS). GPT-4o-mini showed even smaller gains, with some DC variants leading to slight performance declines.

**DC vs. Majority Voting (Claude 3.5 Sonnet):**

On AIME 2024, majority voting (3 samples) performed identically to baseline (23.3%), while DC-Cu reached 50.0%. On AIME 2025, majority voting stayed at 6.7%, while DC-Cu reached 36.7%.

### 3.5 Qualitative Observations

- **Tool usage emergence:** GPT-4o discovered early in Game of 24 that a Python brute-force solver was more reliable than manual arithmetic, stored it, and reused it for subsequent queries — leading to 10% → 99% accuracy.
- **Memory stability:** In Game of 24, both DC-Cu and DC-RS showed high memory stability (measured by LCS similarity between consecutive states) after the first few iterations, though DC-Cu experienced slightly greater fluctuations.
- **Error clustering:** Correct and incorrect answers often cluster in latent embedding space (shown via t-SNE on GPQA-Diamond). DC can transfer strategies within clusters, but erroneous heuristics can also spread without careful curation.

---

## 4. Contributions

1. **Introduces the Dynamic Cheatsheet framework**, a lightweight, parameter-free approach that endows black-box LLMs with persistent, evolving memory at inference time for test-time learning.

2. **Proposes two variants:** DC-Cu (cumulative memory without retrieval) and DC-RS (retrieval and synthesis, with memory updated before generation), providing flexibility for different task characteristics.

3. **Demonstrates substantial accuracy improvements** without ground-truth labels: Claude 3.5 Sonnet more than doubled its AIME accuracy; GPT-4o went from 10% to 99% on Game of 24; near-perfect accuracy achieved on Math Equation Balancer.

4. **Shows that curated memory outperforms naive approaches:** DC significantly outperforms full-history appending and majority voting, demonstrating that selective, evolving retention is superior to both brute-force context accumulation and statistical aggregation.

5. **Reveals emergent tool-usage behavior:** DC fosters models' inclination toward code generation for computationally intensive tasks, as models learn to recognize when external tools (e.g., Python) are more robust than internal chain-of-thought calculations.

6. **Provides gains on knowledge-intensive tasks:** Beyond arithmetic/algorithmic tasks, DC improves performance on GPQA-Diamond (+9.1%) and MMLU-Pro Engineering/Physics (+6.4%/+8.0% for Claude).

7. **Formalizes self-curated memory** that focuses on concise, transferable snippets rather than full transcripts, preventing context ballooning while facilitating meta-learning.

---

## 5. Limitations

1. **Model scale dependency:** Smaller models (GPT-4o-mini, Claude 3.5 Haiku) benefit from DC in limited and inconsistent amounts. These models generate too few correct solutions to populate memory with high-quality strategies, and they struggle to refine stored content. DC amplifies strengths of capable models but cannot fix foundational reasoning gaps.

2. **Memory pollution risk:** Without ground-truth labels, the Curator must self-assess solution quality. Faulty heuristics that enter memory can be amplified and propagated to subsequent queries, especially in clusters of related problems. Ensuring "clean" memory requires careful curation and pruning.

3. **Long-context generation challenges:** Models sometimes merely reference or abbreviate existing memory (e.g., "Previous content [...] preserved") instead of explicitly rewriting it during curation. Such truncated memory updates reduce the quality of stored heuristics over time — this is the precursor to what the ACE paper later formalizes as "context collapse."

4. **Retrieval noise:** Poorly filtered retrieval can introduce confusion, particularly with diverse or loosely related queries. GPT-4o's performance occasionally dipped on GPQA-Diamond due to suboptimal retrieval choices.

5. **Task structure dependency:** DC thrives when test examples share structural similarities, enabling strategy transfer. Performance gains are attenuated for highly diverse tasks without recurring patterns.

6. **Sequential processing bottleneck:** DC's sequential structure (process one query, update memory, process next) poses challenges for large-scale parallel or batch tasks requiring independent inference.

7. **Reasoning models show minimal gains:** Experiments with DeepSeek R1 and o1 showed minimal or inconsistent improvements. These models produce overly verbose solutions that are difficult to distill into compact, reusable memory entries.

8. **No offline adaptation:** DC operates only in the online setting (test-time), without a separate offline training phase to pre-populate memory from a training set — a limitation later addressed by the ACE framework.

9. **Memory transferability limitations:** While larger models can produce higher-quality strategies that could theoretically benefit smaller models via memory transfer, if the smaller model lacks generative capacity to interpret those strategies correctly, performance can stall or degrade.