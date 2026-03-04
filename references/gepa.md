# GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning

**Paper:** Accepted at ICLR 2026 (Oral)  

---

## 1. Problem Formulation

The paper addresses the challenge of **sample-efficient optimization of compound AI systems** — modular systems composed of one or more LLM invocations, potentially interleaved with tool calls and arbitrary control flow. Such systems include agents, multi-agent systems, and general-purpose scaffolding techniques like ReAct and Archon.

### Formal Setup

A compound AI system is defined as Φ = (M, C, X, Y), where M = ⟨M₁, ..., M|M|⟩ denotes language modules, C specifies control flow logic, and X, Y are global input/output schemas. Each module Mᵢ = (πᵢ, θᵢ, Xᵢ, Yᵢ) consists of a prompt πᵢ, model weights θᵢ, and input/output schemas.

The optimization objective is:

> ⟨Π*, Θ*⟩ = argmax E[(x,m)~T] μ(Φ(x; ⟨Π, Θ⟩), m)

subject to a rollout budget constraint B (i.e., total number of system invocations plus evaluations ≤ B).

### Core Challenge

Reinforcement learning methods like GRPO treat success metrics as end-of-rollout scalar rewards and use policy gradients, typically requiring tens of thousands of rollouts (often 24,000+). This is prohibitively expensive when rollouts involve costly tool calls, limited inference budgets, or closed-source models whose weights cannot be finetuned. The paper argues that the **interpretable nature of language** provides a much richer learning medium than sparse scalar rewards, and that algorithms that learn deliberately in natural language by reflecting on execution traces can be far more sample-efficient.

---

## 2. Method: GEPA (Genetic-Pareto)

GEPA is a **reflective prompt optimizer** for compound AI systems that merges textual reflection with multi-objective evolutionary search. It optimizes only prompts ΠΦ while keeping model weights ΘΦ frozen. The method is built on three core principles: genetic prompt evolution, reflection using natural language feedback, and Pareto-based candidate selection.

### 2.0 Data Splits and Rollout Budget

GEPA receives the following inputs: a system Φ instantiated with simple seed prompts, a training dataset D_train, the standard evaluation metric μ, a feedback function μ_f, and the total rollout budget B. D_train is further split into two disjoint subsets:

- **D_feedback**: Used to sample minibatches for generating learning signals during reflective mutation. When GEPA executes a candidate on a minibatch of size b (fixed at b=3 in all experiments), these examples come from D_feedback. The feedback function μ_f is applied to these examples to obtain both numeric scores and textual diagnostic feedback. This is the subset that produces the actual learning signal for prompt updates.
- **D_pareto** (of size n_pareto): Used as the validation set for candidate selection and tracking. Every time a new candidate passes the minibatch improvement check, it is evaluated on the **entire** D_pareto set, and scores are stored in a scores matrix S where S[Φ_k][i] records the score of candidate Φ_k on instance i. This matrix drives the Pareto-based candidate selection (Section 2.3). After the budget is exhausted, GEPA returns the candidate with the best average score on D_pareto.
- **D_test**: A held-out test set never accessed during optimization, used solely for final evaluation.

In practice, the paper uses 150 training examples (D_feedback) and 300 validation examples (D_pareto) for most benchmarks (e.g., HotpotQA, HoVer). AIME-2025 uses 45 train / 45 val from prior years' problems; PUPA uses 111 train / 111 val.

**Rollout budget accounting:** A "rollout" is defined as one invocation of Φ plus evaluation by μ (or μ_f). GEPA's total budget B includes both the minibatch rollouts (b rollouts per iteration on D_feedback) and the full D_pareto evaluations (n_pareto rollouts each time a candidate passes the minibatch gate). To ensure fair comparison with MIPROv2, the authors first record MIPROv2's total rollout expenditure per benchmark, then cap GEPA to match this budget (within 10.15% discrepancy).

### 2.1 Genetic Optimization Loop (Algorithm 1)

GEPA begins with a candidate pool P = [Φ] containing only the base system and a parents array A = [None]. It first evaluates the base system on every instance in D_pareto to initialize the scores matrix S. Then it enters the main loop (while budget B is not exhausted):

1. **SelectCandidate** (line 7): Use Pareto-based selection (Section 2.3) to choose a candidate Φ_k from the pool P based on the scores matrix S.
2. **SelectModule** (line 8): Choose which module j within Φ_k to update. The paper uses a **round-robin** policy across the |M| modules in the system.
3. **Sample minibatch** (line 9): Draw a minibatch M of size b=3 from D_feedback.
4. **Gather feedback** (line 10): Execute Φ_k on each example in M, tracing the full execution (all module inputs, outputs, and reasoning chains). Call the feedback function μ_f to obtain both a numeric score and textual feedback (e.g., which documents were correctly retrieved vs. missed, which constraints were satisfied vs. failed, compiler error messages, etc.). When available, feedback can be module-specific — for example, in multi-hop systems, the evaluator provides feedback after each hop identifying which gold documents remain unretrieved.
5. **UpdatePrompt** (line 11): Pass the current prompt π_j, all feedback texts, and the execution traces for module j to a **reflection LM**, which proposes a new prompt π'_j (see Section 2.2 for details).
6. **Create new candidate** (line 12): Copy Φ_k and replace module j's prompt with π'_j to form Φ'.
7. **Minibatch gate** (lines 13–14): Compute the average scores σ (before) and σ' (after) on the minibatch M. If σ' > σ (i.e., the updated system improved on the minibatch):
   - Add Φ' to the pool P with ancestry record pointing to k.
   - Evaluate Φ' on **every instance in D_pareto** (lines 16–18), storing the per-instance scores in the scores matrix S.
8. If σ' ≤ σ, discard Φ' entirely (no D_pareto evaluation is performed, saving rollouts).

After the budget is exhausted, GEPA returns the candidate Φ* that maximizes the average score across all instances in D_pareto.

### 2.2 Reflective Prompt Mutation

This is the primary mechanism for generating new candidates. The key insight is that LLM execution traces are serialized natural language (instructions, reasoning chains, tool calls, tool outputs), and modern LLMs can leverage these traces via reflection to perform **implicit credit assignment** — attributing responsibility for the final outcome to the relevant modules and specific prompt elements.

**Step-by-step process:** Given candidate Φ_k selected from the Pareto frontier and module j selected by round-robin:

1. **Execute and trace**: Run Φ_k on each of the b=3 minibatch examples. Because Φ is a compound system with control flow C orchestrating modules M₁...M|M|, the execution trace captures every module's input, output, and reasoning at each step.
2. **Obtain feedback**: Call μ_f on each example. Unlike a standard metric μ that returns only a scalar (e.g., exact match = 0 or 1), μ_f additionally returns `feedback_text` — rich textual information from the evaluation process. Examples:
   - **HotpotQA/HoVer**: The feedback function identifies which gold documents were correctly retrieved at each hop and which remain unretrieved, providing per-hop, per-module feedback.
   - **IFBench**: The feedback function lists which output constraints (e.g., "mention a word at least three times") were satisfied and which were failed.
   - **PUPA**: The feedback provides a breakdown of the aggregate score into response quality and PII leakage sub-scores.
   - **Code tasks (NPUEval, KernelBench)**: Compiler error messages, profiling results, and execution failures are returned as feedback text.
3. **Reflect and propose**: The reflection LM is shown a meta-prompt (see below) containing: (a) the current prompt π_j for module j, (b) the minibatch examples with their inputs, module j's outputs, and the corresponding feedback texts. The reflection LM must analyze the failures, identify patterns, and write a new prompt π'_j that addresses the diagnosed issues.

**Evaluation traces as diagnostic signals:** This is a key contribution. Many evaluation metrics apply rich strategies before producing a scalar reward (e.g., code evaluation involves compilation, execution, and profiling; multi-hop QA evaluation checks document retrieval at each stage). Standard RL collapses all this into a single scalar (reward 0 or 1). GEPA instead extracts these intermediate textual traces and feeds them to the reflection LM, providing substantially more learning signal per rollout.

**The meta-prompt for reflection** (shown in Appendix C) has the following structure:

```
I provided an assistant with the following instructions to perform a task for me:
```<current instruction>```

The following are examples of different task inputs provided to the assistant 
along with the assistant's response for each of them, and some feedback on how 
the assistant's response could be better:
```<Inputs, Outputs and Feedback for minibatch of examples>```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task 
description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all 
niche and domain specific factual information about the task and include it in 
the instruction, as a lot of it may not be available to the assistant in the 
future. The assistant may have utilized a generalizable strategy to solve the 
task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks.
```

The meta-prompt explicitly instructs the reflection LM to: (a) infer the task description from the input format, (b) extract domain-specific factual knowledge from the examples and feedback and bake it into the instruction (since this information may not be available at future inference time), and (c) identify and codify generalizable strategies the assistant used successfully.

**Concrete example of prompt evolution (PUPA task, Figure 5):** The seed prompt for privacy-preserving delegation is simply: *"Given a private user query, create a privacy-preserving request for a powerful external LLM. The LLM may assist without learning private information about the user."* (score: 82.26). After one reflective mutation, the prompt expands to include detailed guidance on identifying and generalizing PII, query intent analysis, and reasoning explanations (score: 90.99). Subsequent iterations add structured output formatting, explicit bans on names/codes, transparent transformation rationale, and eventually a rigorous step-wise PII abstraction protocol (score: 97.6 at candidate 11). Each refinement layer adds targeted nuances informed by the specific failures observed during rollouts.

### 2.3 Pareto-Based Candidate Selection (Algorithm 2)

The candidate selection strategy governs the exploration–exploitation tradeoff in GEPA's evolutionary search. A naive greedy approach (always selecting the globally best-performing candidate to mutate) often traps the optimizer in a **local optimum**: once a dominant strategy is found, the search repeatedly attempts to refine it, fails to improve, and exhausts the budget. Figure 6a in the paper illustrates this — after finding one good child node, the SelectBestCandidate strategy generates 17 consecutive failed mutations from that single candidate.

GEPA instead employs a **Pareto-based "illumination" strategy** (inspired by MAP-Elites) that operates at the **instance level** rather than the aggregate level. The algorithm works as follows:

**Step 1 — Build instance-wise Pareto sets:** For each training instance i in D_pareto, identify the maximum score achieved across all candidates: s*[i] = max_k S[Φ_k][i]. Then collect the set of candidates that achieve this best score on instance i: P*[i] = {Φ_k : S[Φ_k][i] = s*[i]}.

**Step 2 — Collect all "winning" candidates:** Take the union C = ∪_i P*[i]. These are all candidates that are the best performer on at least one task instance.

**Step 3 — Prune dominated candidates:** Among C, remove any candidate Φ that is **strictly dominated** by another candidate in C (i.e., there exists another candidate that scores ≥ Φ on all instances and strictly better on at least one). Let D be the set of dominated candidates. Remove D from each P*[i] to get the filtered sets P̂*[i].

**Step 4 — Sample proportional to "wins":** For each non-dominated candidate Φ_k, compute f[Φ_k] = number of instances i for which Φ_k ∈ P̂*[i] (i.e., how many tasks this candidate leads on). Sample a candidate with probability proportional to f[Φ_k].

**Worked example:** Suppose we have 5 task instances and 4 candidates (P₀ through P₃) with scores:

| | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|--------|--------|--------|--------|--------|
| P₀ (base) | 0.6 | 0.5 | 0.7 | 0.4 | 0.5 |
| P₁ | **0.9** | 0.3 | **0.8** | 0.5 | 0.6 |
| P₂ | 0.5 | **0.8** | 0.6 | **0.7** | 0.4 |
| P₃ | 0.7 | 0.6 | 0.7 | 0.5 | **0.8** |

- Step 1: Best per task — Task 1: 0.9 (P₁), Task 2: 0.8 (P₂), Task 3: 0.8 (P₁), Task 4: 0.7 (P₂), Task 5: 0.8 (P₃).
- Step 2: C = {P₁, P₂, P₃}. P₀ is not the best on any task, so it is excluded.
- Step 3: Check domination. P₁ is best on Tasks 1,3; P₂ is best on Tasks 2,4; P₃ is best on Task 5. None strictly dominates another (each has unique strengths), so D = ∅.
- Step 4: f[P₁] = 2 (leads Tasks 1, 3), f[P₂] = 2 (leads Tasks 2, 4), f[P₃] = 1 (leads Task 5). Sampling probabilities: P₁ = 2/5, P₂ = 2/5, P₃ = 1/5.

This means GEPA would select P₁ or P₂ with 40% probability each, and P₃ with 20% probability, even though P₁ might have a higher aggregate score. A candidate that solves even a single unique task instance gets non-zero probability, preserving diverse "winning strategies" in the population.

**Why this matters:** In the paper's ablation (Table 3), SelectBestCandidate achieves +6.05% aggregate improvement while GEPA's Pareto selection achieves +12.44% — more than double. The search trees (Figure 6) show the difference visually: greedy search produces a deep, narrow tree (17 failed refinements of one candidate), while Pareto search produces a broad, balanced tree with multiple independent lineages, discovering a candidate scoring 96.3 vs. the greedy strategy's best of 92.87 (on PUPA).

### 2.4 D_pareto Evaluation: Efficiency Discussion

**How D_pareto evaluation works:** Every time a new candidate Φ' passes the minibatch gate (i.e., improves over its parent on the minibatch of b=3 examples), GEPA evaluates Φ' on the **entire** D_pareto validation set. For most benchmarks, |D_pareto| = 300 instances. This means each successful mutation triggers 300 additional rollouts. The resulting per-instance scores are stored in the scores matrix S, which drives subsequent Pareto-based candidate selection.

**Why it is expensive:** The majority of GEPA's total rollout budget is spent on D_pareto evaluations, not on the minibatch rollouts that produce learning signals. Concretely, the paper reports that if we restrict analysis to **train-set (D_feedback) rollouts only**, GEPA requires just 79–737 rollouts to reach optimal performance. To match GRPO's best validation score, GEPA needs only 102, 32, 6, and 179 train rollouts for HotpotQA, IFBench, HoVer, and PUPA respectively. But the total budget including D_pareto evaluations ranges from 1,839 to 7,051 rollouts — meaning D_pareto evaluations account for roughly 80–97% of the total rollout budget.

**Why it is necessary:** The D_pareto evaluations serve two critical purposes: (a) they populate the scores matrix S that enables instance-level Pareto selection (without per-instance scores, GEPA cannot identify which candidates excel on which task instances), and (b) they provide the aggregate metric for final candidate selection. Without full D_pareto evaluation, the Pareto-based selection mechanism — which is responsible for a +6–7% improvement over greedy selection (Table 3) — simply cannot function.

**Is it efficient?** There is a clear tension. On one hand, D_pareto evaluation is what enables GEPA's strongest contribution (Pareto-based selection). On the other hand, it is the dominant cost. The authors explicitly acknowledge this and propose two future directions for improvement: (1) evaluating on a smaller validation set, and (2) tracking scores on **dynamically selected validation subsets** rather than the full set. Neither is implemented in the current work. A practical concern is that reducing D_pareto size risks degrading the quality of Pareto selection — with fewer instances, the instance-level Pareto front becomes less discriminative, and candidates are more likely to appear non-dominated by chance. The optimal tradeoff between D_pareto size and Pareto selection quality remains an open question.

**Comparison with GRPO's budget:** GRPO uses 24,000 rollouts entirely for training (gradient updates), with validation performed every 20 steps. GEPA's 1,839–7,051 rollouts include both training and validation, making the comparison slightly asymmetric. However, GEPA still uses 3.4–13× fewer total rollouts while outperforming GRPO on 5 of 6 tasks, and the train-only rollout counts (79–737) demonstrate that the reflection mechanism itself is extraordinarily sample-efficient.

### 2.5 System-Aware Merge (Crossover)

GEPA+Merge introduces a crossover strategy specifically designed for compound systems with multiple modules (Algorithm 4). The key idea is to combine complementary optimization lineages — candidates that have independently improved different modules of the system.

**Selection criteria (Algorithm 3 — Desirable check):** Two candidates P_i and P_j are eligible for merge only if all of the following hold:
- They share a common ancestor a (but neither is a direct ancestor of the other).
- They have optimized **disjoint sets of modules** relative to ancestor a: for at least one module m, P_i has evolved m (π_i ≠ π_a) while P_j has not (π_j = π_a), or vice versa. This ensures the candidates contribute complementary improvements.
- Both candidates individually outperform the ancestor a in aggregate score.
- This specific merge (i, j, a) has not been tried before.

**Merge procedure:** For each module m in the system:
- If P_i evolved m but P_j did not (π_a = π_j ≠ π_i): use P_j's (evolved) prompt → actually, take P_i's evolved version.
- If P_j evolved m but P_i did not (π_a = π_i ≠ π_j): use P_j's evolved version.
- If both evolved m differently (π_i ≠ π_j ≠ π_a): choose the prompt from whichever candidate has the higher aggregate score.
- If neither evolved m: use either (defaults to P_i).

Merge is invoked a maximum of 5 times per optimization run. Results show GEPA+Merge provides up to 5% additional improvement on GPT-4.1 Mini but can degrade performance on Qwen3 8B due to fixed hyperparameters, suggesting adaptive invocation timing is needed.

### 2.6 Meta-Prompt for Reflection

The reflection LM receives a structured meta-prompt containing:

```
I provided an assistant with the following instructions to perform a task for me:
```<current instruction>```

The following are examples of different task inputs provided to the assistant 
along with the assistant's response for each of them, and some feedback on how 
the assistant's response could be better:
```<Inputs, Outputs and Feedback for minibatch of examples>```

Your task is to write a new instruction for the assistant.
- Read the inputs carefully and identify the input format and infer detailed 
  task description.
- Identify all niche and domain specific factual information and include it 
  in the instruction, as it may not be available to the assistant in the future.
- If the assistant utilized a generalizable strategy, include it as well.

Provide the new instructions within ``` blocks.
```

The meta-prompt explicitly instructs the reflection LM to: (a) infer the task description from the input format, (b) extract domain-specific factual knowledge from the examples and feedback and bake it into the instruction (since this information may not be available at future inference time), and (c) identify and codify generalizable strategies. Across all experiments, GEPA makes between 17 and 92 reflection LM calls per benchmark (Table 4).

---

## 3. Experiments

### 3.1 Benchmarks (6 tasks)

| Benchmark | Task Type | System Architecture |
|-----------|-----------|-------------------|
| **HotpotQA** | Multi-hop reasoning/QA | Multi-hop retrieval with query writers and summarizers |
| **IFBench** | Instruction following | 2-stage system (answer + rewrite with constraints) |
| **AIME-2025** | Math competition | Single-step Chain-of-Thought |
| **LiveBench-Math** | Math reasoning | Single-step Chain-of-Thought |
| **HoVer** | Retrieval-augmented verification | 3-hop retrieval with 2 query writers and 2 summarizers |
| **PUPA** | Privacy-aware delegation | 2-module system (query rewriter + response rewriter) |

### 3.2 Models

- **Open-source:** Qwen3 8B (temperature 0.6, top-p 0.95, top-k 20)
- **Proprietary:** GPT-4.1 Mini (temperature 1.0)

### 3.3 Baselines

- **GRPO** (24,000 rollouts with LoRA, also full-parameter finetuning)
- **MIPROv2** (state-of-the-art instruction + few-shot optimizer via Bayesian optimization)
- **TextGrad** (textual feedback backpropagation)
- **Trace (OptoPrime)** (generative optimization with execution traces)
- **Unoptimized baseline**

### 3.4 Main Results

#### Qwen3 8B Results

| Method | HotpotQA | IFBench | HoVer | PUPA | AIME-2025 | LiveBench-Math | Aggregate | Improvement |
|--------|----------|---------|-------|------|-----------|---------------|-----------|-------------|
| Baseline | 42.33 | 36.90 | 35.33 | 80.82 | 27.33 | 48.70 | 45.23 | — |
| GRPO (24k rollouts) | 43.33 | 35.88 | 38.67 | 86.66 | 38.00 | 51.26 | 48.91 | +3.68 |
| MIPROv2 | 55.33 | 36.22 | 47.33 | 81.55 | 20.00 | 46.60 | 47.84 | +2.61 |
| **GEPA** | **62.33** | **38.61** | **52.33** | **91.85** | 32.00 | **51.95** | **54.85** | **+9.62** |
| GEPA+Merge | 64.33 | 28.23 | 51.67 | 86.26 | 32.00 | 51.95 | 52.40 | +7.17 |

GEPA used 1,839–7,051 rollouts per benchmark vs. GRPO's fixed 24,000.

#### GPT-4.1 Mini Results

| Method | HotpotQA | IFBench | HoVer | PUPA | AIME-2025 | LiveBench-Math | Aggregate | Improvement |
|--------|----------|---------|-------|------|-----------|---------------|-----------|-------------|
| Baseline | 38.00 | 47.79 | 46.33 | 78.57 | 49.33 | 58.20 | 53.03 | — |
| MIPROv2 | 58.00 | 49.15 | 48.33 | 83.37 | 51.33 | 61.84 | 58.67 | +5.64 |
| TextGrad | 62.33 | 48.64 | 47.67 | 85.68 | 46.67 | 63.84 | 59.14 | +6.11 |
| **GEPA** | **69.00** | 52.72 | 51.67 | 94.47 | **59.33** | 64.13 | **65.22** | **+12.19** |
| **GEPA+Merge** | 65.67 | **55.95** | **56.67** | **96.46** | **59.33** | **64.13** | **66.36** | **+13.33** |

### 3.5 Key Experimental Observations

**Observation 1 — Sample Efficiency:** GEPA outperforms GRPO (24k rollouts) by up to 20% while using up to 35× fewer rollouts. It matches GRPO's best validation score after only 243–1,179 rollouts (up to 78× greater sample efficiency). Restricting to train-set rollouts only, GEPA requires just 6–737 rollouts to reach optimal performance.

**Observation 2 — Instruction-only vs. Few-shot:** GEPA (instruction-only optimization) consistently outperforms MIPROv2 (joint instruction + few-shot optimization) across all settings, with margins up to 11.1% for GPT-4.1 Mini and 10.3% for Qwen3 8B. GEPA more than doubles MIPROv2's aggregate gains (+13.33% vs. +5.64%).

**Observation 3 — Pareto Selection Matters:** Comparing selection strategies on Qwen3 8B, Pareto-based sampling yields +12.44% aggregate improvement vs. +6.05% for SelectBestCandidate (greedy) and +5.11% for BeamSearch(N=4). The greedy strategy quickly stalls in local optima after finding one good candidate.

**Observation 4 — Shorter and Cheaper Prompts:** GEPA-optimized prompts are up to 9.2× shorter than MIPROv2's prompts (which include few-shot examples), reducing inference cost and latency while achieving higher performance.

**Observation 5 — System-Aware Merge:** GEPA+Merge can provide up to 5% additional improvement by combining complementary optimization lineages, though optimal budget allocation between mutation and crossover remains an open question.

**Observation 6 — Cross-Model Generalization:** Prompts optimized on Qwen3-8B and evaluated on GPT-4.1-Mini ("GEPA-Qwen-Opt") achieve +9.00% aggregate improvement, outperforming MIPROv2, TextGrad, and Trace which optimized directly on GPT-4.1-Mini.

### 3.6 Extended Applications

**Inference-Time Search for Code Optimization:**
- NPUEval (AMD NPU kernels): GEPA with GPT-4o achieves 30.52% mean vector utilization vs. 4.25% for baseline sequential refinement and 19.03% for RAG+MIPROv2.
- KernelBench (CUDA kernels): GEPA boosts GPT-4o's near-0% fast₁ score to above 20% with increasing budget.

**Adversarial Prompt Search:** By inverting the reward signal, GEPA finds universal adversarial instructions that reduce GPT-5 Mini's pass@1 on AIME-2025 from 76% to 10%, inserting task-preserving distractors that cause systematic formatting misinterpretation.

---

## 4. Contributions

1. **GEPA Algorithm:** A novel sample-efficient prompt optimizer that combines natural language reflection with evolutionary search and Pareto-based candidate selection for compound AI systems.

2. **Reflective Prompt Mutation:** A mechanism that leverages both execution traces and evaluation traces (diagnostic signals from reward computation) to perform implicit credit assignment at the module level, enabling large and targeted updates from just a few rollouts.

3. **Pareto-Based Illumination Strategy:** An instance-level Pareto frontier approach for candidate selection that balances exploration and exploitation, significantly outperforming greedy and beam-search strategies.

4. **System-Aware Merge:** A crossover strategy for compound systems that identifies and combines complementary optimization lineages across different modules.

5. **Empirical Demonstration of Prompt Optimization > RL:** Across six benchmarks and two model families, GEPA outperforms GRPO by an average of +6% with up to 35× fewer rollouts, and outperforms the leading prompt optimizer MIPROv2 by over +10% in aggregate.

6. **Cross-Model Transfer:** GEPA-optimized prompts generalize across model families, with prompts optimized on Qwen3-8B outperforming baselines optimized directly on GPT-4.1-Mini.

7. **Extended Applications:** Demonstrating GEPA as an inference-time search strategy for code optimization (NPUEval, KernelBench) and as an adversarial prompt search tool.

8. **Cost Efficiency:** All GPT-4.1-Mini experiments in Table 2 cost under $500 total, with GEPA specifically costing $86.

---

## 5. Limitations

1. **Validation Budget Overhead:** The majority of GEPA's rollout budget is spent on validation (evaluating candidates on D_pareto for selection), not on producing learning signals. While the authors propose evaluating on smaller or dynamically selected validation subsets, this remains future work.

2. **Merge Hyperparameter Sensitivity:** The optimal budget allocation between reflective mutation and crossover, as well as when to invoke merge, is not well understood. Fixed hyperparameters led to performance degradation with Qwen3 8B on some tasks while working well for GPT-4.1 Mini. Adaptive techniques are proposed as future work.

3. **Prompt-Only Optimization:** GEPA only optimizes prompts (ΠΦ) while keeping model weights (ΘΦ) frozen. It cannot modify the underlying model, which may limit gains in settings where weight updates are necessary. However, this is also a feature enabling use with closed-source models.

4. **Dependence on Reflection LM Quality:** GEPA requires a capable reflection LM to diagnose problems and propose meaningful prompt updates. The quality of the reflective mutations is bounded by the reflective and instruction-following abilities of the LLM used for this purpose.

5. **Limited Scalability Analysis:** While GEPA is tested on systems with 1–4 modules, scalability to much larger compound systems with many more modules is not thoroughly explored. The round-robin module selection policy may not be optimal for highly complex systems.

6. **Task-Specific Feedback Functions:** GEPA benefits significantly from rich feedback functions (μ_f) that provide module-specific textual feedback. In domains where such feedback is not naturally available, the effectiveness of reflective mutation may be reduced.

7. **AIME Performance with Qwen3 8B:** GEPA underperforms GRPO on the AIME-2025 math benchmark with Qwen3 8B (32.00% vs. 38.00%), suggesting that for certain tasks, weight-space optimization may still be more effective, particularly for mathematical reasoning with smaller models.

8. **Preliminary Nature of Extended Applications:** The inference-time search and adversarial prompt search experiments are described as "early results" that "warrant further systematic study" — they are not as rigorously evaluated as the main benchmarks.

9. **Generalization Gap Variability:** While GEPA generally shows good generalization from validation to test, some configurations show non-trivial generalization gaps (both positive and negative), and the stability of these gaps across different random seeds is not analyzed.

10. **Computational Cost of Reflection:** While GEPA uses fewer rollouts overall, each reflective mutation involves calling a (potentially expensive) reflection LM. The total number of reflection calls ranges from 17–92 across benchmarks, and the cost of these calls relative to rollout cost is not deeply analyzed.