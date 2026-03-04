# PRISM: Persistent Reflective Instruction Skill Memory

## 1. Problem Formulation

### 1.1 The Challenge

We address the problem of **automated knowledge accumulation for LLM systems** — enabling an LLM to progressively build, refine, and curate a persistent library of reusable problem-solving strategies ("skills") through self-reflection on its own execution traces, without modifying model weights.

### 1.2 Formal Setup

Consider an LLM system with frozen weights θ that processes a sequence of tasks T = (t₁, t₂, ..., tₙ) drawn from a task distribution D. Each task tᵢ = (xᵢ, yᵢ) consists of an input xᵢ and a ground-truth evaluation target yᵢ. The system augments its base prompt π₀ with a dynamically selected subset of skills from a persistent skill library L.

A **skill** s ∈ L is a structured knowledge unit s = (name, description, content, metadata) where:
- `content` is a rich markdown document encoding reusable strategies, workflows, examples, and domain knowledge
- `description` is a trigger condition specifying when to apply this skill
- `metadata` tracks performance history: attribution counts, per-instance score vectors, quality assessments, and lineage

The skill library L = {s₁, s₂, ..., s_m} evolves over time through six lifecycle operations: BIRTH (create), ENRICH (append), REFINE (rewrite), SPECIALIZE (split), GENERALIZE (merge), and RETIRE (deactivate).

### 1.3 Optimization Objective

At each step i, the system:

1. **Retrieves** a subset S_i ⊆ L of skills relevant to task tᵢ via a multi-layer retrieval pipeline R(tᵢ, L)
2. **Executes** the task with the augmented prompt: ŷᵢ = M(π₀ ⊕ S_i, xᵢ; θ)
3. **Evaluates** performance: score_i = μ(ŷᵢ, yᵢ)
4. **Reflects** on the execution trace to produce attributions and gap diagnoses
5. **Curates** the library L through lifecycle operations informed by reflection

The objective is to maximize cumulative task performance over the sequence:

> max_L  E[t~D] μ(M(π₀ ⊕ R(t, L), x; θ), y)

subject to the constraint that model weights θ remain frozen and the only adaptable component is the skill library L.

### 1.4 Core Problem: Why Skills, Not Prompts?

Existing approaches to context adaptation optimize either **monolithic prompts** (rewritten wholesale at each mutation) or **flat memory lists** (bullets appended and pruned over time). These representations face fundamental limitations:

**Problem 1 — Knowledge interference.** A monolithic prompt or flat memory must encode all strategies for all task types in a single text block. Improving instructions for geometry problems may degrade performance on combinatorics — all knowledge competes for the same space.

**Problem 2 — No compositional reuse.** Strategies discovered during optimization are locked inside prompt text or memory bullets. They cannot be extracted, recombined, or reused across different task contexts. Each system must rediscover knowledge from scratch.

**Problem 3 — Catastrophic forgetting under rewrite.** When an LLM rewrites accumulated context, it tends to compress or drop previously accumulated knowledge — a phenomenon known as "context collapse." Append-only strategies avoid this but lead to bloated, unfocused context over time.

**PRISM's answer:** Decompose knowledge into **independent, persistent skill units** that:
- Are created, evaluated, and retired independently (no interference)
- Can be selected in arbitrary combinations per task (compositional reuse)
- Accumulate content via ENRICH (append-only) and are rewritten only under quality-controlled REFINE (no collapse)
- Track per-instance performance via a score matrix, enabling Pareto-based lifecycle decisions

### 1.5 The Score Matrix and Pareto-Based Lifecycle

A key technical contribution is the **per-instance score matrix** for lifecycle decisions. Each skill s maintains a score vector:

> score_matrix[s] = { h(tᵢ) → δᵢ }

where h(tᵢ) is a hash of task instance i and δᵢ is the differential score (performance with skill − performance without). This enables:

**Pareto-based RETIRE:** Skill A is retired if another skill B ε-dominates it — i.e., B ≥ A − ε on all shared instances and B > A on at least one. This replaces naive "harmful − helpful > threshold" with a rigorous notion of redundancy.

**Pareto-frequency SPECIALIZE:** A skill with high pareto_frequency (often on the Pareto front) but also high harmful_ratio is a candidate for SPECIALIZE — it contains valuable knowledge for some tasks but hurts others. Splitting into focused children preserves the good while isolating the bad.

**Coverage gap BIRTH:** Instances where all skills score below a threshold indicate uncovered areas of the task distribution, triggering targeted BIRTH of new skills.

**Pareto-aware retrieval:** The assembler boosts skills with high pareto_frequency and maintains exploration slots for untested skills, balancing exploitation of proven skills with exploration of new ones.

### 1.6 Content Quality as a First-Class Signal

A fundamental limitation of existing context adaptation methods is that quality is measured only at the **outcome level** — did the task succeed or fail? This conflates many factors: was the skill content actually good? Was it retrieved appropriately? Did the model follow it? Was the task just hard?

PRISM introduces a **content-level quality gate** via the SkillValidator:

- **Generation-time validation:** Every BIRTH/ENRICH/SPECIALIZE output is scored by an LLM on 5 dimensions (structural completeness, actionability, description alignment, internal consistency, differentiation). Skills below threshold are revised or discarded before entering the library.
- **Periodic audit:** Active skills are periodically re-assessed and REFINED (full content rewrite) when quality has degraded or content has become bloated from repeated enrichment.

This creates a dual feedback loop: task outcomes drive lifecycle decisions (retire, specialize), while content quality drives content improvement (validate, refine). The two signals are complementary and independently actionable.

### 1.7 The PRISM Loop

Putting it all together, PRISM implements a 6-step loop per task:

```
For each task tᵢ:
  ① RETRIEVE    Multi-layer skill retrieval with Pareto-aware selection
                 → select top-k skills balancing exploit (proven) and explore (untested)

  ② EXECUTE     Run augmented prompt through evaluation function
                 → score, execution trace, output

  ③ REFLECT     LLM analyzes trace → per-skill attributions (helpful/harmful/neutral)
                 → knowledge gap diagnoses

  ④ CURATE      Lifecycle operations informed by reflection + Pareto signals:
                 RETIRE   — ε-dominated or high harmful ratio
                 BIRTH    — from reflection gaps + coverage gaps
                 ENRICH   — append to existing skill (with quality gate)
                 SPECIALIZE — split Pareto-optimal but inconsistent skills
                 GENERALIZE — merge keyword-overlapping skills
                 REFINE   — periodic quality-driven content rewrite

  ⑤ DIFFERENTIAL Compare with-skills vs without-skills score
                 → populate per-instance score_matrix with delta

  ⑥ INDEX       Update EMA scores for future retrieval ranking
```

The library L persists across tasks and epochs, with each skill independently accumulating performance data, undergoing quality assessment, and evolving through lifecycle operations. Over time, the library converges toward a set of high-quality, non-redundant, specialized skills that collectively cover the task distribution.

### 1.8 Hypotheses

**H1 — Skill decomposition outperforms monolithic prompts.** By decomposing knowledge into independent, composable skill units, PRISM avoids knowledge interference and enables task-specific skill combinations that a single prompt cannot express.

**H2 — Per-instance Pareto signals produce better lifecycle decisions.** The score matrix enables domination-based retirement and frequency-based specialization triggers that are more principled than aggregate heuristics (harmful−helpful > 3, variance > 0.20).

**H3 — Content-level quality control improves the library over time.** By validating generated content at creation time and periodically refining existing content, the library maintains higher average skill quality than systems relying solely on outcome-based pruning.

**H4 — The dual feedback loop (outcome + content quality) converges faster.** Outcome signals tell the system *which* skills to keep; quality signals tell the system *how* to improve them. The combination reduces the number of tasks needed to build an effective library.

---

## 2. Method

### 2.1 Overview

PRISM runs a six-step loop per task instance: **Retrieve → Execute → Reflect → Curate → Differential Evaluate → Index**. The system maintains a persistent skill library $\mathcal{L}$ where each skill independently accumulates performance data, undergoes lifecycle operations, and evolves over time. The core mechanism is a Pareto-guided skill management system that operates across three levels: per-instance score tracking, skill-level Pareto frequency, and coverage gap analysis.

### 2.2 Skill Representation

Each skill $\sigma \in \mathcal{L}$ is a structured knowledge unit $\sigma = (\text{name}, \text{description}, \text{content}, \text{metadata})$ containing:

- **content**: a rich markdown document encoding reusable strategies, workflows, and domain knowledge (capped at 8,000 characters, approximately 2,000 tokens)
- **description**: natural language trigger condition specifying when to apply this skill
- **module\_tag**: which system module this skill belongs to (e.g., "general", "solver")
- **task\_types**, **keywords**: semantic categories and tags for lightweight retrieval
- **trigger\_conditions**, **strategy\_notes**, **pitfalls**: structured metadata for matching and content organization
- **embedding**: optional vector representation for cosine similarity retrieval
- **helpful\_count**, **harmful\_count**, **neutral\_count**: attribution counters updated by the Reflector
- **score\_matrix**: per-instance performance record (see §2.3)
- **pareto\_frequency**: fraction of evaluated instances where the skill is Pareto-optimal (see §2.4)
- **status**: "active" or "retired"
- **parent\_id**, **children\_ids**: lineage tracking for SPECIALIZE operations

Skills are persisted as markdown files (`SKILL.md`) with YAML frontmatter and stored in a directory-based library. A companion `_meta.json` file tracks all numerical metadata, score matrices, and lineage information.

### 2.3 The Skill Score Matrix

Each skill $\sigma$ maintains a sparse score matrix $S[\sigma]$: a mapping from task instance keys to differential scores.

$$S[\sigma]: h(t_i) \rightarrow \delta_i \in \mathbb{R}$$

where $h(t_i)$ is an 8-character MD5 hash of the task question text and $\delta_i$ is the **differential score**: the difference between the system's performance with skills and without skills on that instance.

$$\delta_i = \mu(\hat{y}_i^{\text{with}}, y_i) - \mu(\hat{y}_i^{\text{without}}, y_i)$$

The differential evaluation runs the same task twice — once with the augmented prompt (base prompt $\oplus$ selected skills) and once with the base prompt alone — producing a score that isolates the marginal contribution of the skill set. When differential evaluation is disabled, the raw score is recorded directly.

The matrix is inherently sparse: each skill is only evaluated on instances where the Assembler selected it. This sparsity is handled explicitly in all downstream computations (§2.5).

### 2.4 Pareto Frequency

After each curation step, the system computes per-skill **Pareto frequency** — the fraction of evaluated instances on which a skill is non-dominated.

For each task instance key $j$ present in any active skill's score matrix, the best observed score is:

$$s^*(j) = \max_{\sigma \in \mathcal{L}_{\text{active}}} S[\sigma][j]$$

A skill $\sigma$ is **non-dominated on instance $j$** if its score is within $\varepsilon$ of the best:

$$S[\sigma][j] \geq s^*(j) - \varepsilon$$

where $\varepsilon = 0.05$ is a soft tolerance threshold. The Pareto frequency of skill $\sigma$ is then:

$$f(\sigma) = \frac{|\{j \mid \sigma \text{ is non-dominated on } j\}|}{|\{j \mid j \in S[\sigma]\}|}$$

This scalar $f(\sigma)$ serves as the primary signal for three downstream decisions: skill selection (§2.6), skill retirement, and skill specialization (§2.7).

### 2.5 Soft $\varepsilon$-Domination

To make lifecycle decisions, we define a conservative domination relation. Skill $\sigma_a$ **$\varepsilon$-dominates** skill $\sigma_b$ if and only if:

1. They share at least $n_{\min} = 3$ evaluated instances: $|K_a \cap K_b| \geq n_{\min}$, where $K_\sigma = \text{keys}(S[\sigma])$
2. On every shared instance, $\sigma_a$ scores at least as well (within tolerance): $\forall j \in K_a \cap K_b: S[\sigma_a][j] \geq S[\sigma_b][j] - \varepsilon$
3. On at least one shared instance, $\sigma_a$ is strictly better: $\exists j \in K_a \cap K_b: S[\sigma_a][j] > S[\sigma_b][j]$

Condition 1 prevents premature retirement when two skills have insufficient comparison data. The $\varepsilon$ tolerance in condition 2 avoids retiring a skill that is only marginally worse on some instances — it must be consistently dominated, not just occasionally slightly behind.

A skill is **Pareto-optimal** if no other active skill $\varepsilon$-dominates it. The set of Pareto-optimal skills forms the **Pareto front** of the library.

### 2.6 Multi-Layer Skill Retrieval

At inference time, the Assembler selects a subset of skills for a new task instance $x$ through a four-layer scoring pipeline:

**Layer 1 — Module filter.** Only active skills matching the current module tag are considered.

**Layer 2 — Semantic similarity.** If an embedding function is available, each candidate skill is scored by cosine similarity between its embedding and the task embedding:

$$\text{sim}(\sigma, x) = \cos(\text{emb}(\sigma.\text{description}),\ \text{emb}(x))$$

**Layer 3 — EMA index lookup.** A task type index maintains exponential moving average (EMA) scores per skill per task type. For the classified task type, the index returns the top-scoring skills. The final base score is the maximum of Layer 2 and Layer 3:

$$\text{base}(\sigma) = \max(\text{sim}(\sigma, x),\ \text{ema}(\sigma, \text{type}(x)))$$

**Pareto boost.** After base scoring, skills receive a bonus proportional to their Pareto frequency:

$$\text{score}(\sigma) = \text{base}(\sigma) + 0.3 \cdot f(\sigma)$$

**Exploration bonus.** Unevaluated skills (total evaluations = 0) that would otherwise score 0.0 receive a fixed bonus of 0.6, ensuring newly created skills are tested.

**Pool-based selection.** The top-$k$ skill slots are split into **exploit slots** and **explore slots**:

- **Exploit slots** ($k - e$): filled from the highest-scoring skills, prioritizing Pareto-optimal and well-evaluated skills
- **Explore slots** ($e$): filled from skills not in the exploit pool, ensuring under-tested or novel skills receive evaluation

The number of explore slots adapts to library maturity. When the total library evaluations are below a maturity threshold ($n_{\text{mature}} = 50$), the system allocates more explore slots ($e = \min(2, \lfloor k/2 \rfloor)$) to accelerate score matrix population. Once the library matures, explore slots reduce to $e = 1$.

**Layer 4 — LLM re-ranking (optional).** When the number of candidates exceeds $k$, an LLM selects the most relevant skills from the candidate pool, informed by skill descriptions and Pareto frequencies.

**Token budget enforcement.** Selected skills are accumulated in order until a token budget (default 2,000 tokens, estimated at 4 characters per token) is exhausted. Skills exceeding the remaining budget are skipped.

### 2.7 Lifecycle Operations

The Curator orchestrates six lifecycle operations after each reflection step, guided by Pareto signals and reflector diagnoses.

#### RETIRE — Remove dominated or consistently harmful skills

A skill is retired under two conditions:

- **Primary (Pareto-based):** The skill is $\varepsilon$-dominated by another active skill with at least $n_{\min}$ shared instances. This means a strictly better alternative exists across all evaluated instances.
- **Fallback (ratio-based):** When insufficient score matrix data exists for Pareto comparison, the skill is retired if its harmful ratio exceeds 0.7 with at least 5 total evaluations. This handles the cold-start period before score matrices are populated.

Retired skills are marked with `status = "retired"` and excluded from future retrieval.

#### BIRTH — Create new skills from identified gaps

New skills are created when the Reflector identifies knowledge gaps — areas where the current skill library lacks relevant strategies. Two sources of gap signals are combined:

- **Reflector gaps:** The Reflector analyzes execution traces and diagnoses specific missing knowledge (e.g., "no strategy for modular arithmetic with large primes").
- **Coverage gaps:** The system scans all task instance keys across active skills and identifies instances where the best score falls below a coverage threshold ($\theta_{\text{cov}} = 0.3$). When at least $n_{\text{gap}} = 2$ uncovered instances are found, they are surfaced as coverage gap descriptions.

Both gap sources are merged and passed to an LLM that decides whether to CREATE a new skill or ENRICH an existing one.

#### ENRICH — Append new knowledge to existing skills

When the LLM determines that an identified gap is best addressed by augmenting an existing skill, new content is appended as a markdown section. A content budget (8,000 characters) prevents unbounded growth:

- If the enriched content stays within budget, it is appended directly.
- If it exceeds the budget, the oldest enrichment section (identified by `## Enrichment` headers) is removed before appending.
- If still over budget after removal, the content is hard-truncated.

#### SPECIALIZE — Split broadly relevant but inconsistent skills

A skill is a candidate for specialization when it is **both frequently Pareto-optimal and frequently harmful**:

$$f(\sigma) \geq \theta_{\text{pareto}} \quad \text{AND} \quad \frac{\text{harmful\_count}}{\text{total\_evals}} \geq \theta_{\text{harmful}}$$

with $\theta_{\text{pareto}} = 0.4$ and $\theta_{\text{harmful}} = 0.25$, requiring at least 5 total evaluations. This trigger identifies skills that contain valuable knowledge for some task types but hurt performance on others — the ideal candidates for splitting.

An LLM receives the parent skill's content and generates two specialized child skills with narrower trigger conditions. The parent is retired, and both children inherit the parent's `skill_id` as their `parent_id` for lineage tracking.

#### GENERALIZE — Merge overlapping skills

When two active skills share sufficient keyword overlap (Jaccard similarity $\geq 0.6$), they are candidates for merging into a single, more comprehensive skill. An LLM combines the content of both skills, and both originals are retired.

#### REFINE — Quality-driven content rewrite

Periodically, active skills undergo content-level quality assessment via the SkillValidator. Skills whose content has degraded (e.g., through repeated enrichment) or whose quality score falls below threshold are fully rewritten by an LLM while preserving the core strategies. This operation is triggered by the quality audit cycle rather than by task-level outcomes.

### 2.8 The Reflect-Attribute Loop

After each task execution, the Reflector — a separate LLM call — analyzes the execution trace and produces:

- **Attributions**: For each skill used in the current task, the Reflector assigns a tag: `helpful` (the skill contributed to a correct solution), `harmful` (the skill led the model astray), or `neutral` (no discernible effect).
- **Gaps**: Descriptions of missing knowledge that would have helped solve the task.
- **Diagnosis**: A free-text analysis of what went right or wrong.

Attribution counts are accumulated on each skill (`helpful_count`, `harmful_count`, `neutral_count`) and inform both Pareto-based lifecycle decisions and the harmful ratio calculations used in RETIRE and SPECIALIZE triggers.

### 2.9 Differential Evaluation

To isolate the marginal contribution of the skill library, each task is evaluated twice:

1. **With skills:** The augmented prompt $\pi_0 \oplus S_i$ is evaluated, producing score $\mu_{\text{with}}$.
2. **Without skills:** The base prompt $\pi_0$ alone is evaluated on the same task, producing score $\mu_{\text{without}}$.

The differential $\delta = \mu_{\text{with}} - \mu_{\text{without}}$ is stored in the score matrix for each selected skill. A positive $\delta$ indicates the skills improved performance; a negative $\delta$ indicates they hurt.

This differential signal is strictly more informative than the raw score: a skill that was present when the system scored 0.8 on an easy task ($\mu_{\text{without}} = 0.9$) is properly credited with $\delta = -0.1$ rather than being rewarded for the high absolute score.

### 2.10 EMA Index Update

After evaluation, an exponential moving average (EMA) index is updated for each skill on the current task type. This index provides a fast, non-Pareto retrieval signal for Layer 3 of the Assembler, smoothing noisy per-instance scores into a stable ranking per task category.

### 2.11 Handling Sparsity and Cold Start

The score matrix is very sparse, especially early in optimization. PRISM uses three mechanisms to handle this gracefully:

1. **Optimistic inclusion in Pareto front.** Skills with no score matrix entries are included in the Pareto front by default — they cannot be dominated because there is no evidence against them. This prevents premature retirement of untested skills.

2. **Fallback lifecycle heuristics.** When score matrices have insufficient data for $\varepsilon$-domination checks ($< n_{\min}$ shared instances), the system falls back to aggregate ratio-based heuristics (harmful ratio $> 0.7$) for retirement decisions. As score matrices populate, the Pareto-based mechanism smoothly takes over.

3. **Adaptive exploration.** The Assembler allocates more explore slots when the library is immature (total evaluations $< n_{\text{mature}}$), accelerating score matrix population. Unevaluated skills receive a fixed exploration bonus to ensure they are selected and tested. As the library matures, exploration decreases in favor of exploitation of proven skills.

### 2.12 Multi-Epoch Training

The system supports multi-epoch training over a fixed task set. In each epoch, every task instance is processed through the full six-step loop. Across epochs:

- The score matrix accumulates more entries, making Pareto computations increasingly accurate
- Lifecycle operations progressively retire dominated skills, specialize inconsistent ones, and birth new skills for coverage gaps
- The library converges toward a set of high-quality, non-redundant, specialized skills

Periodic maintenance (every $n_{\text{maint}}$ steps) persists the library and index to disk, and optional validation on a held-out set monitors generalization performance.
