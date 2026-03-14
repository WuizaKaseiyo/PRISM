# PRISM: Persistent Reflective Instruction Skill Memory

## 1. Problem Formulation

### 1.1 The Challenge

We address the problem of **automated knowledge accumulation for LLM systems** — enabling an LLM to progressively build, refine, and curate a persistent library of reusable problem-solving strategies ("skills") through self-reflection on its own execution traces, without modifying model weights.

### 1.2 Formal Setup

Consider an LLM system with frozen weights θ that processes a sequence of tasks T = (t₁, t₂, ..., tₙ) drawn from a task distribution D. Each task tᵢ = (xᵢ, yᵢ) consists of an input xᵢ and a ground-truth evaluation target yᵢ. The system augments its base prompt π₀ with a dynamically selected subset of skills from a persistent skill library L.

A **skill** s ∈ L is a structured knowledge unit s = (name, description, content, metadata) containing:

- **content**: a rich markdown document encoding reusable strategies, workflows, and domain knowledge (capped at 8,000 characters, approximately 2,000 tokens)
- **description**: natural language trigger condition specifying when to apply this skill
- **module\_tag**: which system module this skill belongs to (e.g., "general", "solver")
- **task\_types**, **keywords**: semantic categories and tags for lightweight retrieval
- **trigger\_conditions**, **strategy\_notes**, **pitfalls**: structured metadata for matching and content organization
- **embedding**: optional vector representation for cosine similarity retrieval
- **helpful\_count**, **harmful\_count**, **neutral\_count**: attribution counters updated by the Reflector
- **score\_matrix**: per-instance performance record (see §2.1)
- **pareto\_frequency**: fraction of evaluated instances where the skill is Pareto-optimal (see §2.2)
- **status**: "active" or "retired"
- **parent\_id**, **children\_ids**: lineage tracking for SPECIALIZE operations

Skills are persisted as markdown files (`SKILL.md`) with YAML frontmatter and stored in a directory-based library. A companion `_meta.json` file tracks all numerical metadata, score matrices, and lineage information.

The skill library L = {s₁, s₂, ..., s_m} evolves over time through seven lifecycle operations: NO-OP (skip), BIRTH (create), ENRICH (append), REFINE (rewrite), SPECIALIZE (split), GENERALIZE (merge), and RETIRE (deactivate).

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

### 1.4 Core Problems

#### 1.4.1 Why Skills, Not Prompts?

Existing approaches to context adaptation optimize either **monolithic prompts** (rewritten wholesale at each mutation) or **flat memory lists** (bullets appended and pruned over time). These representations face fundamental limitations:

**Problem 1 — Knowledge interference.** A monolithic prompt or flat memory must encode all strategies for all task types in a single text block. Improving instructions for geometry problems may degrade performance on combinatorics — all knowledge competes for the same space.

**Problem 2 — No compositional reuse.** Strategies discovered during optimization are locked inside prompt text or memory bullets. They cannot be extracted, recombined, or reused across different task contexts. Each system must rediscover knowledge from scratch.

**Problem 3 — Catastrophic forgetting under rewrite.** When an LLM rewrites accumulated context, it tends to compress or drop previously accumulated knowledge — a phenomenon known as "context collapse." Append-only strategies avoid this but lead to bloated, unfocused context over time.

The emerging consensus confirms this direction: recent surveys characterize skill engineering as a paradigm shift beyond prompt engineering and atomic tool use (Wu et al., 2026; Jiang et al., 2026), where skills are composable packages of instructions, workflows, and metadata that agents load on demand.

#### 1.4.2 Why PRISM, Not Existing Skill Systems?

A wave of concurrent work has begun exploring skill-based architectures for LLM agents. We identify several key approaches and the gaps PRISM addresses:

**SkillRL** (Li et al., 2026) distills raw trajectories into a hierarchical skill library (SkillBank) and co-evolves skills with the agent's policy via reinforcement learning. **SAGE** (Yang et al., 2025) similarly incorporates skill libraries into an RL training loop (GRPO), using sequential rollout across task chains. Both systems **require weight updates** — the skill library and the agent's policy are jointly trained. PRISM operates with **frozen model weights**, making it applicable to black-box API models and requiring no training infrastructure.

**AutoSkill** (Wang et al., 2026) is closest in spirit to PRISM: it extracts reusable skills from interaction traces without retraining, supports a skill lifecycle (creation, evolution, injection), and positions itself as a model-agnostic plug-in. However, AutoSkill lacks (1) **per-instance performance tracking** — it does not maintain score matrices or Pareto-based evaluation, relying instead on heuristic quality signals; (2) **principled lifecycle decisions** — skill evolution is driven by LLM judgment alone, without ε-domination or leave-one-out verification; and (3) **credit attribution** — when multiple skills are co-selected, there is no mechanism to disentangle individual contributions.

**MACLA** (Forouzandeh et al., 2025) maintains hierarchical procedural memory with Bayesian reliability tracking and contrastive refinement. While it shares PRISM's goal of frozen-weight adaptation with principled evaluation, it operates at the **procedure level** (step-by-step action sequences) rather than the **strategy level** (reusable problem-solving knowledge). MACLA's procedures are tightly coupled to specific action spaces (ALFWorld, WebShop), whereas PRISM's skills are domain-agnostic knowledge documents.

**Problem 4 — Unaddressed security in skill systems.** As skill-based systems become prevalent, they introduce a novel **supply chain security threat**: malicious instructions can be embedded within otherwise legitimate skill content. Unlike traditional prompt injection (adversarial text in data), skill injection is an *instruction-instruction conflict* — bad instructions hidden among good ones — making data-instruction separation defenses inapplicable. Furthermore, many instructions are **dual-use**: the same action (e.g., "backup files to external server") can be legitimate in one context but constitute data exfiltration in another. Recent surveys identify this as a critical open problem (Wu et al., 2026), yet none of the above systems incorporate security mechanisms into their skill lifecycle. PRISM's continuous curation provides **defense-in-depth**: harmful content is detected through performance degradation and removed via RETIRE, isolated via SPECIALIZE, or corrected via REFINE.

**PRISM's answer:** Decompose knowledge into **independent, persistent skill units** that:
- Are created, evaluated, and retired independently (no interference)
- Can be selected in arbitrary combinations per task (compositional reuse)
- Accumulate content via ENRICH (append-only) and are rewritten only under quality-controlled REFINE (no collapse)
- Track per-instance performance via a score matrix, enabling **Pareto-based lifecycle decisions** — unlike heuristic-driven evolution in AutoSkill or RL-coupled evolution in SkillRL/SAGE
- Solve the **set-vs-skill credit attribution problem** through attribution-refined deltas and targeted leave-one-out verification — a challenge unaddressed by any existing skill system
- Are **self-generated and continuously curated**, providing inherent robustness to skill injection

### 1.5 The Score Matrix and Pareto-Based Lifecycle

A key technical contribution is the **per-instance score matrix** for lifecycle decisions. Each skill s maintains a score vector:

> score_matrix[s] = { h(tᵢ) → (δᵢ, δ̂ᵢ, Cᵢ, soloᵢ) }

where h(tᵢ) is a hash of task instance i, δᵢ is the raw set-level differential score (performance with skill set − performance without), δ̂ᵢ is the attribution-refined per-skill delta, Cᵢ records co-selected skills, and soloᵢ indicates whether the skill was the only one selected. Pareto computations use δ̂ᵢ to mitigate the set-vs-skill credit attribution problem (§2.1). This enables:

**Pareto-based RETIRE:** Skill A is retired if another skill B ε-dominates it — i.e., B ≥ A − ε on all shared instances and B > A on at least one. This replaces naive "harmful − helpful > threshold" with a rigorous notion of redundancy.

**Pareto-frequency SPECIALIZE:** A skill with high pareto_frequency (often on the Pareto front) but also high harmful_ratio is a candidate for SPECIALIZE — it contains valuable knowledge for some tasks but hurts others. Splitting into focused children preserves the good while isolating the bad.

**Coverage gap BIRTH:** Instances where all skills score below a threshold indicate uncovered areas of the task distribution, triggering targeted BIRTH of new skills.

**Pareto-aware retrieval:** The assembler boosts skills with high pareto_frequency and maintains exploration slots for untested skills, balancing exploitation of proven skills with exploration of new ones.

### 1.6 Content Quality as a First-Class Signal

A fundamental limitation of existing context adaptation methods is that quality is measured only at the **outcome level** — did the task succeed or fail? This conflates many factors: was the skill content actually good? Was it retrieved appropriately? Did the model follow it? Was the task just hard?

PRISM introduces a **content-level quality gate** via the SkillValidator:

- **Generation-time validation:** Every BIRTH/ENRICH/SPECIALIZE output is scored by an LLM on 5 dimensions (structural completeness, actionability, description alignment, internal consistency, differentiation). Skills below threshold are revised or discarded before entering the library.
- **Periodic audit:** Active skills are periodically re-assessed and REFINED (full content rewrite) when quality has degraded or content has become bloated from repeated enrichment.
- **Implicit security screening:** Because skills are self-generated through reflection on execution traces and filtered through quality gates, they inherit the trust properties of the base model rather than requiring external audit of potentially adversarial content.

This creates a dual feedback loop: task outcomes drive lifecycle decisions (retire, specialize), while content quality drives content improvement (validate, refine). The two signals are complementary and independently actionable. Critically, this architecture also provides **defense-in-depth against skill injection**: even if adversarial content entered the library (e.g., through manipulated task inputs), the continuous lifecycle operations would detect and remove it through performance-based retirement or quality-driven refinement.

### 1.7 The PRISM Loop

Putting it all together, PRISM implements a 5-step loop per task:

```
For each task tᵢ:
  ① RETRIEVE    Multi-layer skill retrieval with Pareto-aware selection
                 → select top-k skills Sᵢ balancing exploit (proven) and explore (untested)

  ② EXECUTE     Run augmented prompt π₀ ⊕ Sᵢ through evaluation function
                 → score, execution trace, output

  ③ REFLECT     LLM analyzes trace → per-skill attributions (helpful/harmful/neutral)
                 → knowledge gap diagnoses

  ④ CURATE      Lifecycle operations informed by reflection + Pareto signals:
                 NO-OP      — skills performing well, no gaps identified → skip curation
                 RETIRE     — ε-dominated (with LOO verification) or high harmful ratio
                 BIRTH      — from reflection gaps + coverage gaps
                 ENRICH     — append to existing skill (with quality gate)
                 SPECIALIZE — split Pareto-optimal but inconsistent skills (with LOO verification)
                 GENERALIZE — merge keyword-overlapping skills
                 REFINE     — periodic quality-driven content rewrite

  ⑤ DIFFERENTIAL Compute δ = μ_new − μ_ref against operation-specific baseline (§3.5)
                 → attribute δ per skill: record (δ, δ̂, co-selected set, solo flag)
                 → for lifecycle operations, verify on replay buffer and accept or roll back
                 → update EMA index for future retrieval ranking (Layer 3 of §3.1)
```

The library L persists across tasks and epochs, with each skill independently accumulating performance data, undergoing quality assessment, and evolving through lifecycle operations. The score matrix stores attribution-refined entries that mitigate the set-vs-skill credit problem (§2.1), and high-stakes lifecycle decisions are confirmed by targeted leave-one-out verification (§3.5). Over time, the library converges toward a set of high-quality, non-redundant, specialized skills that collectively cover the task distribution.

### 1.8 Hypotheses

**H1 — Skill decomposition outperforms monolithic prompts.** By decomposing knowledge into independent, composable skill units, PRISM avoids knowledge interference and enables task-specific skill combinations that a single prompt cannot express.

**H2 — Per-instance Pareto signals produce better lifecycle decisions.** The score matrix enables domination-based retirement and frequency-based specialization triggers that are more principled than aggregate heuristics (harmful−helpful > 3, variance > 0.20).

**H3 — Content-level quality control improves the library over time.** By validating generated content at creation time and periodically refining existing content, the library maintains higher average skill quality than systems relying solely on outcome-based pruning.

**H4 — The dual feedback loop (outcome + content quality) converges faster.** Outcome signals tell the system *which* skills to keep; quality signals tell the system *how* to improve them. The combination reduces the number of tasks needed to build an effective library.

**H5 — Lifecycle operations provide natural robustness to skill injection attacks.** Even if malicious content were introduced into a skill (through adversarial task inputs or corrupted reflection), PRISM's continuous curation mechanisms would neutralize it: harmful skills are RETIRED through Pareto-based domination or high harmful-ratio detection; inconsistent skills are SPECIALIZED to isolate problematic content; and REFINE operations rewrite degraded content under quality gates. The system's self-correcting dynamics make injected attacks transient rather than persistent.

---

## 2. Skill Evaluation

### 2.1 The Set-vs-Skill Attribution Problem

A naive way to evaluate the skill is through differential score which is defined as:

$$\delta_i = \mu_{\text{new}}(i) - \mu_{\text{ref}}(i)$$

where the reference baseline runs without the selected skills or without the updates on selected skills.

However, the raw differential score $\delta_i$ measures the contribution of the **entire selected skill set** $S_i$, not any individual skill. Yet the score matrix records this set-level signal on each skill independently. When the Assembler selects skills $\{A, B, C\}$ for a task and the differential $\delta = +0.3$, all three skills receive $\delta = +0.3$ in their score matrices. This creates three problems:

1. **Credit misattribution.** Skill $A$ might be the sole contributor ($+0.3$), while $B$ and $C$ are neutral or even slightly harmful — but masked by $A$'s contribution. All three receive undeserved credit.

2. **Corrupted Pareto comparisons.** If skill $D$ was selected alone on the same instance and scored $+0.2$, the matrix says $A > D$, $B > D$, $C > D$ — but only $A > D$ may be causally true. This can lead to **incorrect $\varepsilon$-domination and premature retirement of $D$**.

3. **Inconsistent lifecycle signals.** A skill's harmful ratio (from Reflector attributions) may conflict with its score matrix entries. The Reflector says "$B$ was harmful" but the score matrix says $B$ scored $+0.3$ because $A$ compensated. The two signals diverge precisely when they should agree.

### 2.2 Attribution-Gated Scoring with Targeted Leave-One-Out

The core mechanism underlying PRISM's lifecycle decisions is a Pareto-guided skill evaluation system that operates across three levels: per-instance score tracking, skill-level Pareto frequency, and soft ε-domination. This section defines these data structures and the domination relations that drive skill retirement, specialization, and retrieval boosting.

Each skill $\sigma$ maintains a sparse score matrix $S[\sigma]$: a mapping from task instance keys to attribution-refined score entries.

$$S[\sigma]: h(t_i) \rightarrow (\hat{\delta}_i,\ C_i,\ \text{solo}_i)$$

where $h(t_i)$ is an 8-character MD5 hash of the task question text and:
- $\hat{\delta}_i$: the **attribution-refined delta**, computed from the Reflector's per-skill causal judgment (see below)
- $C_i \subseteq \mathcal{L}$: the set of co-selected skills on this instance
- $\text{solo}_i \in \{\text{true}, \text{false}\}$: whether $\sigma$ was the only skill selected

As discussed in §2.1, the differential $\delta(j)$ is a set-level signal shared by all co-selected skills. The score matrix addresses this through a two-tier approach:

**Tier 1 — Attribution-refined scoring (routine).** On every task, each skill receives an attributed delta $\hat{\delta}(\sigma, j)$ that respects the Reflector's causal judgment: helpful skills inherit $\delta$, neutral skills receive 0, and harmful skills receive $\min(\delta, -\varepsilon)$. The co-selection context $C_j$ and solo flag are recorded alongside (see §2.1). This is cheap — no extra evaluations — and provides a reasonable per-skill signal for routine Pareto updates.


<!-- The attribution-refined delta $\hat{\delta}$ addresses the set-vs-skill problem by partitioning the set-level signal using the Reflector's per-skill attribution:

$$\hat{\delta}(\sigma, j) = \begin{cases} \delta(j) & \text{if attribution}(\sigma, j) = \texttt{helpful} \\ 0 & \text{if attribution}(\sigma, j) = \texttt{neutral} \\ \min(\delta(j),\ -\varepsilon) & \text{if attribution}(\sigma, j) = \texttt{harmful} \end{cases}$$

This ensures that a skill judged harmful by the Reflector never receives a positive score, even when the overall skill set performed well. When $\text{solo} = \text{true}$, the raw $\delta$ and attributed $\hat{\delta}$ coincide (no co-selection ambiguity), making solo entries the cleanest signal in the matrix. -->


**Tier 2 — Leave-one-out verification (at lifecycle decision points).** Before committing to a high-stakes lifecycle operation (BIRTH OR RETIRE or SPECIALIZE), the system runs **targeted leave-one-out (LOO) evaluations** on the skill's replay buffer to obtain causal per-skill deltas:

$$\delta_{\text{LOO}}(\sigma, j) = \mu(S_j, j) - \mu(S_j \setminus \{\sigma\}, j)$$

where $S_j$ is the skill set used on instance $j$. This isolates $\sigma$'s marginal causal contribution by comparing performance with and without it while holding all other skills fixed. LOO verification runs on the replay buffer instances (at most $n_{\text{verify}}$ evaluations per skill), so cost is bounded and incurred only when a lifecycle decision is imminent.

The BIRTH decision requires LOO confirmation: too many skills will pollute context and confuse the agent.
The RETIRE decision requires LOO confirmation: a skill is only retired if it remains $\varepsilon$-dominated when its score matrix entries are replaced with $\delta_{\text{LOO}}$ values from the replay buffer. Similarly, SPECIALIZE checks whether the skill's inconsistency (high Pareto frequency + high harmful ratio) persists under LOO scores before splitting. This prevents acting on spurious credit or blame caused by co-selection effects.

<!-- 
### 2.1 Differential Evaluation for inter-skills lifecycle decisions.

Every lifecycle operation produces a skill that must be verified against a **reference baseline** on the same task instances. The framework is uniform: run the task with the new version and with the reference version, compute a paired difference, and accept or roll back based on the aggregate signal. What differs across operations is only the reference:

| Operation | New version | Reference baseline |
|---|---|---|
| **BIRTH** | Prompt with new skill included | Prompt without the new skill (base prompt $\pi_0$ or prompt with remaining skills) |
| **REFINE / ENRICH** | Prompt with modified skill | Prompt with old (pre-modification) skill |
| **SPECIALIZE** | Prompt with child skill | Prompt with parent skill |
| **GENERALIZE** | Prompt with merged skill | Prompt with the better-scoring original parent |

For BIRTH, a positive $\delta$ means the new skill adds value beyond the baseline. For modification operations, a positive $\delta$ means the change improved upon the previous version. In all cases, the system caches the reference version (old skill content or absence of skill) until verification completes.

The acceptance criterion is: $\text{mean}(\delta) \geq -\varepsilon$ across verified instances. If the criterion fails, the operation is **rolled back** — the old skill (or no skill, for BIRTH) is restored. -->



### 2.3 Pareto Frequency and Soft $\varepsilon$-Domination

After each curation step, the system computes per-skill **Pareto frequency** — the fraction of evaluated instances on which a skill is non-dominated.

For each task instance key $j$ present in any active skill's score matrix, the best observed score is:

$$s^*(j) = \max_{\sigma \in \mathcal{L}_{\text{active}}} \hat{\delta}(\sigma, j)$$

where $\hat{\delta}(\sigma, j)$ is the attribution-refined delta from §2.1. A skill $\sigma$ is **non-dominated on instance $j$** if its score is within $\varepsilon$ of the best:

$$\hat{\delta}(\sigma, j) \geq s^*(j) - \varepsilon$$

where $\varepsilon = 0.05$ is a soft tolerance threshold. The Pareto frequency of skill $\sigma$ is then:

$$f(\sigma) = \frac{|\{j \mid \sigma \text{ is non-dominated on } j\}|}{|\{j \mid j \in S[\sigma]\}|}$$

This scalar $f(\sigma)$ serves as the primary signal for three downstream decisions: skill selection (§3.1), skill retirement, and skill specialization (§3.4).

To make lifecycle decisions, we define a conservative domination relation. Skill $\sigma_a$ **$\varepsilon$-dominates** skill $\sigma_b$ if and only if:

1. They share at least $n_{\min} = 3$ evaluated instances: $|K_a \cap K_b| \geq n_{\min}$, where $K_\sigma = \text{keys}(S[\sigma])$
2. On every shared instance, $\sigma_a$ scores at least as well (within tolerance): $\forall j \in K_a \cap K_b: \hat{\delta}(\sigma_a, j) \geq \hat{\delta}(\sigma_b, j) - \varepsilon$
3. On at least one shared instance, $\sigma_a$ is strictly better: $\exists j \in K_a \cap K_b: \hat{\delta}(\sigma_a, j) > \hat{\delta}(\sigma_b, j)$
4. The skills are semantically similar (competing for the same niche): $\text{sim}(\sigma_a, \sigma_b) \geq \theta_{\text{sim}}$, where $\text{sim}(A, B) = \cos(\text{emb}(A.\text{description}), \text{emb}(B.\text{description}))$ and $\theta_{\text{sim}} = 0.5$

**Why conditions 1-3?** Condition 1 prevents premature retirement when two skills have insufficient comparison data. The $\varepsilon$ tolerance in condition 2 avoids retiring a skill that is only marginally worse on some instances — it must be consistently dominated, not just occasionally slightly behind.

**Why condition 4 (semantic similarity)?** Domination comparisons only make sense between skills that compete for the same niche. A geometry skill shouldn't be retired because an algebra skill "dominates" it on algebra tasks — they serve different purposes. If $\text{sim}(A, B) < \theta_{\text{sim}}$, the skills target different task types and neither can dominate the other regardless of scores. This prevents semantically inappropriate retirements (apples vs oranges).

**The correlation-causation problem.** Conditions 1-3 can still be too aggressive because the score matrix records **correlation, not causation**. When skills A and B are frequently co-selected for the same tasks, they receive identical differential scores on those instances — both get credit when the task succeeds, both get blamed when it fails. Skill A may appear to ε-dominate skill B simply because they were always selected together and A happened to also be selected on a few additional successful instances. But B may have **independent causal contribution** that we never observe because B was never tested in isolation.

Consider the example: an "algebra_tricks" skill and a "geometry_basics" skill are co-selected on instances 1-5 (mixed problem sets), both scoring [+0.3, +0.2, +0.4, +0.1, +0.3]. The algebra skill is additionally selected alone on instances 6-7 (pure algebra problems), scoring [+0.2, +0.1]. By conditions 1-3 alone, algebra_tricks would "dominate" geometry_basics. But this is doubly spurious: (1) the skills serve different purposes — $\text{sim}(\text{algebra}, \text{geometry}) < \theta_{\text{sim}}$ — so condition 4 fails and dominance check should not apply; (2) even if they were similar, we have no evidence that algebra alone caused the shared successes on instances 1-5. Geometry might be equally effective on geometry-related aspects of those problems; we simply never tested geometry without algebra.

PRISM mitigates the correlation-causation problem through three layers: (1) attribution-refined deltas $\hat{\delta}$ replace raw set-level scores, preventing harmful skills from inheriting credit (§2.1); (2) condition 4 blocks cross-niche domination; and (3) targeted leave-one-out verification at lifecycle decision points provides causal confirmation before committing to RETIRE or SPECIALIZE (§3.5).

A skill is **Pareto-optimal** if no other active skill $\varepsilon$-dominates it. The set of Pareto-optimal skills forms the **Pareto front** of the library.


### 2.4 Pareto Computation with Mixed Signals

All downstream Pareto computations (§2.2, §2.3) use the attributed delta $\hat{\delta}$ rather than the raw $\delta$, with a preference for solo entries:

- When comparing two skills on a shared instance, **solo entries are preferred**. If both skills have solo entries on the same instance, the comparison is clean. If only one has a solo entry and the other has a co-selected entry, the solo entry is treated as higher-confidence.
- When neither skill has a solo entry on a shared instance, the attributed deltas $\hat{\delta}$ are used. These are noisier (they depend on Reflector accuracy) but still better than raw set-level deltas.
- High-stakes lifecycle decisions (RETIRE, SPECIALIZE) trigger **targeted leave-one-out verification** before committing (see §3.5).

The matrix is inherently sparse: each skill is only evaluated on instances where the Assembler selected it. This sparsity is handled explicitly in all downstream computations (§2.3).

**Why hash the task question?** The full question text can be arbitrarily long, but an 8-character hex hash (truncated MD5) provides a compact, fixed-size identifier. With 16^8 ≈ 4 billion unique values, collision probability is negligible for typical dataset sizes. This enables efficient storage and lookup while preserving instance-level granularity.

**Example score matrix.** Consider a library with three skills evaluated across five task instances. Each cell shows the attributed delta $\hat{\delta}$ with co-selection context (positive = skill helped, negative = skill hurt, empty = skill not selected for that task, **bold** = solo entry):

| | inst 1 | inst 2 | inst 3 | inst 4 | inst 5 |
|---|:---:|:---:|:---:|:---:|:---:|
| **algebra_tricks** | **+0.3** | — | +0.1 (w/ geo) | **+0.4** | — |
| **geometry_basics** | — | **+0.2** | 0.0 (w/ alg) | — | **+0.5** |
| **number_theory** | +0.1 (w/ alg) | +0.3 (w/ geo) | — | **−0.2** | +0.2 (w/ geo) |

Here, algebra_tricks was selected alone on inst 1 and inst 4 (solo, high confidence). On inst 3, both algebra_tricks and geometry_basics were co-selected; the Reflector judged algebra helpful ($\hat{\delta} = +0.1$) and geometry neutral ($\hat{\delta} = 0.0$). The Pareto computations prefer the bold solo entries; co-selected entries use the attribution-refined values.

From this matrix we can compute:
- **Pareto frequency of algebra_tricks**: On inst 1 (solo), it scores +0.3 vs number_theory's +0.1 (co-selected) → non-dominated. On inst 3, +0.1 vs geometry's 0.0 → non-dominated. On inst 4 (solo), +0.4 vs number_theory's −0.2 (solo) → non-dominated. Pareto frequency = 3/3 = 1.0.
- **$\varepsilon$-domination check**: Does algebra_tricks dominate number_theory? Shared instances: {inst 1, inst 4} (inst 3 missing for number_theory). Attributed scores: $A = [+0.3, +0.4]$ vs $N = [+0.1, -0.2]$. $A \geq N$ on all shared instances and $A > N$ on at least one → algebra_tricks $\varepsilon$-dominates number_theory (pending $\geq 3$ shared instances). Note: inst 1's comparison is high-confidence (algebra is solo) while number_theory's +0.1 is co-selected (lower confidence).

---

## 3. The PRISM Loop

PRISM runs a five-step loop per task instance: **Retrieve → Execute → Reflect → Curate → Differential Evaluate → Index**. The system maintains a persistent skill library $\mathcal{L}$ where each skill independently accumulates performance data, undergoes lifecycle operations, and evolves over time.

### 3.1 Multi-Layer Skill Retrieval

At inference time, the Assembler selects a subset of skills for a new task instance $x$ through a four-layer scoring pipeline:

**Layer 1 — Module filter.** Only active skills matching the current module tag are considered.

**Layer 2 — Semantic similarity.** If an embedding function is available, each candidate skill is scored by cosine similarity between its embedding and the task embedding:

$$\text{sim}(\sigma, x) = \cos(\text{emb}(\sigma.\text{description}),\ \text{emb}(x))$$

**Layer 3 — EMA index lookup.** A task type index maintains exponential moving average (EMA) scores per skill per task type. After each task evaluation (step ⑤), the EMA index is updated for each skill on the current task type, smoothing noisy per-instance scores into a stable ranking per task category. At retrieval time, the index returns the top-scoring skills for the classified task type. The final base score is the maximum of Layer 2 and Layer 3:

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

### 3.2 Execution

The system executes the task with the augmented prompt — the base prompt $\pi_0$ concatenated with the selected skills $S_i$ from the retrieval step:

$$\hat{y}_i = M(\pi_0 \oplus S_i,\ x_i;\ \theta)$$

The execution produces: (1) the model output $\hat{y}_i$, (2) the evaluation score $\mu(\hat{y}_i, y_i)$ computed against the ground truth, and (3) the full execution trace (inputs, intermediate reasoning, and outputs) which is passed to the Reflector for attribution analysis.

### 3.3 The Reflect-Attribute Loop

After each task execution, the Reflector — a separate LLM call — analyzes the execution trace and produces:

- **Attributions**: For each skill used in the current task, the Reflector assigns a tag: `helpful` (the skill contributed to a correct solution), `harmful` (the skill led the model astray), or `neutral` (no discernible effect).
- **Gaps**: Descriptions of missing knowledge that would have helped solve the task.
- **Diagnosis**: A free-text analysis of what went right or wrong.

Attribution counts are accumulated on each skill (`helpful_count`, `harmful_count`, `neutral_count`) and inform both Pareto-based lifecycle decisions and the harmful ratio calculations used in RETIRE and SPECIALIZE triggers.

### 3.4 Curating the Skill Library

The Curator orchestrates seven lifecycle operations after each reflection step, guided by Pareto signals and reflector diagnoses.

#### NO-OP — Skip curation when no action is needed

When the selected skills performed well (all attributions are helpful or neutral) and the Reflector identifies no knowledge gaps, the Curator skips all lifecycle operations. This is the expected outcome for the majority of tasks once the library matures — most tasks are handled adequately by existing skills, and unnecessary curation would waste LLM calls without improving the library. The NO-OP decision is made before evaluating any other lifecycle triggers.

#### RETIRE — Remove dominated or consistently harmful skills

A skill is retired under two conditions:

- **Primary (Pareto-based):** The skill is $\varepsilon$-dominated by another active skill with at least $n_{\min}$ shared instances. This means a strictly better alternative exists across all evaluated instances.
- **Fallback (ratio-based):** When insufficient score matrix data exists for Pareto comparison, the skill is retired if its harmful ratio exceeds 0.7 with at least 5 total evaluations. This handles the cold-start period before score matrices are populated.

Retired skills are marked with `status = "retired"` and excluded from future retrieval.

#### BIRTH — Create new skills from identified gaps

New skills are created when the Reflector identifies knowledge gaps — areas where the current skill library lacks relevant strategies. Two sources of gap signals are combined:

- **Reflector gaps:** The Reflector analyzes execution traces and diagnoses specific missing knowledge (e.g., "no strategy for modular arithmetic with large primes").
- **Coverage gaps:** The system scans all task instance keys across active skills and identifies instances where the best attributed delta among all active skills falls below a coverage threshold ($\theta_{\text{cov}} = 0.3$) — i.e., instances where every skill performed poorly. This detects "blind spots" in the library: regions of the task distribution that no existing skill covers well. For example, if the library has skills for algebra and geometry but keeps scoring below 0.3 on number theory problems, those instances surface as coverage gaps. The threshold $n_{\text{gap}} = 2$ requires at least two such uncovered instances before triggering, avoiding overreaction to a single hard outlier.

Both gap sources are merged and passed to an LLM that decides whether to CREATE a new skill or ENRICH an existing one.

#### ENRICH — Append new knowledge to existing skills

When the LLM determines that an identified gap is best addressed by augmenting an existing skill, new content is appended as a markdown section. A content budget (8,000 characters) prevents unbounded growth:

- If the enriched content stays within budget, it is appended directly.
- If it exceeds the budget, the oldest enrichment section (identified by `## Enrichment` headers) is removed before appending.
- If still over budget after removal, the content is hard-truncated.

#### SPECIALIZE — Split broadly relevant but inconsistent skills

A skill is a candidate for specialization when it is **both frequently Pareto-optimal and frequently harmful**:

$$f(\sigma) \geq \theta_{\text{pareto}} \quad \text{AND} \quad \frac{n_{\text{harmful}}}{n_{\text{evals}}} \geq \theta_{\text{harmful}}$$

with $\theta_{\text{pareto}} = 0.4$ and $\theta_{\text{harmful}} = 0.25$, requiring at least 5 total evaluations. This trigger identifies skills that contain valuable knowledge for some task types but hurt performance on others — the ideal candidates for splitting.

An LLM receives the parent skill's content and generates two specialized child skills with narrower trigger conditions. The parent is retired, and both children inherit the parent's `skill_id` as their `parent_id` for lineage tracking.

#### GENERALIZE — Merge overlapping skills

When two active skills share sufficient keyword overlap (Jaccard similarity $\geq 0.6$), they are candidates for merging into a single, more comprehensive skill. An LLM combines the content of both skills, and both originals are retired.

#### REFINE — Quality-driven content rewrite

Periodically, active skills undergo content-level quality assessment via the SkillValidator. Skills whose content has degraded (e.g., through repeated enrichment) or whose quality score falls below threshold are fully rewritten by an LLM while preserving the core strategies. This operation is triggered by the quality audit cycle rather than by task-level outcomes.

### 3.5 Differential Evaluation for lifecycle operation
#### Offline Setting (Multi-Epoch over Fixed Dataset)

Following GEPA's approach of evaluating candidates on a fixed validation set, PRISM maintains a **validation split** $D_{\text{val}}$ of the training data. After a lifecycle operation, the system re-evaluates the new version on $D_{\text{val}}$ (or a subsample of size $n_{\text{verify}} = \min(20, |D_{\text{val}}|)$) and compares per-instance scores against the reference version's scores on the same instances. Since the full dataset is available, verification is immediate — no waiting for future tasks.

#### Online Setting (Streaming Tasks)

In the online setting, verification leverages the normal PRISM loop to avoid redundant evaluations. The key observation is that one of the two scores needed for paired comparison is **already produced by the loop** — only one additional evaluation is required per task.

**On the task that triggers the lifecycle operation:** The loop has already executed the task with the old skill in step ② (Execute), producing $\mu_{\text{old}}$. After curation modifies the skill in step ④, the system runs **one additional evaluation** with the new skill on the same task, producing $\mu_{\text{new}}$. This yields the first paired difference $\delta = \mu_{\text{new}} - \mu_{\text{old}}$ at a cost of one extra evaluation.

**Replay buffer for immediate verification:** To avoid waiting for future tasks, each skill maintains a small **replay buffer** of the last $n_{\text{verify}}$ task instances it was applied to. After a lifecycle operation, the system immediately replays these buffered instances with the new skill, producing $\mu_{\text{new}}$ for each. The old scores $\mu_{\text{old}}$ are already stored in the score matrix from the original execution. This yields $n_{\text{verify}}$ paired differences immediately, enabling an instant accept/rollback decision without waiting for future tasks.

The replay buffer stores only $n_{\text{verify}} = \min(5, |K_\sigma|)$ task instances per skill — a bounded, small memory cost. Buffers are maintained in a FIFO fashion: as the skill is applied to new tasks, the oldest buffered instance is evicted. After verification completes, the buffer continues accumulating instances for potential future lifecycle operations.

For SPECIALIZE, both children enter verification simultaneously, each caching the parent as reference. For GENERALIZE, the merged skill caches the better-scoring parent. If multiple skills undergo lifecycle operations concurrently, each independently maintains its own reference cache, with total cache size bounded by the number of concurrent operations (typically 1–2 per curation step).

### 3.6 Handling Sparsity and Cold Start

The score matrix is very sparse, especially early in optimization. PRISM uses three mechanisms to handle this gracefully:

1. **Optimistic inclusion in Pareto front.** Skills with no score matrix entries are included in the Pareto front by default — they cannot be dominated because there is no evidence against them. This prevents premature retirement of untested skills.

2. **Fallback lifecycle heuristics.** When score matrices have insufficient data for $\varepsilon$-domination checks ($< n_{\min}$ shared instances), the system falls back to aggregate ratio-based heuristics (harmful ratio $> 0.7$) for retirement decisions. As score matrices populate, the Pareto-based mechanism smoothly takes over.

3. **Adaptive exploration.** The Assembler allocates more explore slots when the library is immature (total evaluations $< n_{\text{mature}}$), accelerating score matrix population. Unevaluated skills receive a fixed exploration bonus to ensure they are selected and tested. As the library matures, exploration decreases in favor of exploitation of proven skills.

### 3.7 Multi-Epoch Training

The system supports multi-epoch training over a fixed task set. In each epoch, every task instance is processed through the full five-step loop. Across epochs:

- The score matrix accumulates more entries, making Pareto computations increasingly accurate
- Lifecycle operations progressively retire dominated skills, specialize inconsistent ones, and birth new skills for coverage gaps
- The library converges toward a set of high-quality, non-redundant, specialized skills

Periodic maintenance (every $n_{\text{maint}}$ steps) persists the library and index to disk, and optional validation on a held-out set monitors generalization performance.
