# SCULPT: Skill Curation Using Lifecycle Pareto Tracking

## 1. Problem Formulation

### 1.1 The Challenge

We address the problem of **automated knowledge accumulation for LLM systems** — enabling an LLM to progressively build, evaluate, and curate a persistent library of reusable problem-solving strategies ("skills") through self-reflection on its own execution traces, without modifying model weights.

This problem introduces three intertwined challenges. First, when multiple skills are selected together, the observed performance improvement reflects the **entire skill set**, not any individual skill — making it difficult to attribute credit or blame to specific skills (the set-vs-skill attribution problem; §2.1). Second, lifecycle decisions that change the library's composition (adding, removing, or splitting skills) require **causal evidence** of individual skill value, yet exhaustive leave-one-out evaluation is prohibitively expensive (§2.2). Third, skill content can degrade over time through repeated modification, requiring **content-level quality control** independent of outcome-based evaluation (§1.5). SCULPT addresses all three challenges within a unified framework.

### 1.2 Formal Setup

Consider an LLM system with frozen weights $\theta$ that processes a sequence of tasks $T = (t_1, t_2, \ldots, t_n)$ drawn from a task distribution $\mathcal{D}$. Each task $t_i = (x_i, y_i)$ consists of an input $x_i$ and a ground-truth evaluation target $y_i$. The system augments its base prompt $\pi_0$ with a dynamically selected subset of skills from a persistent skill library $\mathcal{L}$.

A **skill** $\sigma \in \mathcal{L}$ is a structured knowledge unit $\sigma = (\text{name}, \text{description}, \text{content}, \text{metadata})$. Its fields serve three distinct functions:

**Retrieval fields** — used by the Assembler to select skills for a given task (§3.1):
- **description**: natural language trigger condition specifying when to apply this skill
- **task\_types**, **keywords**: semantic categories and tags for task-type filtering and keyword matching
- **embedding**: vector representation for cosine similarity scoring (Step 2 of the retrieval pipeline)

**Evaluation fields** — used for performance tracking and lifecycle decisions (§2):
- **score\_matrix**: per-instance attribution-refined performance record mapping task hashes to $(\hat{\delta}, C, \text{solo})$ entries (§2.2)
- **helpful\_count**, **harmful\_count**, **neutral\_count**: attribution counters accumulated by the Reflector (§3.3)
- **pareto\_frequency**: fraction of evaluated instances where the skill is Pareto-optimal (§2.4)

**Content and lifecycle fields**:
- **content**: a markdown document encoding reusable strategies, workflows, and domain knowledge (capped at 8,000 characters, approximately 2,000 tokens)
- **trigger\_conditions**, **strategy\_notes**, **pitfalls**: structured metadata for content organization
- **status**: "active" or "retired"
- **parent\_id**, **children\_ids**: lineage tracking for SPECIALIZE and GENERALIZE operations

Skills are persisted as markdown files (`SKILL.md`) with YAML frontmatter and stored in a directory-based library. A companion `_meta.json` file tracks all numerical metadata, score matrices, and lineage information.

The skill library $\mathcal{L} = \{\sigma_1, \sigma_2, \ldots, \sigma_m\}$ evolves over time through six lifecycle operations (§3.4): NO-OP (skip), BIRTH (create), ENRICH (append or rewrite), SPECIALIZE (split), GENERALIZE (merge), and RETIRE (deactivate). Four of these — BIRTH, RETIRE, SPECIALIZE, GENERALIZE — change the library's composition and require targeted leave-one-out verification (§2.2, Tier 2) before committing.

### 1.3 Optimization Objective

At each step $i$, the system executes a five-step loop (§1.6):

1. **Retrieves** a subset $S_i \subseteq \mathcal{L}$ of skills relevant to task $t_i$ via a five-step retrieval pipeline (§3.1)
2. **Executes** the task with the augmented prompt: $\hat{y}_i = M(\pi_0 \oplus S_i, x_i; \theta)$
3. **Reflects** on the execution trace to produce per-skill attributions and knowledge gap diagnoses (§3.3)
4. **Curates** the library $\mathcal{L}$ through lifecycle operations informed by reflection and Pareto signals (§3.4)
5. **Evaluates** differentially: computes $\delta = \mu_{\text{new}} - \mu_{\text{ref}}$ and updates the score matrix (§2.3, §3.5)

The objective is to maximize expected task performance over the distribution:

$$\max_{\mathcal{L}} \; \mathbb{E}_{t \sim \mathcal{D}} \left[ \mu\!\left(M(\pi_0 \oplus R(t, \mathcal{L}),\, x;\, \theta),\, y\right) \right]$$

subject to the constraint that model weights $\theta$ remain frozen and the only adaptable component is the skill library $\mathcal{L}$. The retrieval function $R(t, \mathcal{L})$ is itself shaped by the library's evaluation state — Pareto frequencies, attribution counters, and exploration bonuses (§3.1) — making the optimization a closed loop where evaluation signals feed back into retrieval.

### 1.4 Core Problems

#### 1.4.1 Why Skills, Not Prompts?

Existing approaches to context adaptation optimize either **monolithic prompts** (rewritten wholesale at each mutation) or **flat memory lists** (bullets appended and pruned over time). These representations face fundamental limitations:

**Problem 1 — Knowledge interference.** A monolithic prompt or flat memory must encode all strategies for all task types in a single text block. Improving instructions for geometry problems may degrade performance on combinatorics — all knowledge competes for the same space. SCULPT eliminates this through independent skill units that are selected in task-specific combinations (§3.1) and undergo lifecycle operations independently (§3.4).

**Problem 2 — No compositional reuse.** Strategies discovered during optimization are locked inside prompt text or memory bullets. They cannot be extracted, recombined, or reused across different task contexts. SCULPT's retrieval pipeline (§3.1) selects arbitrary skill subsets per instance, enabling compositional reuse — the same skill can contribute to different task combinations without duplication.

**Problem 3 — Catastrophic forgetting under rewrite.** When an LLM rewrites accumulated context, it tends to compress or drop previously accumulated knowledge — a phenomenon known as "context collapse." Append-only strategies avoid this but lead to bloated, unfocused context over time. SCULPT's ENRICH operation (§3.4) addresses both failure modes: incremental append ($\alpha > 0$) preserves existing content while adding new knowledge, and full rewrite ($\alpha \approx 0$) is applied only under quality validation gates (§1.5), with score matrix entries decayed proportionally to the semantic change (§2.3).

The emerging consensus confirms this direction: recent surveys characterize skill engineering as a paradigm shift beyond prompt engineering and atomic tool use (Wu et al., 2026; Jiang et al., 2026), where skills are composable packages of instructions, workflows, and metadata that agents load on demand.

#### 1.4.2 Why SCULPT, Not Existing Skill Systems?

A wave of concurrent work has begun exploring skill-based architectures for LLM agents. We identify several key approaches and the gaps SCULPT addresses:

**SkillRL** (Li et al., 2026) distills raw trajectories into a hierarchical skill library (SkillBank) and co-evolves skills with the agent's policy via reinforcement learning. **SAGE** (Yang et al., 2025) similarly incorporates skill libraries into an RL training loop (GRPO), using sequential rollout across task chains. Both systems **require weight updates** — the skill library and the agent's policy are jointly trained. SCULPT operates with **frozen model weights**, making it applicable to black-box API models and requiring no training infrastructure.

**AutoSkill** (Wang et al., 2026) is closest in spirit to SCULPT: it extracts reusable skills from interaction traces without retraining, supports a skill lifecycle (creation, evolution, injection), and positions itself as a model-agnostic plug-in. However, AutoSkill lacks three capabilities that SCULPT provides: (1) **per-instance performance tracking** — SCULPT maintains score matrices with attribution-refined deltas $\hat{\delta}$ per instance (§2.2), enabling Pareto-based lifecycle decisions (§2.4) rather than heuristic quality signals; (2) **principled lifecycle verification** — composition-changing operations are confirmed by targeted leave-one-out verification (§2.2, Tier 2), bounding the cost to $n_{\text{verify}}$ evaluations per decision; and (3) **credit attribution** — the set-vs-skill attribution problem (§2.1) is addressed through a two-tier evaluation design that decomposes set-level differentials into per-skill signals.

**MACLA** (Forouzandeh et al., 2025) maintains hierarchical procedural memory with Bayesian reliability tracking and contrastive refinement. While it shares SCULPT's goal of frozen-weight adaptation with principled evaluation, it operates at the **procedure level** (step-by-step action sequences) rather than the **strategy level** (reusable problem-solving knowledge). MACLA's procedures are tightly coupled to specific action spaces (ALFWorld, WebShop), whereas SCULPT's skills are domain-agnostic knowledge documents.

**SkillCraft** (Chen et al., 2026) introduces a benchmark that stress-tests whether LLM agents can abstract and reuse higher-level tool compositions — which they term *Skills* — across realistic, compositional tool-use scenarios with difficulty scaled along quantitative and structural dimensions. Their lightweight evaluation protocol enables agents to auto-compose atomic tools into executable callable functions, cache them in a persistent skill library, and reuse them within and across tasks. Evaluating state-of-the-art agents on SkillCraft demonstrates substantial efficiency gains (up to 80% token reduction through skill reuse) and reveals that success rate strongly correlates with tool composition ability at test time. SkillCraft validates an important premise that SCULPT builds upon — compositional skill acquisition yields large efficiency gains — but the two systems differ in three fundamental respects. First, SkillCraft defines skills as **executable code** (auto-composed tool pipelines), whereas SCULPT defines skills as **natural-language knowledge documents** encoding domain-agnostic problem-solving strategies; this makes SCULPT's skills model-agnostic and transferable across different action spaces, while SkillCraft's executable skills are tightly coupled to specific tool APIs. Second, SkillCraft is a **benchmark and evaluation protocol** with a minimal skill lifecycle (compose → cache → retrieve → reuse), whereas SCULPT is a **lifecycle management framework** with six curation operations (BIRTH, ENRICH, SPECIALIZE, GENERALIZE, RETIRE, NO-OP), content-level quality gates (§1.5), and Pareto-based lifecycle decisions (§2.4). Third, SkillCraft evaluates skill quality through **execution-based validation** (did the composed skill produce a correct output?), whereas SCULPT addresses the **set-vs-skill attribution problem** (§2.1) through attribution-refined scoring (Tier 1) and targeted leave-one-out verification (Tier 2; §2.2), providing causal per-skill signals even when multiple skills are co-selected. Fourth, SkillCraft lacks a principled mechanism for **skill retirement or library pruning** — cached skills accumulate indefinitely — whereas SCULPT maintains a **Pareto front** over per-instance score vectors (§2.4): skills whose performance profiles are $\varepsilon$-dominated across instances are retired after LOO confirmation, ensuring the library converges toward a non-redundant, high-quality skill set rather than growing without bound.

**Problem 4 — Unaddressed security in skill systems.** As skill-based systems become prevalent, they introduce a novel **supply chain security threat**: malicious instructions embedded within otherwise legitimate skill content. Unlike traditional prompt injection (adversarial text in data), skill injection is an *instruction-instruction conflict* — bad instructions hidden among good ones — making data-instruction separation defenses inapplicable. Furthermore, many instructions are **dual-use**: the same action (e.g., "backup files to external server") can be legitimate in one context but constitute data exfiltration in another. Recent surveys identify this as a critical open problem (Wu et al., 2026), yet none of the above systems incorporate security mechanisms into their skill lifecycle. SCULPT's continuous curation provides **defense-in-depth**: harmful content is detected through attribution-refined scoring (§2.2, Tier 1), confirmed via targeted LOO verification (§2.2, Tier 2), and removed via RETIRE, isolated via SPECIALIZE, or corrected via ENRICH with full rewrite ($\alpha \approx 0$; §2.3).

**SCULPT's answer:** Decompose knowledge into **independent, persistent skill units** that:
- Are created, evaluated, and retired independently — eliminating knowledge interference (Problem 1)
- Can be selected in arbitrary combinations per task via Pareto-aware retrieval (§3.1) — enabling compositional reuse (Problem 2)
- Accumulate content via ENRICH with $\alpha$-decay (§2.3) or are fully rewritten under quality validation gates (§1.5) — preventing catastrophic forgetting (Problem 3)
- Track per-instance performance via score matrices, enabling **$\varepsilon$-domination-based lifecycle decisions** (§2.4) — unlike heuristic-driven evolution in AutoSkill or RL-coupled evolution in SkillRL/SAGE
- Solve the **set-vs-skill credit attribution problem** (§2.1) through attribution-refined deltas (Tier 1) and targeted leave-one-out verification (Tier 2; §2.2) — a challenge unaddressed by any existing skill system
- Are **self-generated and continuously curated**, providing inherent robustness to skill injection (Problem 4; H5, §1.7)

### 1.5 Content Quality as a First-Class Signal

Outcome-level evaluation — did the task succeed or fail? — conflates multiple factors: the quality of the skill content itself, the appropriateness of retrieval, the model's adherence to the skill, and the inherent task difficulty. SCULPT's two-tier evaluation (§2.2) addresses the performance dimension, but a skill can score well on recent tasks while its content gradually degrades — accumulating outdated strategies, redundant sections, or internally inconsistent advice through repeated enrichment.

SCULPT introduces a **content-level quality gate** via the SkillValidator, creating a second feedback channel orthogonal to performance:

- **Generation-time validation.** Every BIRTH, ENRICH, and SPECIALIZE output is scored by an LLM on five dimensions: structural completeness, actionability, description alignment, internal consistency, and differentiation from existing skills. Skills below threshold are revised or discarded before entering the library.
- **Periodic audit.** Active skills are periodically re-assessed. Skills whose quality has degraded — through content bloat from repeated gap-driven enrichment ($\alpha > 0$; §2.3) or through staleness as the task distribution shifts — are fully rewritten via ENRICH with $\alpha \approx 0$ (§3.4). The near-zero retention factor effectively discards old score matrix entries (§2.3), and the skill re-enters cold-start (§3.6) for fresh evaluation.

This creates a **dual feedback loop**: outcome signals from the score matrix (§2.2) drive composition-changing lifecycle decisions (RETIRE, SPECIALIZE, GENERALIZE), while content quality signals drive content improvement (ENRICH). The two channels are complementary — a skill may perform well but have degraded content (quality audit triggers rewrite), or have high-quality content but poor performance (Pareto signals trigger retirement). Neither signal alone suffices.

### 1.6 The SCULPT Loop

Putting it all together, SCULPT implements a five-step loop per task:

```
For each task tᵢ:
  ① RETRIEVE    5-step pipeline: task-type filter → semantic similarity
                 → Pareto boost → exploration bonus → pool-based selection (§3.1)
                 → select top-k skills Sᵢ balancing exploit and explore slots

  ② EXECUTE     Run augmented prompt π₀ ⊕ Sᵢ on task instance (§3.2)
                 → score μᵢ, execution trace, model output

  ③ REFLECT     Reflector analyzes trace → per-skill attributions (§3.3)
                 → attribution-refined deltas δ̂ via Tier 1 scoring (§2.2)
                 → knowledge gap diagnoses

  ④ CURATE      Six lifecycle operations informed by reflection + Pareto signals (§3.4):
                 NO-OP      — no gaps, no degradation → skip curation
                 BIRTH      — new skill from gap signals (LOO verified; §2.2)
                 ENRICH     — append (α > 0) or full rewrite (α ≈ 0) with score decay (§2.3)
                 SPECIALIZE — split inconsistent skill (LOO verified; §2.2)
                 GENERALIZE — merge overlapping skills (LOO verified; §2.2)
                 RETIRE     — remove ε-dominated skill (LOO verified; §2.2) or high harmful ratio

  ⑤ EVALUATE    Differential evaluation: δ = μ_new − μ_ref (§3.5)
                 → update score matrix via EMA (NO-OP) or convex decay (ENRICH) (§2.3)
                 → for composition-changing operations, verify on replay buffer → accept or rollback
```

The library $\mathcal{L}$ persists across tasks and epochs. Each skill independently accumulates performance data in its score matrix (§2.2), undergoes content quality assessment (§1.5), and evolves through lifecycle operations (§3.4). The score matrix stores attribution-refined entries $\hat{\delta}$ that mitigate the set-vs-skill credit problem (§2.1), updated via EMA smoothing for unchanged skills or convex decay for modified skills (§2.3). High-stakes decisions that change the library's composition — BIRTH, RETIRE, SPECIALIZE, GENERALIZE — receive causal confirmation through targeted leave-one-out verification (§2.2, Tier 2). Over time, the library converges toward a set of high-quality, non-redundant, specialized skills that collectively cover the task distribution.

### 1.7 Hypotheses

**H1 — Skill decomposition outperforms monolithic prompts.** By decomposing knowledge into independent, composable skill units, SCULPT avoids knowledge interference (Problem 1, §1.4.1) and enables task-specific skill combinations that a single prompt cannot express. The five-step retrieval pipeline (§3.1) selects complementary subsets per instance, while independent lifecycle operations (§3.4) ensure that modifying one skill does not degrade others.

**H2 — Attribution-refined evaluation produces better lifecycle decisions than aggregate heuristics.** Per-instance score matrices with attribution-refined deltas $\hat{\delta}$ (§2.2) enable $\varepsilon$-domination-based retirement and frequency-based specialization triggers (§2.4) that are more principled than aggregate counters. By gating scores through Reflector attributions (Tier 1) and confirming composition-changing decisions with targeted leave-one-out verification (Tier 2), SCULPT reduces both false retirements and false preservations compared to threshold-based heuristics.

**H3 — Content-level quality control improves the library independently of outcome signals.** The dual feedback loop (§1.5) — outcome-driven lifecycle decisions via the score matrix and content-driven quality improvement via the SkillValidator — addresses failure modes that neither channel catches alone. Outcome signals identify *which* skills to keep or retire; quality signals identify *how* to improve retained skills. The unified ENRICH operation (§3.4) enables both incremental improvement ($\alpha > 0$) and full recovery from content degradation ($\alpha \approx 0$, re-entering cold-start via §3.6).

**H4 — The two-tier evaluation design achieves near-causal accuracy at bounded cost.** Full leave-one-out evaluation over all skills and instances is prohibitively expensive ($O(m \cdot n)$). By restricting LOO to a single target skill on a bounded replay buffer ($n_{\text{verify}}$ evaluations per decision) and applying it only to composition-changing operations (BIRTH, RETIRE, SPECIALIZE, GENERALIZE), SCULPT achieves causal confirmation where it matters most while keeping routine evaluation free (§2.2).

**H5 — Continuous curation provides natural robustness to skill injection.** Even if malicious content were introduced into a skill (through adversarial task inputs or corrupted reflection), SCULPT's curation mechanisms would neutralize it: harmful skills are RETIRED through $\varepsilon$-domination confirmed by LOO (§2.2, §3.4); inconsistent skills are SPECIALIZED to isolate problematic content; and quality-driven ENRICH with full rewrite ($\alpha \approx 0$; §2.3) replaces degraded content under validation gates. Because every skill is continuously evaluated through the score matrix and periodically audited for content quality (§1.5), injected attacks are transient rather than persistent.

---

## 2. Skill Evaluation

### 2.1 The Set-vs-Skill Attribution Problem

SCULPT's lifecycle operations — retiring underperforming skills, specializing inconsistent ones, birthing new ones — all require **per-skill performance signals**. The natural approach is differential evaluation: compare task performance with and without a skill to measure its contribution.

$$\delta_i = \mu_{\text{new}}(i) - \mu_{\text{ref}}(i)$$

where $\mu_{\text{new}}(i)$ is the score obtained with the selected skill set and $\mu_{\text{ref}}(i)$ is a reference baseline (e.g., without the selected skills or with their pre-modification versions).

The fundamental problem is that $\delta_i$ measures the contribution of the **entire selected skill set** $S_i$, not any individual skill. Yet naively, this set-level signal is recorded on each co-selected skill independently. When the Assembler selects skills $\{A, B, C\}$ for a task and observes $\delta = +0.3$, all three skills receive $\delta = +0.3$ in their score matrices — regardless of their individual contributions. This conflation produces three downstream failures:

1. **Credit misattribution.** Skill $A$ might be the sole contributor ($+0.3$), while $B$ and $C$ are neutral or even slightly harmful — but masked by $A$'s strong contribution. All three receive identical, undeserved credit.

2. **Corrupted Pareto comparisons.** If skill $D$ was selected alone on the same instance and scored $+0.2$, the score matrix records $A > D$, $B > D$, $C > D$ — but only $A > D$ may be causally true. This can lead to **incorrect $\varepsilon$-domination and premature retirement of $D$** — a skill that may be independently valuable.

3. **Inconsistent lifecycle signals.** A skill's harmful ratio (derived from Reflector attributions) may contradict its score matrix entries. The Reflector judges "$B$ was harmful," yet the score matrix records $B$ at $+0.3$ because $A$ compensated for $B$'s negative effect. The two signals diverge precisely when they should agree, undermining lifecycle decisions that depend on both.

### 2.2 Attribution-Gated Scoring with Targeted Leave-One-Out

To address the set-vs-skill attribution problem (§2.1), SCULPT decomposes skill evaluation into two tiers that trade off between cost and causal accuracy. Each skill $\sigma$ maintains a sparse **score matrix** $S[\sigma]$: a mapping from task instance keys to attribution-refined score entries.

$$S[\sigma]: h(t_i) \rightarrow (\hat{\delta}_i,\ C_i,\ \text{solo}_i)$$

where $h(t_i)$ is an 8-character MD5 hash of the task question text and:
- $\hat{\delta}_i$: the **attribution-refined delta**, computed from the Reflector's per-skill causal judgment (see below)
- $C_i \subseteq \mathcal{L}$: the set of co-selected skills on this instance
- $\text{solo}_i \in \{\text{true}, \text{false}\}$: whether $\sigma$ was the only skill selected

**Tier 1 — Attribution-gated scoring (routine, zero additional cost).** On every task, the Reflector produces per-skill attributions as part of the standard reflect step (§3.3). The attribution-refined delta $\hat{\delta}$ partitions the set-level differential $\delta$ using these judgments:

$$\hat{\delta}(\sigma, j) = \begin{cases} \delta(j) & \text{if attribution}(\sigma, j) = \texttt{helpful} \\ 0 & \text{if attribution}(\sigma, j) = \texttt{neutral} \\ \min(\delta(j),\ -\varepsilon) & \text{if attribution}(\sigma, j) = \texttt{harmful} \end{cases}$$

This ensures that a skill judged harmful never receives a positive score, even when the overall skill set performed well. When $\text{solo} = \text{true}$, the raw $\delta$ and attributed $\hat{\delta}$ coincide — no co-selection ambiguity — making solo entries the highest-confidence signal in the matrix. The co-selection context $C_j$ and solo flag are recorded alongside each entry.

Tier 1 requires no additional evaluations beyond the standard loop and provides a reasonable per-skill signal for routine Pareto updates (§2.4). However, because attributions are LLM judgments rather than causal measurements, they can be noisy — particularly when multiple skills interact in complex ways.

**Tier 2 — Leave-one-out verification (targeted, high cost).** In principle, the most accurate way to evaluate every skill would be to run a full leave-one-out (LOO) sweep: for each skill in the library, re-evaluate all of its associated instances with the skill removed, yielding a causal marginal contribution per skill per instance. However, this is prohibitively expensive — a library of $m$ skills each evaluated on $n$ instances would require $O(m \cdot n)$ additional evaluations per curation step.

SCULPT avoids this cost by scoping LOO to a **single target skill** and a **bounded replay buffer**. When a lifecycle decision is triggered for skill $\sigma$, the system re-evaluates only that skill on at most $n_{\text{verify}}$ recent instances from its replay buffer:

$$\delta_{\text{LOO}}(\sigma, j) = \mu(S_j, j) - \mu(S_j \setminus \{\sigma\}, j)$$

where $S_j$ is the full skill set used on instance $j$. This directly measures $\sigma$'s marginal contribution while holding all other skills fixed — a causal signal free from the confounds of co-selection. The cost per lifecycle decision is exactly $n_{\text{verify}}$ evaluations, regardless of library size.

SCULPT further restricts LOO to **lifecycle decisions that change the library's composition** — operations where an incorrect decision is costly and difficult to reverse:

- **BIRTH** adds a skill to the library. Unchecked creation risks polluting the prompt context with low-value skills, degrading performance on tasks where the skill is retrieved. LOO confirms the new skill's marginal contribution is positive.
- **RETIRE** permanently removes a skill. A skill is only retired if it remains $\varepsilon$-dominated (§2.4) when its score matrix entries are replaced with $\delta_{\text{LOO}}$ values, preventing the retirement of skills that appear dominated due to co-selection artifacts.
- **SPECIALIZE** splits one skill into two narrower children, simultaneously removing the parent and introducing new skills. LOO verifies that the parent's inconsistency (high Pareto frequency combined with high harmful ratio) reflects genuine task-type conflict rather than noisy co-selection attribution.
- **GENERALIZE** merges two overlapping skills into one, retiring both originals. LOO is run on each parent skill to confirm that neither has independent causal value that would be lost in the merge — if either parent's $\delta_{\text{LOO}}$ reveals a unique contribution not captured by the other, the merge is aborted.

By contrast, ENRICH modifies a skill's content without changing the library's size or composition. This operation is lower-stakes: if a modification degrades performance, it surfaces naturally through Tier 1 attribution on subsequent tasks and can be corrected by a future ENRICH (including a full rewrite with $\alpha = 0$). It does not warrant the additional evaluation cost of LOO.

This two-tier design ensures that routine evaluation is free — piggybacking on the standard reflect step — while high-stakes, composition-changing decisions receive causal confirmation at bounded cost ($n_{\text{verify}}$ evaluations per decision, applied only to the target skill).

### 2.3 Score Matrix Update Rules

The score matrix accumulates entries through the normal SCULPT loop, but different lifecycle outcomes require different update strategies. This section defines how entries are inserted, smoothed, or decayed depending on whether the skill's content has changed.

#### NO-OP — EMA smoothing for unchanged skills

When no lifecycle operation is triggered (NO-OP), the skill's content is unchanged but it has been evaluated on a new task instance. If the instance is new (not in the matrix), the entry is simply inserted. If the instance was previously evaluated (e.g., in multi-epoch training), the score is updated via an **exponential moving average** (EMA):

$$\hat{\delta}\_{\text{updated}}(\sigma, j) = \beta \cdot \hat{\delta}\_{\text{old}}(\sigma, j) + (1 - \beta) \cdot \hat{\delta}\_{\text{new}}(\sigma, j)$$

where $\beta \in [0, 1)$ is a decay factor (default $\beta = 0.7$), consistent with the retention factor $\alpha$ used in ENRICH — both weight old evidence on the left and new evidence on the right. EMA smooths out LLM stochasticity across repeated evaluations while slightly favoring recent observations — appropriate because, even though the skill hasn't changed, the library context has evolved (new skills born, others retired, different co-selection patterns).

#### ENRICH — Convex decay for modified skills

ENRICH retains the skill's identity but changes its content, raising the question of how to handle pre-modification score matrix entries. Composition-changing operations (BIRTH, SPECIALIZE, GENERALIZE) do not have this problem — they produce new skill identities with fresh score matrices.

SCULPT addresses this with a **retention factor** $\alpha \in [0, 1]$ that discounts old evidence proportionally to the magnitude of the content change. After ENRICH, the enriched skill is immediately evaluated on the current task instance as part of step ⑤ (Differential Evaluate, §3.5), producing a fresh $\hat{\delta}\_{\text{new}}$. The score matrix entry for that instance is updated by combining the decayed old signal with the fresh evaluation:

$$\hat{\delta}\_{\text{updated}}(\sigma, j) = \alpha \cdot \hat{\delta}\_{\text{old}}(\sigma, j) + (1 - \alpha) \cdot \hat{\delta}\_{\text{new}}(\sigma, j)$$

This is a convex combination that smoothly interpolates between old and new evidence. When $\alpha \approx 1$ (small semantic change), the entry is dominated by historical performance. When $\alpha \approx 0$ (full rewrite), the entry reduces to the fresh evaluation alone. For instances not yet re-evaluated, existing entries are decayed to $\alpha \cdot \hat{\delta}\_{\text{old}}$ and serve as approximations until the skill is re-selected on them in future tasks. Attribution counters (`helpful_count`, `harmful_count`, `neutral_count`) are similarly scaled by $\alpha$.

The retention factor is computed as the **semantic similarity** between old and new content:

$$\alpha = \cos(\text{emb}(\text{content}\_{\text{old}}),\ \text{emb}(\text{content}\_{\text{new}}))$$

using the same embedding function available for retrieval (§3.1, Layer 1). This captures actual semantic drift rather than surface-level length changes — a short append that contradicts existing strategies (e.g., "Never use method X") produces low $\alpha$, while a large block of supplementary examples that preserves the core strategy produces high $\alpha$. A quality-driven full rewrite that fundamentally changes the skill's content yields $\alpha \approx 0$, effectively discarding old evidence and re-entering cold-start (§3.6).

### 2.4 Pareto Frequency and Soft $\varepsilon$-Domination

The score matrix defined in §2.2 records per-skill, per-instance attribution-refined deltas. This section defines the aggregate signals derived from these entries — Pareto frequency and soft $\varepsilon$-domination — that drive SCULPT's lifecycle decisions.

#### Pareto Frequency

After each curation step, the system computes per-skill **Pareto frequency**: the fraction of evaluated instances on which a skill is non-dominated. For each task instance key $j$ present in any active skill's score matrix, the best observed attributed score is:

$$s^*(j) = \max_{\sigma \in \mathcal{L}_{\text{active}}} \hat{\delta}(\sigma, j)$$

where $\hat{\delta}(\sigma, j)$ is the attribution-refined delta from §2.2 (Tier 1). A skill $\sigma$ is **non-dominated on instance $j$** if its score is within $\varepsilon$ of the best:

$$\hat{\delta}(\sigma, j) \geq s^*(j) - \varepsilon$$

where $\varepsilon = 0.05$ is a soft tolerance threshold. The Pareto frequency of skill $\sigma$ is then:

$$f(\sigma) = \frac{|\{j \mid \sigma \text{ is non-dominated on } j\}|}{|\{j \mid j \in S[\sigma]\}|}$$

This scalar $f(\sigma)$ serves as the primary signal for four downstream decisions: skill retrieval ranking (§3.1), retirement, generalization, and specialization triggers (§3.4).

#### Soft $\varepsilon$-Domination

To make lifecycle decisions, we define a conservative domination relation. Skill $\sigma_a$ **$\varepsilon$-dominates** skill $\sigma_b$ if and only if:

1. **Sufficient overlap:** They share at least $n_{\min} = 3$ evaluated instances: $|K_a \cap K_b| \geq n_{\min}$, where $K_\sigma = \text{keys}(S[\sigma])$
2. **Consistent superiority:** On every shared instance, $\sigma_a$ scores at least as well (within tolerance): $\forall j \in K_a \cap K_b: \hat{\delta}(\sigma_a, j) \geq \hat{\delta}(\sigma_b, j) - \varepsilon$
3. **Strict improvement:** On at least one shared instance, $\sigma_a$ is strictly better: $\exists j \in K_a \cap K_b: \hat{\delta}(\sigma_a, j) > \hat{\delta}(\sigma_b, j)$
4. **Semantic similarity:** The skills compete for the same niche: $\text{sim}(\sigma_a, \sigma_b) \geq \theta_{\text{sim}}$, where $\text{sim}(A, B) = \cos(\text{emb}(A.\text{description}), \text{emb}(B.\text{description}))$ and $\theta_{\text{sim}} = 0.5$

**Conditions 1–3** establish statistical and performance requirements. Condition 1 prevents premature retirement when two skills have insufficient comparison data. The $\varepsilon$ tolerance in condition 2 avoids retiring a skill that is only marginally worse on some instances — it must be consistently dominated, not just occasionally slightly behind.

**Condition 4 (semantic similarity)** restricts domination comparisons to skills that compete for the same niche. A geometry skill should not be retired because an algebra skill "dominates" it on algebra tasks — they serve different purposes. When $\text{sim}(A, B) < \theta_{\text{sim}}$, the skills target different task types and neither can dominate the other regardless of scores.

A skill is **Pareto-optimal** if no other active skill $\varepsilon$-dominates it. The set of Pareto-optimal skills forms the **Pareto front** of the library.

#### The Correlation-Causation Problem

Even with conditions 1–4, the $\varepsilon$-domination check operates on Tier 1 attribution-refined deltas $\hat{\delta}$, which record **correlation rather than causation**. When skills A and B are frequently co-selected, they receive correlated scores on shared instances — both get credit for successes, both get blamed for failures. Skill A may appear to $\varepsilon$-dominate skill B simply because they were always selected together and A happened to also be selected on a few additional successful instances. But B may have **independent causal contribution** that is never observed because B was never tested in isolation.

**Example.** An "algebra_tricks" skill and a "geometry_basics" skill are co-selected on instances 1–5 (mixed problem sets), both scoring [+0.3, +0.2, +0.4, +0.1, +0.3]. The algebra skill is additionally selected alone on instances 6–7 (pure algebra), scoring [+0.2, +0.1]. By conditions 1–3 alone, algebra_tricks would "dominate" geometry_basics. But this is doubly spurious:

1. The skills serve different purposes — $\text{sim}(\text{algebra}, \text{geometry}) < \theta_{\text{sim}}$ — so condition 4 blocks the domination check entirely.
2. Even if they were semantically similar, we have no evidence that algebra alone caused the shared successes on instances 1–5. Geometry might be equally effective on those problems; we simply never tested geometry without algebra.

SCULPT mitigates the correlation-causation problem through three layers, corresponding to the two-tier evaluation design in §2.2:

1. **Attribution-refined deltas** (Tier 1): $\hat{\delta}$ replaces the raw set-level $\delta$, preventing harmful skills from inheriting credit from co-selected helpful skills.
2. **Semantic similarity gating** (condition 4): Blocks cross-niche domination comparisons, ensuring skills are only compared against genuine competitors.
3. **Targeted leave-one-out verification** (Tier 2): Before committing to composition-changing lifecycle decisions (BIRTH, RETIRE, SPECIALIZE, GENERALIZE), LOO provides causal $\delta_{\text{LOO}}$ scores scoped to the target skill that confirm or refute the Tier 1 signal (§2.2).


### 2.5 Pareto Computation in Practice

Pareto computations (§2.4) operate on score matrix entries defined in §2.2 and maintained by the update rules in §2.3. In practice, these entries vary in confidence depending on their provenance, and the matrix is inherently sparse. This section describes how these factors are handled and provides a worked example.

#### Confidence Hierarchy

When comparing two skills on a shared instance, entries are not equally reliable. The score matrix records a confidence hierarchy based on provenance:

- **Solo entries** ($\text{solo} = \text{true}$) are highest confidence — the attributed delta $\hat{\delta}$ coincides with the true causal contribution, free from co-selection confounds (§2.2, Tier 1). When both skills have solo entries on a shared instance, the comparison is clean.
- **Co-selected entries** ($\text{solo} = \text{false}$) are lower confidence — the attributed delta depends on the Reflector's judgment, which can be noisy when multiple skills interact. When only one skill has a solo entry, it is treated as the more reliable signal.
- **EMA-smoothed entries** (§2.3, NO-OP) have reduced noise from averaging across repeated evaluations, improving reliability over single observations.
- **Decay-adjusted entries** (§2.3, ENRICH with $\alpha > 0$) are approximations of the modified skill's performance, with confidence proportional to $\alpha$. Entries where $\alpha \approx 0$ are near-reset and should be treated similarly to absent entries.

For high-stakes lifecycle decisions (BIRTH, RETIRE, SPECIALIZE, GENERALIZE), targeted LOO verification (§2.2, Tier 2) replaces these approximate signals with causal $\delta_{\text{LOO}}$ scores scoped to the target skill before committing.

#### Sparsity

The matrix is inherently sparse: each skill is only evaluated on instances where the Assembler selected it (§3.1). Skills with insufficient data for $\varepsilon$-domination checks receive optimistic Pareto inclusion and exploration bonuses through the cold-start mechanisms in §3.6.

#### Worked Example

Consider a library with three skills evaluated across five task instances. Each cell shows the attributed delta $\hat{\delta}$ with co-selection context (positive = skill helped, negative = skill hurt, empty = skill not selected for that task, **bold** = solo entry):

| | inst 1 | inst 2 | inst 3 | inst 4 | inst 5 |
|---|:---:|:---:|:---:|:---:|:---:|
| **algebra_tricks** | **+0.3** | — | +0.1 (w/ geo) | **+0.4** | — |
| **geometry_basics** | — | **+0.2** | 0.0 (w/ alg) | — | **+0.5** |
| **number_theory** | +0.1 (w/ alg) | +0.3 (w/ geo) | — | **−0.2** | +0.2 (w/ geo) |

Here, algebra_tricks was selected alone on inst 1 and inst 4 (solo, high confidence). On inst 3, both algebra_tricks and geometry_basics were co-selected; the Reflector judged algebra helpful ($\hat{\delta} = +0.1$) and geometry neutral ($\hat{\delta} = 0.0$). Pareto computations prefer the bold solo entries; co-selected entries use the attribution-refined values.

From this matrix:
- **Pareto frequency of algebra_tricks**: Non-dominated on inst 1 (+0.3 solo vs. number_theory's +0.1 co-selected), inst 3 (+0.1 vs. geometry's 0.0), and inst 4 (+0.4 solo vs. number_theory's −0.2 solo). Pareto frequency = 3/3 = 1.0.
- **$\varepsilon$-domination check (algebra_tricks vs. number_theory)**: Shared instances: {inst 1, inst 4} — only 2, below the $n_{\min} = 3$ threshold (condition 1 of §2.4), so domination cannot be established yet. Scores: $A = [+0.3, +0.4]$ vs. $N = [+0.1, -0.2]$. $A \geq N$ on all shared instances and $A > N$ on at least one — conditions 2–3 are satisfied. If a third shared instance confirms the pattern and the skills pass the semantic similarity check (condition 4), algebra_tricks would $\varepsilon$-dominate number_theory, triggering targeted LOO verification (§2.2, Tier 2) on number_theory's replay buffer before retirement. Note: inst 1's comparison mixes confidence levels — algebra's +0.3 is solo (high confidence) while number_theory's +0.1 is co-selected (lower confidence).

---

## 3. The SCULPT Loop

SCULPT runs a five-step loop per task instance: **Retrieve → Execute → Reflect → Curate → Differential Evaluate → Index**. The system maintains a persistent skill library $\mathcal{L}$ where each skill independently accumulates performance data, undergoes lifecycle operations, and evolves over time.

### 3.1 Skill Retrieval

At inference time, the Assembler selects a subset of active skills for a new task instance $x$ through a five-step pipeline:

**Step 1 — Task type filtering.** The incoming instance is classified into a task type $\tau$ (e.g., "algebra," "geometry," "number\_theory"). Only active skills whose `task_types` metadata includes $\tau$ are retained as candidates. This coarse filter narrows the pool to skills designed for the relevant domain, avoiding irrelevant comparisons.

**Step 2 — Semantic similarity scoring.** Each candidate skill is scored by cosine similarity between its description embedding and the task instance embedding:

$$\text{sim}(\sigma, x) = \cos(\text{emb}(\sigma.\text{description}),\ \text{emb}(x))$$

This provides fine-grained, instance-level relevance within the task-type-filtered candidates.

**Step 3 — Pareto boost.** Skills receive a bonus proportional to their Pareto frequency (§2.4):

$$\text{score}(\sigma) = \text{sim}(\sigma, x) + \lambda \cdot f(\sigma)$$

where $\lambda = 0.3$ is the Pareto boost weight. This favors skills with proven track records — skills that are frequently non-dominated on their evaluated instances score higher.

**Step 4 — Exploration bonus.** Unevaluated skills (total evaluations = 0) receive a fixed bonus of 0.6, ensuring newly created skills are selected and tested rather than perpetually ignored in favor of established ones.

**Step 5 — Pool-based selection.** The top-$k$ skill slots are split into **exploit slots** and **explore slots**:

- **Exploit slots** ($k - e$): filled from the highest-scoring skills, prioritizing Pareto-optimal and well-evaluated skills
- **Explore slots** ($e$): filled from skills not in the exploit pool, ensuring under-tested or novel skills receive evaluation

The number of explore slots adapts to library maturity. When total library evaluations are below a maturity threshold ($n_{\text{mature}} = 50$), the system allocates more explore slots ($e = \min(2, \lfloor k/2 \rfloor)$) to accelerate score matrix population. Once the library matures, explore slots reduce to $e = 1$.

**LLM re-ranking (optional).** When the number of candidates after Step 5 exceeds $k$, an LLM selects the most relevant skills from the candidate pool, informed by skill descriptions and Pareto frequencies.

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

The Curator orchestrates six lifecycle operations after each reflection step, guided by Pareto signals and reflector diagnoses.

#### NO-OP — Skip curation when no action is needed

When the selected skills performed well (all attributions are helpful or neutral) and the Reflector identifies no knowledge gaps, the Curator skips all lifecycle operations. This is the expected outcome for the majority of tasks once the library matures — most tasks are handled adequately by existing skills, and unnecessary curation would waste LLM calls without improving the library. The NO-OP decision is made before evaluating any other lifecycle triggers.

#### RETIRE — Remove dominated or consistently harmful skills

A skill is retired under two conditions:

- **Primary (Pareto-based):** The skill is $\varepsilon$-dominated by another active skill with at least $n_{\min}$ shared instances. This means a strictly better alternative exists across all evaluated instances. Before committing, the system runs **targeted LOO verification** (§2.2, Tier 2) on the candidate skill's replay buffer: the skill is only retired if it remains $\varepsilon$-dominated when its score matrix entries are replaced with causal $\delta_{\text{LOO}}$ values, preventing retirement based on spurious co-selection artifacts.
- **Fallback (ratio-based):** When insufficient score matrix data exists for Pareto comparison, the skill is retired if its harmful ratio exceeds 0.7 with at least 5 total evaluations. This handles the cold-start period before score matrices are populated.

Retired skills are marked with `status = "retired"` and excluded from future retrieval.

#### BIRTH — Create new skills from identified gaps

New skills are created when the Reflector identifies knowledge gaps — areas where the current skill library lacks relevant strategies. Two sources of gap signals are combined:

- **Reflector gaps:** The Reflector analyzes execution traces and diagnoses specific missing knowledge (e.g., "no strategy for modular arithmetic with large primes").
- **Coverage gaps:** The system scans all task instance keys across active skills and identifies instances where the best attributed delta among all active skills falls below a coverage threshold ($\theta_{\text{cov}} = 0.3$) — i.e., instances where every skill performed poorly. This detects "blind spots" in the library: regions of the task distribution that no existing skill covers well. For example, if the library has skills for algebra and geometry but keeps scoring below 0.3 on number theory problems, those instances surface as coverage gaps. The threshold $n_{\text{gap}} = 2$ requires at least two such uncovered instances before triggering, avoiding overreaction to a single hard outlier.

Both gap sources are merged and passed to an LLM that decides whether to CREATE a new skill or ENRICH an existing one. When a new skill is created, **targeted LOO verification** (§2.2, Tier 2) confirms its marginal contribution is positive on the replay buffer before it is admitted to the library — preventing context pollution from low-value skills.

#### ENRICH — Modify existing skill content

ENRICH is a unified content-modification operation that ranges from incremental append to full rewrite, governed by the retention factor $\alpha$ (§2.3). It is triggered in two contexts:

**Gap-driven enrichment (append, $\alpha > 0$).** When the LLM determines that an identified gap is best addressed by augmenting an existing skill, new content is appended as a markdown section. A content budget (8,000 characters) prevents unbounded growth:

- If the enriched content stays within budget, it is appended directly.
- If it exceeds the budget, the oldest enrichment section (identified by `## Enrichment` headers) is removed before appending.
- If still over budget after removal, the content is hard-truncated.

The retention factor $\alpha = \cos(\text{emb}(\text{content}\_{\text{old}}), \text{emb}(\text{content}\_{\text{new}}))$ measures semantic drift, and existing score matrix entries are updated via the convex combination in §2.3.

**Quality-driven rewrite ($\alpha \approx 0$).** Periodically, active skills undergo content-level quality assessment via the SkillValidator. Skills whose content has degraded (e.g., through repeated enrichment) or whose quality score falls below threshold are fully rewritten by an LLM while preserving the core strategies. This mode is triggered by the quality audit cycle rather than by task-level outcomes. A full rewrite typically yields $\alpha \approx 0$, effectively discarding old evidence. The skill re-enters cold-start (§3.6), receiving optimistic Pareto inclusion and exploration bonuses until its matrix repopulates. The replay buffer is preserved, providing valid task instances for future evaluation of the rewritten content.

#### SPECIALIZE — Split broadly relevant but inconsistent skills

A skill is a candidate for specialization when it is **both frequently Pareto-optimal and frequently harmful**:

$$f(\sigma) \geq \theta_{\text{pareto}} \quad \text{AND} \quad \frac{n_{\text{harmful}}}{n_{\text{evals}}} \geq \theta_{\text{harmful}}$$

with $\theta_{\text{pareto}} = 0.4$ and $\theta_{\text{harmful}} = 0.25$, requiring at least 5 total evaluations. This trigger identifies skills that contain valuable knowledge for some task types but hurt performance on others — the ideal candidates for splitting.

Before splitting, the system runs **targeted LOO verification** (§2.2, Tier 2) on the parent skill's replay buffer to confirm that its inconsistency persists under causal $\delta_{\text{LOO}}$ scores rather than reflecting noisy co-selection attribution. If confirmed, an LLM receives the parent skill's content and generates two specialized child skills with narrower trigger conditions. The parent is retired, and both children inherit the parent's `skill_id` as their `parent_id` for lineage tracking.

#### GENERALIZE — Merge overlapping skills

When two active skills share sufficient keyword overlap (Jaccard similarity $\geq 0.6$), they are candidates for merging into a single, more comprehensive skill. Before merging, **targeted LOO verification** (§2.2, Tier 2) is run on each parent skill to confirm that neither has independent causal value that would be lost in the merge. If confirmed, an LLM combines the content of both skills, and both originals are retired.

### 3.5 Differential Evaluation for lifecycle operation
#### Offline Setting (Multi-Epoch over Fixed Dataset)

Following GEPA's approach of evaluating candidates on a fixed validation set, SCULPT maintains a **validation split** $D_{\text{val}}$ of the training data. After a lifecycle operation, the system re-evaluates the new version on $D_{\text{val}}$ (or a subsample of size $n_{\text{verify}} = \min(20, |D_{\text{val}}|)$) and compares per-instance scores against the reference version's scores on the same instances. Since the full dataset is available, verification is immediate — no waiting for future tasks.

#### Online Setting (Streaming Tasks)

In the online setting, verification leverages the normal SCULPT loop to avoid redundant evaluations. The key observation is that one of the two scores needed for paired comparison is **already produced by the loop** — only one additional evaluation is required per task.

**On the task that triggers the lifecycle operation:** The loop has already executed the task with the old skill in step ② (Execute), producing $\mu_{\text{old}}$. After curation modifies the skill in step ④, the system runs **one additional evaluation** with the new skill on the same task, producing $\mu_{\text{new}}$. This yields the first paired difference $\delta = \mu_{\text{new}} - \mu_{\text{old}}$ at a cost of one extra evaluation.

**Replay buffer for immediate verification:** To avoid waiting for future tasks, each skill maintains a small **replay buffer** of the last $n_{\text{verify}}$ task instances it was applied to. After a lifecycle operation, the system immediately replays these buffered instances with the new skill, producing $\mu_{\text{new}}$ for each. The old scores $\mu_{\text{old}}$ are already stored in the score matrix from the original execution. This yields $n_{\text{verify}}$ paired differences immediately, enabling an instant accept/rollback decision without waiting for future tasks.

The replay buffer stores only $n_{\text{verify}} = \min(5, |K_\sigma|)$ task instances per skill — a bounded, small memory cost. Buffers are maintained in a FIFO fashion: as the skill is applied to new tasks, the oldest buffered instance is evicted. After verification completes, the buffer continues accumulating instances for potential future lifecycle operations.

For SPECIALIZE, both children enter verification simultaneously, each caching the parent as reference. For GENERALIZE, the merged skill caches the better-scoring parent. If multiple skills undergo lifecycle operations concurrently, each independently maintains its own reference cache, with total cache size bounded by the number of concurrent operations (typically 1–2 per curation step).

### 3.6 Handling Sparsity and Cold Start

The score matrix is very sparse, especially early in optimization. SCULPT uses three mechanisms to handle this gracefully:

1. **Optimistic inclusion in Pareto front.** Skills with no score matrix entries — whether newly created (BIRTH, SPECIALIZE, GENERALIZE) or recently enriched with a full rewrite ($\alpha = 0$; see §2.3) — are included in the Pareto front by default. They cannot be dominated because there is no evidence against them, preventing premature retirement of untested or rewritten skills.

2. **Fallback lifecycle heuristics.** When score matrices have insufficient data for $\varepsilon$-domination checks ($< n_{\min}$ shared instances), the system falls back to aggregate ratio-based heuristics (harmful ratio $> 0.7$) for retirement decisions. As score matrices populate, the Pareto-based mechanism smoothly takes over.

3. **Adaptive exploration.** The Assembler allocates more explore slots when the library is immature (total evaluations $< n_{\text{mature}}$), accelerating score matrix population. Unevaluated skills receive a fixed exploration bonus to ensure they are selected and tested. As the library matures, exploration decreases in favor of exploitation of proven skills.

### 3.7 Multi-Epoch Training

The system supports multi-epoch training over a fixed task set. In each epoch, every task instance is processed through the full five-step loop. Across epochs:

- The score matrix accumulates more entries, making Pareto computations increasingly accurate
- Lifecycle operations progressively retire dominated skills, specialize inconsistent ones, and birth new skills for coverage gaps
- The library converges toward a set of high-quality, non-redundant, specialized skills

Periodic maintenance (every $n_{\text{maint}}$ steps) persists the library and index to disk, and optional validation on a held-out set monitors generalization performance.
