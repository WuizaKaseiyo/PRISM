"""Seed skills for AIME competition math."""
from __future__ import annotations

from prism import Skill

SEED_SKILLS = [
    Skill(
        name="AIME Problem Strategy",
        description=(
            "Use when solving competition-level math problems that require "
            "creative insight, multi-step reasoning, and an integer answer "
            "between 000 and 999."
        ),
        content="""\
# AIME Problem Strategy

## Overview
The American Invitational Mathematics Examination (AIME) features 15 problems
with integer answers from 000 to 999. Problems require deep mathematical
insight, not just computation.

## Workflow
1. **Read carefully** — identify all given information and what is asked
2. **Classify the domain** — algebra, combinatorics, geometry, or number theory
3. **Look for structure** — symmetry, invariants, modular arithmetic, bijections
4. **Try small cases** — build intuition before generalizing
5. **Choose a method** — direct computation, complementary counting, coordinate
   geometry, generating functions, etc.
6. **Execute cleanly** — show each step, watch for arithmetic errors
7. **Verify** — plug back in, check boundary cases, confirm answer is 000–999
8. **Box the answer** — always present as \\boxed{integer}

## Key Principles
- AIME problems reward cleverness over brute force
- If a computation is getting very messy, reconsider the approach
- Many problems have elegant solutions via parity, modular arithmetic, or symmetry
- Double-check arithmetic — a single sign error can cascade
- Answers are always non-negative integers ≤ 999

## Common Techniques by Domain

### Algebra
- Vieta's formulas for symmetric polynomials
- Substitution to reduce degree
- AM-GM, Cauchy-Schwarz, and other inequalities
- Telescoping sums/products

### Combinatorics
- Stars and bars, PIE (inclusion-exclusion)
- Bijective proofs and complementary counting
- Generating functions for counting sequences
- Recursion with small-case verification

### Geometry
- Coordinate geometry when synthetic is unclear
- Similar triangles, power of a point, radical axes
- Area ratios via determinants or shoelace formula
- Trigonometric identities (law of cosines, extended law of sines)

### Number Theory
- Modular arithmetic and CRT (Chinese Remainder Theorem)
- Euler's totient, Fermat's little theorem
- Prime factorization and divisor counting
- Floor/ceiling function manipulation
""",
        module_tag="general",
        keywords=[
            "aime", "competition", "math", "algebra", "combinatorics",
            "geometry", "number-theory", "integer", "proof", "olympiad",
        ],
        task_types=["math"],
    ),
]
