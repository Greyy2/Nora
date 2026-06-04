# Senior SOVA: Expert Dual-Core Cognitive Intelligence

This document details the final, code-dense architecture of SOVA, designed as a 100% autonomous financial agent functioning without external APIs or high-resource hardware.

---

## 1. High-Level Cognitive Architecture (Global Flow)

SOVA is architected as a **Dual-Core Neuro-Matrix**, where logic and creativity work in a continuous, self-optimizing feedback loop.

```mermaid
graph TD
    subgraph "THE LOGICAL BRAIN (ML Core - The Neuron Matrix)"
        L1[Market RAW Data] --> L2[Neural Perception Layer]
        L2 --> L3[Synaptic Vortex Memory]
        L3 --> L4[Structural ML Judge]
    end

    subgraph "THE CREATIVE BRAIN (LLM Core - Reasoning Matrix)"
        L2 --> R1[Strategic Tokenizer]
        R1 --> R2[Recursive CoT Engine]
        R2 --> R3[Recursive Genetic Evolver]
        R3 --> R4[Candidate Alpha]
    end

    R4 --> L4
    L4 -->|Reinforcement| L3
    L4 -->|Advisory| SA[Senior Strategic Analyst]
    SA -->|Strategic Adjustment| R2
```

---

## 2. The Logical Mind: Neuron Matrix & Perception

Designed for **High-Precision Sense** and **Reinforced Memory Retrieval**.

### A. Neuron Perception Layer (Sense)
The "neurons" of SOVA's perception use advanced signal processing to decompose market noise:
- **Kalman Signal Denoising**: Extracting the structural trend from volatile price action.
- **Spectral Entropy**: Measuring the disorder of returns via Welch's power spectrum.
- **Fractal Complexity (Higuchi)**: Quantifying the self-similarity of price paths.
- **Hurst Persistence**: Sensing whether the current regime is trending or mean-reverting.

### B. Synaptic Vortex Memory (REINFORCE)
A bio-inspired sharded knowledge base:
- **Sharded Memory**: Knowledge is sharded by Market Regime (e.g., `CAPITULATION_CRASH`, `EXPONENTIAL_BULL`).
- **Long-Term Potentiation**: Successful alphas reinforce existing synapses, increasing their probability of recall.
- **Selective Forgetting**: Automatic decay of weak or redundant knowledge to prevent over-fitting.

---

## 3. The Creative Mind: Recursive Reasoning & Evolution

Designed for **Unlimited Strategic Innovation**.

### A. Recursive Thinking Core (Reasoning)
SOVA simulates a Transformer-style reasoning engine without an external LLM:
- **Structural Tokenization**: Alpha expressions are tokenized into "genes" (tokens).
- **Next-Gene Prediction**: The reasoning engine deliberates on the most likely "optimal gene" based on current market axioms.
- **Chain-of-Thought (CoT)**: Multi-stage monologue that aligns perception with strategic intent (Momentum vs. Reversion).

### B. Recursive Alpha Evolver (Generate)
Creative alpha synthesis through **Structural Stacking**:
- **Atomic Mutation**: Randomly substituting variables (tokens) to discover hidden correlations.
- **Structural Hybridization**: Cross-breeding successful alphas with strategic intents (e.g., Mean Reversion filters).
- **Syntactic Governance**: Ensures every synthesized alpha is mathematically sound and free of look-ahead bias.

---

## 4. Senior Summary: The Learning Loop

SOVA achieves **Autonomous Intelligence Growth** through its performance feedback loop:

```mermaid
flowchart TD
    A[Alpha Generation] --> B[Backtest Execution]
    B --> C[Performance Metrics - IC/Sharpe]
    C --> D{Learning Layer}
    D -->|Reinforce Memory| E[Synaptic Vortex]
    D -->|Update Judge| F[Structural ML Model]
    D -->|Self-Correct| G[Adjust Mutation Rate]
    G --> A
```

This ensures that every session makes SOVA smarter, more precise, and more specialized in the Chinese A-share market microstructure.
