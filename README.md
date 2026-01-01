# SPX-regime-risk-model

## Overview

This repository documents a **research-grade SPX risk engine** designed to produce **market regime state**, **conditional volatility**, and **tail-risk diagnostics** using only information available at the decision time.

The project is intentionally **risk-first and execution-agnostic**.  
Downstream decision rules, capital allocation, and trading actions are **out of scope**.

## Quickstart

To run the public snapshot locally:

pip install -r requirements.txt
python "SPX risk snapshot (public).py

---

## What This Project Is

- A standalone **risk diagnostics engine** for the S&P 500 index
- Focused on **probability and distribution**, not price direction
- Built with **leakage-safe**, walk-forward principles
- Designed to be **auditable, conservative, and failure-aware**

---

## What This Project Is Not

This repository intentionally does **not** include:

- Trading rules or execution logic  
- Option structures, strikes, or premium evaluation  
- Policy thresholds, cooldown logic, or capital sizing  
- Backtests or performance claims  

The goal is to demonstrate **risk estimation architecture**, not a deployable trading system.

---

## Methodology (High Level)

- **Leakage-safe realized volatility features**  
  Realized volatility inputs are shifted to avoid look-ahead bias.

- **Two-state regime detection (CALM / STRESSED)**  
  Regimes are inferred from volatility level, persistence, and slope behavior.

- **Conditional volatility modeling**  
  EGARCH and GJR-GARCH models with Student-t innovations are evaluated, with conservative selection based on downside tail behavior.

- **Distributional tail modeling**  
  A hybrid empirical + EVT/GPD approach is used to characterize extreme tail risk, with stability checks and safe fallbacks.

- **Diagnostic VaR ladder**  
  Produces a probability-indexed view of next-day downside risk for scenario analysis and monitoring.

---

## Outputs (Conceptual)

A typical daily snapshot includes:

- Regime label (CALM / STRESSED)
- Regime confidence score
- One-day-ahead conditional volatility estimate
- Diagnostic left-tail VaR levels across multiple probabilities
- Shock flag for abnormal return behavior

These outputs are **diagnostic only** and do not constitute trading signals.

---

## Walk-Forward Discipline

All features and estimates are computed using **only historical data available at the time of the snapshot**.  
No future information is used in regime detection, volatility estimation, or tail modeling.

Operational policy governance (e.g., when to act on risk estimates) is intentionally excluded.

---

## Governance Note

This repository represents a **policy-sanitized research implementation**.  
Live deployment parameters, downstream decision rules, and execution constraints are intentionally omitted to prevent overfitting and misuse.

---

## Intended Use

- Risk research and education  
- Architecture discussion in interviews  
- Demonstrating probabilistic thinking and tail-risk awareness  
- Illustrating separation between risk estimation and decision-making  

---

## License

MIT License.
