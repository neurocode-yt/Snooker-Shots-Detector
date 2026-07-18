# Benchmark report template

**Dataset:** _name / date_  
**System version:** 0.1.0-phase1  
**Config hash / notes:**  
**Hardware:** CPU / GPU  

> Treat production targets as **goals**, not claims. Fill measured values only.

## Summary metrics

| Metric | Target | Measured |
|--------|--------|----------|
| Shot-event precision | ≥ 0.97 | |
| Shot-event recall | ≥ 0.98 | |
| F1 | — | |
| Missed-shot rate | ≤ 0.02 | |
| False-shot rate | ≤ 0.03 | |
| Median cue-strike error (s) | ≤ 0.25 | |
| Median ball-stop error (s) | ≤ 0.50 | |
| Replay false-positive rate | — | |
| Replay duplicate rate | ≤ 0.01 | |
| Useful action removed | 0 reviewed | |
| Incorrect segments / 100 shots | ≤ 1 | |
| Audio sync error | < 1 frame | |
| Processing speed (× realtime) | — | |
| Peak RAM | — | |

## Per-video table

| Video | #GT | #Pred | P | R | med Δstrike | med Δend | Notes |
|-------|-----|-------|---|---|-------------|----------|-------|
| | | | | | | | |

## Failure analysis

- Missed pots / safeties:
- False strikes (camera / player / replay):
- Boundary errors:

## Command

```bash
snooker-ai evaluate ./test-dataset --output benchmark_report.json
```

## Honest Phase 1 statement

Phase 1 is a **rule-based baseline**. Accuracy on full broadcast matches has **not**
been certified in this repository. Measure on your target broadcasters before
claiming production readiness.
