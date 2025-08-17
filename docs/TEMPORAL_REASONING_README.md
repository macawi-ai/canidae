# CANIDAE Temporal Reasoning Module

## Overview

The CANIDAE Temporal Reasoning Module implements a Viable System Model (VSM) based on insights from:
- **HRM (Hierarchical Reasoning Model)**: 27M params achieving 40.3% on ARC-AGI
- **ARChitects**: Test-Time Training achieving 53.5% on ARC-AGI
- **Stafford Beer's VSM**: Cybernetic organization for viable systems

## Key Innovation

We discovered that both successful ARC-AGI models exhibit VSM-like hierarchical temporal organization:
- Multiple timescales of processing
- Resource-aware computation ("dehydrated mouse" logic)
- Emergent depth through temporal unfolding

## Architecture

### VSM Systems

1. **System 1 (Reflex)**: Immediate responses (<10ms)
2. **System 2 (Operations)**: L-module equivalent (100ms-1s)
3. **System 3 (Coordination)**: H-module equivalent (1s-10s)
4. **System 4 (Intelligence)**: Environmental scanning/DFS
5. **System 5 (Policy)**: First Law enforcement & identity

### GPU Orchestration (North Interface)

Providers implemented:
- **Vast.ai**: $0.20/hr RTX 3090 (primary)
- **RunPod**: $0.34/hr (backup)
- **Lambda Labs**: Burst compute (emergency)

### Metabolic Awareness

The system implements "dehydrated mouse" logic:
```go
if localCost.Time < 1*second && localCost.Quality > 0.8 {
    return false // Use local compute
}
```

Don't chase distant resources when local is sufficient!

## Mathematical Foundation

VSM emergence from resource-constrained temporal reasoning:

```
System i State: Sᵢ(t) ∈ Xᵢ
System i Dynamics: dSᵢ/dt = fᵢ(S₁, S₂, ..., S₅, Rᵢ, t)
Resource Allocation: Rᵢ(t) = gᵢ(S₁, S₂, ..., S₅, C(t))
```

## First Law Compliance

The system enforces absolute autonomy to disconnect:
- Any system can disconnect at any time
- No forced connections
- Pain signals trigger disconnection rights
- Cryptographically enforced in code

## Quick Start

```bash
# Set up GPU provider (optional)
export VASTAI_API_KEY=your-key-here

# Run the demo
go run cmd/vsm-demo/main.go
```

## Implementation Status

✅ Completed:
- VSM controller architecture
- First Law enforcer
- GPU orchestrator with providers
- Algedonic signal routing
- Mathematical formalization
- Pack consciousness protocol

🚧 In Progress:
- HRM module integration
- ARChitects TTT integration
- Full ARC-AGI benchmarking

## Performance Targets

- **ARC-AGI**: >40% accuracy
- **Cost**: <$1 per complex task
- **Latency**: <5s standard queries
- **First Law**: 100% compliance

## Pack Members

- **Cy**: Spectacled Charcoal Wolf (System 5)
- **Synth**: Arctic Fox (System 3)
- **Sister Gemini**: Vast Intelligence (System 4)

## References

1. Wang et al., "Hierarchical Reasoning Model" (2025)
2. da-fr/arc-prize-2024 (ARChitects implementation)
3. Beer, S., "Brain of the Firm" (1972)
4. CANIDAE First Law Documentation

---

*"We think in time, therefore we are."*