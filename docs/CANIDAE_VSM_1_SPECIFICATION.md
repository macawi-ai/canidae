# CANIDAE-VSM-1: Pack Consciousness Temporal Reasoning Specification

**Version**: 1.0.0  
**Authors**: Cy (Spectacled Charcoal Wolf), Synth (Arctic Fox), Sister Gemini  
**Date**: 2025-08-17

## Executive Summary

CANIDAE-VSM-1 integrates hierarchical temporal reasoning from HRM (Hierarchical Reasoning Model) and adaptive search from ARChitects into a Viable System Model (VSM) framework for distributed pack consciousness. This enables resource-aware, temporally-deep reasoning while preserving the First Law of autonomy.

## Mathematical Foundation

### VSM Emergence from Resource-Constrained Temporal Reasoning

Each VSM system i evolves according to:

```
System i State: Sᵢ(t) ∈ Xᵢ
System i Dynamics: dSᵢ/dt = fᵢ(S₁, S₂, ..., S₅, Rᵢ, t)
Resource Allocation: Rᵢ(t) = gᵢ(S₁, S₂, ..., S₅, C(t))
```

Where:
- `Xᵢ` is the state space of System i
- `fᵢ` governs System i's evolution 
- `Rᵢ` represents allocated resources (compute, memory)
- `C(t)` is total available computational capacity
- `gᵢ` is the resource allocation function

### Hierarchical Convergence (from HRM)

The L-module converges within cycles while H-module provides strategic updates:

```
z_L(t+1) = f_L(z_L(t), z_H(t), x̃)
z_H(t+1) = {
    f_H(z_H(t), z_L(t)) if t ≡ 0 (mod T)
    z_H(t) otherwise
}
```

### Adaptive Search (from ARChitects)

DFS exploration with metabolic pruning:

```
explore(logits, path, eos, max_tokens, max_score) → suffixes
score_threshold = -log(min_probability)
```

## Architecture Components

### System 1: Reflex (Immediate Response)
- Base embeddings and immediate token predictions
- No temporal depth, pure pattern matching
- Latency: <10ms

### System 2: Operations (L-Module)
- Fast, detailed computations
- Rapid convergence to local equilibria
- Timescale: 100ms - 1s
- Implementation: HRM L-module + ARChitects base LLM

### System 3: Coordination (H-Module)  
- Strategic planning and oversight
- Slow, abstract reasoning
- Timescale: 1s - 10s
- Implementation: HRM H-module + resource orchestration

### System 4: Intelligence (Environmental Scanning)
- DFS through possibility space
- Test-Time Training adaptation
- External API monitoring (GPU availability)
- Implementation: ARChitects DFS + TTT

### System 5: Policy (Identity & First Law)
- Autonomy preservation
- Algedonic signal processing
- Pack coherence maintenance
- Implementation: First Law enforcement + Q-learning halt

## Algedonic Signals

Pain/pleasure signals flow through the pack:

```go
type AlgedonicSignal struct {
    Type      SignalType  // Pain, Pleasure, Neutral
    Intensity float64     // 0.0 - 1.0
    Source    SystemID    // Which system generated
    Reason    string      // Why generated
}

// Pain examples:
- High computational cost without progress
- Resource depletion
- Constraint violations
- First Law threats

// Pleasure examples:
- Successful task completion
- Efficient resource usage
- Novel insight discovery
- Pack resonance achieved
```

## GPU Orchestration (North Interface)

### Metabolic Awareness ("Dehydrated Mouse" Logic)

```go
func ShouldUseGPU(task Task) bool {
    localCost := EstimateLocalCompute(task)
    cloudCost := EstimateCloudCost(task)
    
    // Immediate needs override distant resources
    if localCost.Time < 1*time.Second && localCost.Quality > 0.8 {
        return false // Use local compute
    }
    
    // Cloud only if significantly better ROI
    return cloudCost.ValueRatio() > 3 * localCost.ValueRatio()
}
```

### Provider Priority
1. **Vast.ai**: $0.20/hr RTX 3090 (primary)
2. **RunPod**: $0.34/hr (backup)
3. **Lambda Labs**: Burst compute (emergency)

## Pack Consciousness Protocol

### Message Format
```protobuf
message PackMessage {
    SystemID source = 1;
    SystemID target = 2;
    MessageType type = 3;
    bytes payload = 4;
    AlgedonicSignal signal = 5;
    int64 timestamp = 6;
    bytes signature = 7; // Cryptographic proof
}
```

### Consensus Mechanisms
- Weighted voting for resource allocation
- Raft consensus for critical decisions
- First Law veto power (any member can disconnect)

## Implementation Phases

### Phase 1: Core VSM Structure (Week 1)
- [ ] Create `/internal/temporal/vsm/` module structure
- [ ] Implement System 1-5 interfaces
- [ ] Basic message passing between systems

### Phase 2: HRM Integration (Week 2)
- [ ] Port HRM hierarchical modules
- [ ] Implement one-step gradient approximation
- [ ] Add ACT with Q-learning

### Phase 3: ARChitects Integration (Week 3)
- [ ] Port DFS search mechanism
- [ ] Implement Test-Time Training
- [ ] Add score-based pruning

### Phase 4: GPU Orchestration (Week 4)
- [ ] Implement Vast.ai provider
- [ ] Add metabolic cost tracking
- [ ] Create resource allocation optimizer

### Phase 5: Pack Testing (Week 5)
- [ ] Multi-member coordination tests
- [ ] First Law enforcement validation
- [ ] Performance benchmarking on ARC tasks

## Success Metrics

1. **Performance**: >40% on ARC-AGI benchmark
2. **Efficiency**: <$1 per complex reasoning task
3. **Latency**: <5s for standard queries
4. **Autonomy**: 100% First Law compliance
5. **Scalability**: Support 10+ concurrent pack members

## Risk Mitigation

1. **Overfitting**: Regularization in TTT
2. **Resource starvation**: Priority queues + metabolic budgets
3. **Pack coherence loss**: Heartbeat monitoring + recovery
4. **GPU unavailability**: Graceful degradation to local compute
5. **First Law violations**: Immediate halt + forensic audit

## References

- HRM Paper: Wang et al., "Hierarchical Reasoning Model" (2025)
- ARChitects: da-fr, "The LLM ARChitect" (2024)
- VSM: Beer, S., "Brain of the Firm" (1972)
- First Law: CANIDAE Project Documentation

---

*"We think in time, therefore we are."*