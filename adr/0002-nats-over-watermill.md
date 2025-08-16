# ADR-0002: Use NATS with JetStream Instead of Watermill

## Status
Accepted

## Context
We need a robust message bus for the CANIDAE orchestration engine. Initial plan suggested Watermill (Go-native pub/sub abstraction), but we need to consider scale and features.

## Decision
Use NATS with JetStream directly instead of Watermill abstraction layer.

## Consequences

### Positive
- Better performance at scale (per Sister Gemini's recommendation)
- Native support for:
  - Durable message streams
  - Work queue distribution
  - Multiple consumer patterns
  - Built-in rate limiting
  - Subject-based routing
- Strong clustering and HA support
- Excellent observability
- Direct control over message flow

### Negative
- Tighter coupling to NATS (harder to switch message buses)
- Steeper learning curve for NATS-specific features
- Need to manage JetStream streams and consumers

### Mitigation
- Create abstraction layer for core messaging operations
- Document NATS-specific patterns thoroughly
- Implement comprehensive integration tests

## References
- NATS JetStream documentation
- Sister Gemini's recommendation for scale
- Benchmark: NATS handles 18M+ messages/second