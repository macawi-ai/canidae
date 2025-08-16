# ADR-0001: Use Go Instead of Rust for CANIDAE Implementation

## Status
Accepted

## Context
The initial CANIDAE design suggested using Rust for the client SDK due to its memory safety guarantees and ability to compile to multiple targets via UniFFI. However, our pack has strong Go expertise and values development velocity.

## Decision
We will implement the entire CANIDAE platform in Go, including:
- Core orchestration engine
- Client SDKs (via gomobile and WASM)
- Backend services
- Provider adapters

## Consequences

### Positive
- Faster development velocity due to team expertise
- Simpler codebase with one language
- Excellent concurrency model for message processing
- Strong ecosystem for backend services
- Good enough security for our use case
- Native Kubernetes integration

### Negative
- Slightly larger mobile SDK size vs Rust
- Garbage collection pauses (mitigated by careful tuning)
- Less fine-grained memory control
- WASM output larger than Rust equivalent

### Mitigation
- Use pprof for performance profiling
- Implement careful resource pooling
- Monitor GC metrics in production
- Consider Rust for specific performance-critical components if needed

## References
- Sister Gemini's analysis: "No mandatory requirements for Rust"
- Go 1.23 improvements in GC and generics
- Success of other Go-based orchestrators (Kubernetes, Docker)