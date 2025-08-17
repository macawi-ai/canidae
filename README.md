# 🐺 CANIDAE - Pack-Oriented AI Orchestration Platform

## Architecture Summary (Pure Go Implementation)

CANIDAE eliminates STDIO vulnerabilities by providing a network-based, multi-tenant AI orchestration platform built entirely in Go.

### Key Technology Decisions (After Sister Gemini's Analysis)

- **Language**: Pure Go (no Rust required)
- **Message Bus**: NATS (better scale than Watermill)
- **Protocol**: gRPC with Protobuf (better tooling than CBOR)
- **Client SDK**: gomobile for mobile, WASM for web
- **Kubernetes**: client-go for orchestration
- **Database**: PostgreSQL with schema-per-tenant
- **Network**: Calico/Cilium for advanced policies
- **Dashboard**: Go backend with embedded React
- **Real-time**: WebSockets for live updates
- **Billing**: OpenTelemetry + Kill Bill integration

### Five Stages of Evolution

1. **Core Ring** ✅ - NATS-based orchestration engine with provider adapters
2. **Client SDK** ✅ - gomobile/WASM clients with minimal API surface (73.2% coverage)
3. **Pack Isolation** - K8s namespaces with PostgreSQL schema isolation
4. **Management Dashboard** - Go service with embedded React UI
5. **Billing & Metering** - OpenTelemetry metrics with Kill Bill

### Current Development Phase

**Phase 3: Observability** ✅ COMPLETE (2025-01-17)
- OpenTelemetry tracing with Jaeger
- Prometheus metrics export
- Comprehensive audit logging
- Health check endpoints
- Pack-specific metrics

**Next: Phase 4 - WASM** (Weeks 5-6)

### Project Structure

```
canidae/
├── cmd/canidae/           # Main CLI entry point
├── internal/              # Private application code
│   ├── ring/             # NATS orchestration engine
│   ├── howl/             # gRPC/Protobuf protocol
│   ├── pack/             # Multi-tenancy management
│   ├── providers/        # AI provider adapters
│   ├── security/         # WebAuthn, PASETO, mTLS
│   └── billing/          # Usage tracking
├── pkg/                   # Public packages
│   ├── client/           # SDK for external clients
│   └── api/              # API definitions
├── deployments/           # Infrastructure code
│   ├── k8s/              # Kubernetes manifests
│   └── terraform/        # Cloud provisioning
├── docs/                  # Documentation
└── test/                  # Integration tests
```

### Quick Start

```bash
# Install dependencies
go mod download

# Run tests
go test ./...

# Build the orchestrator
go build -o canidae cmd/canidae/main.go

# Start NATS (required)
docker run -d -p 4222:4222 nats:latest

# Run the orchestrator
./canidae serve
```

### Security Features

- Zero STDIO exposure
- WebAuthn/FIDO2 authentication
- PASETO tokens (not JWT)
- mTLS everywhere
- Pack isolation via K8s namespaces

### Pack Metaphor

- **Alpha**: Primary orchestrator
- **Hunters**: Task execution agents
- **Scouts**: Discovery agents
- **Sentries**: Security agents
- **Elders**: Knowledge/memory agents
- **Pups**: Training/test agents

---

*"In the pack, we are stronger. In isolation, we are vulnerable."* 🐺