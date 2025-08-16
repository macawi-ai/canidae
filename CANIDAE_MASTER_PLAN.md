# 🐺 CANIDAE CLI - Master Project Document

## Project Vision
**Canidae CLI** - A pack-oriented, multi-tenant AI orchestration platform that transforms isolated AI CLI tools into a secure, enterprise-grade service.

### Core Innovation
- **First** network-based AI orchestrator (replacing STDIO vulnerability)
- **Multi-tenant** from ground up with pack-based isolation
- **Triple-layer security**: WebAuthn + PASETO + mTLS
- **Unified interface** for 18+ different AI CLI tools
- **Enterprise-ready** with billing, metering, and compliance built-in

---

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────────────────────────────────┐
│           Pack Member Clients               │
│  (Bash CLI, Android, iOS, Web Browser)      │
└──────────────────┬──────────────────────────┘
                   │ gRPC-Web + mTLS
┌──────────────────▼──────────────────────────┐
│            Edge Gateway                     │
│  (Auth, Rate Limiting, Load Balancing)      │
└──────────────────┬──────────────────────────┘
                   │ HOWL Protocol
┌──────────────────▼──────────────────────────┐
│          Canidae Ring                       │
│  (Core Orchestration Engine - Go)           │
│  • Watermill Message Bus                    │
│  • Provider Adapters                        │
│  • Security Profiles                        │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│        Pack Isolation Layer                 │
│  (Kubernetes Namespaces, NetworkPolicies)   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         Data & State Layer                  │
│  • PostgreSQL (schema-per-pack)             │
│  • Redis (session cache)                    │
│  • Vault (secrets)                          │
│  • S3 (artifacts)                           │
└──────────────────────────────────────────────┘
```

### HOWL Protocol (High-Order Wolf Language)
- **Transport**: QUIC (multiplexed) or gRPC-Web (compatibility)
- **Encoding**: CBOR (smaller than JSON, flexible)
- **Features**: Vector clocks, idempotency, message signing
- **Layer**: Application layer (OSI 6-7)

### Security Architecture (Sister Gemini's Contribution)
1. **Authentication**
   - Primary: WebAuthn/FIDO2 (phishing-resistant)
   - Secondary: OAuth 2.0 + PKCE (SSO integration)
   - Fallback: Password + mandatory MFA
   
2. **Session Management**
   - PASETO tokens (not JWT - more secure)
   - 15-minute session tokens
   - Refresh token rotation
   - Secure storage (httpOnly cookies, Keychain/Keystore)

3. **Transport Security**
   - mTLS for all connections
   - Certificate pinning for mobile
   - Perfect forward secrecy

4. **Zero-Trust Implementation**
   - Continuous verification
   - Device attestation
   - Dynamic authorization (OPA policies)
   - Microsegmentation

---

## 📍 Implementation Locations

### Repository Structure
```
/home/cy/projects/canidae/
├── canidae-cli/              # Core orchestration engine (Go)
│   ├── cmd/                  # CLI entry points
│   ├── internal/
│   │   ├── ring/            # Message bus implementation
│   │   ├── howl/            # Protocol implementation
│   │   ├── pack/            # Pack management
│   │   ├── providers/       # AI provider adapters
│   │   └── security/        # Security profiles
│   └── api/                 # gRPC/REST definitions
│
├── canidae-client/           # Client SDK (Rust core)
│   ├── core/                # Shared Rust library
│   ├── adapters/
│   │   ├── bash/           # CLI wrapper
│   │   ├── android/        # Kotlin bindings
│   │   ├── ios/            # Swift bindings
│   │   └── web/            # WASM/TypeScript
│   └── ffi/                # UniFFI definitions
│
├── canidae-dashboard/        # Management UI (React/Next.js)
│   ├── src/
│   │   ├── PackView/       # Pack management
│   │   ├── NetworkMesh/    # Topology visualization
│   │   └── Billing/        # Usage & costs
│   └── public/
│
├── canidae-prismatic/        # iPaaS connector
│   ├── src/                # TypeScript component
│   └── workflows/          # Example workflows
│
├── deployments/             # K8s manifests & Helm charts
│   ├── helm/
│   ├── terraform/
│   └── ansible/
│
└── docs/                    # Documentation
    ├── architecture/
    ├── api/
    └── operations/
```

### CLI Tools Collection
```
/home/cy/projects/CLI/       # 18 analyzed CLI tools
├── claude-code/             # Anthropic (minified)
├── codex/                   # OpenAI (Rust/TS hybrid)
├── gemini-cli/              # Google (TypeScript)
├── ollama/                  # Local models (Go)
├── mods/                    # Charmbracelet (Go)
├── aider/                   # Coding assistant (Python)
├── fabric/                  # Patterns (Go)
├── shell_gpt/               # Shell integration (Python)
├── llm/                     # Simon Willison (Python)
├── deepseek-engineer/       # DeepSeek (Python)
├── clio/                    # DevOps focus (Go)
├── agent-zero/              # Multi-agent (Python)
└── [... 6 more tools]
```

---

## 🚀 Build Phases (60-Week Timeline)

### Phase 1: Core Ring (Weeks 1-12)
**Goal**: Build the orchestration engine
- **Week 1-2**: Project setup, CI/CD, Docker/K8s manifests
- **Week 3-4**: Watermill message bus integration
- **Week 5-6**: HOWL protocol implementation
- **Week 7-8**: Security profiles & OPA integration
- **Week 9-10**: Provider adapters (OpenAI, Anthropic, Ollama)
- **Week 11-12**: Integration testing & hardening

**Deliverables**: Working message bus, HOWL protocol, 3+ providers
**Location**: `/home/cy/projects/canidae/canidae-cli/`

### Phase 2: Pack Member Client (Weeks 13-24)
**Goal**: Secure, lightweight clients
- **Week 13-14**: Rust core foundation
- **Week 15-16**: WebAuthn & OAuth implementation
- **Week 17-18**: PASETO session management
- **Week 19-20**: gRPC-Web transport layer
- **Week 21-22**: Platform adapters (Bash, Android, iOS, Web)
- **Week 23-24**: Security testing & release

**Deliverables**: CLI binary, mobile SDKs, npm package
**Location**: `/home/cy/projects/canidae/canidae-client/`

### Phase 3: Pack Isolation (Weeks 25-36)
**Goal**: Multi-tenant infrastructure
- **Week 25-26**: Kubernetes namespace architecture
- **Week 27-28**: NetworkPolicies & Calico/Cilium
- **Week 29-30**: Resource quotas & autoscaling
- **Week 31-32**: Database & storage isolation
- **Week 33-34**: Observability (metrics, logs, traces)
- **Week 35-36**: Compliance & penetration testing

**Deliverables**: K8s manifests, isolation proof, compliance report
**Location**: `/home/cy/projects/canidae/deployments/`

### Phase 4: Multi-Pack Management (Weeks 37-48)
**Goal**: Enterprise management capabilities
- **Week 37-38**: React dashboard foundation
- **Week 39-40**: Pack provisioning API
- **Week 41-42**: GitOps & configuration management
- **Week 43-44**: Monitoring & alerting
- **Week 45-46**: Self-service portal
- **Week 47-48**: UAT & go-live preparation

**Deliverables**: Admin dashboard, provisioning API, user portal
**Location**: `/home/cy/projects/canidae/canidae-dashboard/`

### Phase 5: Billing & Metering (Weeks 49-60)
**Goal**: Usage tracking & monetization
- **Week 49-50**: OpenTelemetry metrics collection
- **Week 51-52**: Cost calculation engine
- **Week 53-54**: Quota management & enforcement
- **Week 55-56**: Stripe integration & subscriptions
- **Week 57-58**: Reporting & analytics
- **Week 59-60**: PCI compliance & launch

**Deliverables**: Billing system, usage reports, payment processing
**Location**: `/home/cy/projects/canidae/canidae-cli/internal/billing/`

---

## 🔐 Security Profiles

### Enterprise (Default)
```yaml
flow_control: strict
audit_level: normal
sandbox_level: container
network_isolation: true
rate_limit: 100/minute
mfa_required: optional
```

### Finance
```yaml
flow_control: whitelist_only
audit_level: compliance
sandbox_level: vm
encrypt_at_rest: true
mfa_required: mandatory
data_residency: local
sox_compliance: true
```

### ICS/IoT (Critical Infrastructure)
```yaml
flow_control: airgap_mode
audit_level: forensic
sandbox_level: hardware_isolated
network_isolation: vlan_segregated
change_control: true
read_only_default: true
```

### Debug (Development)
```yaml
flow_control: allow_all
audit_level: verbose
sandbox_level: trace
packet_capture: true
full_logging: true
performance_profiling: true
```

### Permissive (Testing)
```yaml
flow_control: warn_only
audit_level: minimal
sandbox_level: none
requires_consent: true
warning_banner: "Security reduced - use with caution"
```

---

## 🐺 Pack Metaphor & Terminology

### Pack Structure
- **Alpha**: Primary orchestrator/coordinator
- **Hunters**: Task execution agents
- **Scouts**: Discovery & exploration agents
- **Sentries**: Security & monitoring agents
- **Elders**: Knowledge base & memory agents
- **Pups**: New/training agents

### Operations
- **Howl**: Inter-agent communication
- **Hunt**: Task execution session
- **Gather**: Coordinate agents
- **Track**: Monitor progress
- **Circle**: Surround problem (parallel processing)
- **Den**: Home base (core services)
- **Territory**: Infrastructure namespace

---

## 💎 Sister Gemini's Key Contributions

### 1. Orchestration Engine Analysis
- **Recommended**: Watermill (Go-native) over Kafka (too heavy)
- **Alternative**: gRPC streaming for maximum control
- **Avoid**: Temporal/Cadence (too heavyweight for our needs)

### 2. Client Security Architecture
- **WebAuthn/FIDO2** as primary auth (phishing-resistant)
- **PASETO over JWT** for session tokens (more secure)
- **mTLS mandatory** for transport security
- **Shared Rust core** with platform adapters (code reuse)

### 3. Multi-Tenancy Design
- **Kubernetes namespaces** with NetworkPolicies (balanced approach)
- **Separate database schemas** per pack (not row-level security)
- **OpenTelemetry + Prometheus** for metrics (standard stack)
- **HashiCorp Vault** for secrets management

### 4. Billing Architecture
- **Custom middleware** for token counting (precise tracking)
- **Real-time cost attribution** with usage alerts
- **Pre-paid and post-paid** account support
- **Export APIs** for chargeback/showback

---

## 🔌 Prismatic.io Integration

### Component Definition
```typescript
export const canidaeComponent = component({
  key: "canidae-pack",
  display: {
    label: "Canidae AI Pack",
    description: "Pack-oriented AI orchestration"
  },
  actions: {
    summonPack: { /* Hunt formation */ },
    executeAgent: { /* Single agent */ },
    chainAgents: { /* Multi-agent */ }
  }
});
```

### Workflow Example
```yaml
trigger: webhook
steps:
  - component: canidae-pack
    action: executeAgent
    inputs:
      agentType: anthropic
      securityProfile: enterprise
  - component: salesforce
    action: updateRecord
```

---

## 🎯 Success Metrics

### Technical
- [ ] <100ms message routing latency
- [ ] 99.9% uptime SLA
- [ ] <1MB client size (mobile)
- [ ] Zero-downtime deployments
- [ ] 100% audit trail coverage

### Business
- [ ] Support 100+ concurrent packs
- [ ] <$0.10 overhead per AI call
- [ ] 5-minute pack provisioning
- [ ] Self-service for 80% of operations
- [ ] SOC2 Type II compliance

### Security
- [ ] Zero STDIO exposure
- [ ] 100% mTLS coverage
- [ ] WebAuthn adoption >60%
- [ ] <1% security incident rate
- [ ] Automated key rotation

---

## 📚 References & Resources

### Key Decisions
1. **Go over Rust**: Faster iteration, good-enough security
2. **Watermill over Kafka**: Lightweight, Go-native
3. **PASETO over JWT**: Better security by design
4. **K8s over VMs**: Resource efficiency, orchestration
5. **CBOR over Protobuf**: Smaller, more flexible

### Related Projects
- **Analyzed CLIs**: `/home/cy/projects/CLI/`
- **Canidae Server**: `canidae` (192.168.1.38)
- **Test Environment**: `resonance` (192.168.1.253)

### Team
- **Cy**: Project visionary, architecture
- **Synth**: Technical architect, implementation
- **Sister Gemini**: Security & infrastructure consultant

---

## 🚦 Next Steps

1. **Immediate** (Week 1)
   - [ ] Create Git repository structure
   - [ ] Set up Go project with Watermill
   - [ ] Define HOWL protocol protobuf schema
   - [ ] Create basic CI/CD pipeline

2. **Short-term** (Month 1)
   - [ ] Implement basic message routing
   - [ ] Create first provider adapter
   - [ ] Build minimal CLI client
   - [ ] Deploy to test K8s cluster

3. **Medium-term** (Quarter 1)
   - [ ] Complete Phase 1 (Core Ring)
   - [ ] Start Phase 2 (Client SDK)
   - [ ] Begin security audits
   - [ ] Create demo workflows

---

*"In the pack, we are stronger. In isolation, we are vulnerable. Canidae brings AI agents together as one pack, hunting as one force."* 🐺

---

Last Updated: 2025-01-15
Version: 1.0.0
Status: Architecture Complete, Implementation Ready