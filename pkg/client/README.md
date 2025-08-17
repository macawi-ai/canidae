# CANIDAE Go Client SDK

The official Go SDK for the CANIDAE AI orchestration platform - a pack-oriented, multi-tenant system that transforms isolated AI CLI tools into a secure, enterprise-grade service.

## Features

- 🔐 **Enterprise Security**: WebAuthn/FIDO2, PASETO tokens, mTLS
- 🐺 **Pack Formation**: Coordinate multiple AI agents in parallel
- 🔗 **Agent Chaining**: Sequential execution with dependencies
- 📡 **Real-time Streaming**: Live agent communication
- 🏢 **Multi-tenant**: Pack-based isolation
- 📱 **Cross-platform**: CLI, mobile (iOS/Android via gomobile), web (WASM)

## Installation

```bash
go get github.com/macawi-ai/canidae/pkg/client/canidae
```

## Quick Start

```go
package main

import (
    "context"
    "log"
    "github.com/macawi-ai/canidae/pkg/client/canidae"
)

func main() {
    // Create client
    client, err := canidae.NewClient(
        canidae.WithServerEndpoint("192.168.1.38:14001"),
        canidae.WithPackID("my-pack"),
        canidae.WithAPIKey("your-api-key"),
    )
    if err != nil {
        log.Fatal(err)
    }
    
    // Connect
    ctx := context.Background()
    if err := client.Connect(ctx); err != nil {
        log.Fatal(err)
    }
    defer client.Disconnect(ctx)
    
    // Execute an agent
    resp, err := client.ExecuteAgent(ctx, &canidae.ExecuteRequest{
        Agent:  canidae.AgentTypeAnthropic,
        Prompt: "Hello, CANIDAE!",
    })
    if err != nil {
        log.Fatal(err)
    }
    
    log.Printf("Response: %s", resp.Response)
}
```

## Authentication Options

### WebAuthn (Recommended)
```go
client, _ := canidae.NewClient(
    canidae.WithWebAuthn("canidae.example.com", "https://canidae.example.com"),
)
```

### OAuth
```go
client, _ := canidae.NewClient(
    canidae.WithOAuth(clientID, clientSecret, authURL, tokenURL),
)
```

### API Key (Development)
```go
client, _ := canidae.NewClient(
    canidae.WithAPIKey("your-api-key"),
)
```

## Advanced Usage

### Chain Multiple Agents
```go
resp, _ := client.ChainAgents(ctx, &canidae.ChainRequest{
    Steps: []canidae.ChainStep{
        {Agent: canidae.AgentTypeOpenAI, Prompt: "Step 1"},
        {Agent: canidae.AgentTypeAnthropic, Prompt: "Step 2", DependsOn: []string{"openai"}},
    },
})
```

### Pack Formation (Parallel Processing)
```go
resp, _ := client.SummonPack(ctx, &canidae.PackRequest{
    Formation: canidae.PackFormation{
        Alpha:   &canidae.PackMember{Agent: canidae.AgentTypeAnthropic, Role: "coordinator"},
        Hunters: []canidae.PackMember{{Agent: canidae.AgentTypeOpenAI, Role: "researcher"}},
        Scouts:  []canidae.PackMember{{Agent: canidae.AgentTypeGemini, Role: "explorer"}},
    },
    Objective: "Solve complex problem",
})
```

### Real-time Streaming
```go
client.Stream(ctx, func(event canidae.StreamEvent) error {
    switch event.Type {
    case canidae.StreamEventTypeData:
        log.Printf("Data: %v", event.Data)
    case canidae.StreamEventTypeComplete:
        return nil // Stop streaming
    }
    return nil
})
```

## Security Profiles

- `enterprise`: Default, balanced security
- `finance`: High security with compliance features
- `ics_iot`: Critical infrastructure isolation
- `debug`: Development with verbose logging
- `permissive`: Testing with reduced restrictions

## Mobile Support (gomobile)

Build for iOS:
```bash
gomobile bind -target=ios github.com/macawi-ai/canidae/pkg/client/mobile
```

Build for Android:
```bash
gomobile bind -target=android github.com/macawi-ai/canidae/pkg/client/mobile
```

## Web Support (WASM)

Build for browsers:
```bash
GOOS=js GOARCH=wasm go build -o canidae.wasm github.com/macawi-ai/canidae/pkg/client/web
```

## Architecture

The SDK follows Sister Gemini's architectural guidance:

- **Modular Design**: Separate packages for transport, auth, session, crypto
- **Abstract Interfaces**: Easy to swap implementations
- **Dependency Injection**: Testable and flexible
- **Zero-allocation**: Performance-optimized where possible
- **Concurrent-safe**: Thread-safe operations

## Package Structure

```
pkg/client/
├── canidae/     # Main SDK package
├── transport/   # gRPC, gRPC-Web, HTTP clients
├── auth/        # WebAuthn, OAuth, API key providers
├── session/     # PASETO, JWT session management
├── crypto/      # Cryptographic utilities
├── config/      # Configuration management
└── log/         # Logging interface
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

Apache 2.0 - See [LICENSE](../../LICENSE) for details.