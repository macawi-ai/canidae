package ring

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
	
	"github.com/canidae/canidae/internal/providers"
	"github.com/canidae/canidae/pkg/types"
	"github.com/nats-io/nats.go"
)

// Ring is the simplified core orchestration engine for CANIDAE
type Ring struct {
	nc          *nats.Conn
	js          nats.JetStreamContext
	registry    *providers.Registry
	mu          sync.RWMutex
	config      Config
	packRoutes  map[string]string // pack ID -> routing rules
}

// Config holds Ring configuration
type Config struct {
	NatsURL      string
	Name         string
	MaxRetries   int
	RetryDelay   time.Duration
}

// NewRing creates a new Ring orchestrator
func NewRing(config Config) (*Ring, error) {
	// Connect to NATS
	nc, err := nats.Connect(config.NatsURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to NATS: %w", err)
	}
	
	// Create JetStream context
	js, err := nc.JetStream()
	if err != nil {
		nc.Close()
		return nil, fmt.Errorf("failed to create JetStream context: %w", err)
	}
	
	ring := &Ring{
		nc:         nc,
		js:         js,
		registry:   providers.NewRegistry(),
		config:     config,
		packRoutes: make(map[string]string),
	}
	
	// Initialize default providers
	if err := ring.initializeProviders(); err != nil {
		log.Printf("Warning: Failed to initialize providers: %v", err)
	}
	
	return ring, nil
}

// initializeProviders sets up mock providers for testing
func (r *Ring) initializeProviders() error {
	// Create mock providers for testing
	mock1 := providers.NewMockProvider("mock-1")
	mock2 := providers.NewMockProvider("mock-2")
	
	if err := r.registry.Register(mock1); err != nil {
		return err
	}
	if err := r.registry.Register(mock2); err != nil {
		return err
	}
	
	log.Printf("Initialized %d mock providers", 2)
	return nil
}

// Process handles an incoming request
func (r *Ring) Process(ctx context.Context, req *types.Request) (*types.Response, error) {
	// Validate request
	if req.ProviderID == "" {
		return nil, fmt.Errorf("provider ID required")
	}
	if req.Prompt == "" {
		return nil, fmt.Errorf("prompt required")
	}
	
	// Get provider from registry
	provider, err := r.registry.Get(req.ProviderID)
	if err != nil {
		return nil, fmt.Errorf("provider not found: %w", err)
	}
	
	// Check provider status
	status := provider.GetStatus()
	if !status.Available {
		return nil, fmt.Errorf("provider %s is unavailable: %s", req.ProviderID, status.Message)
	}
	
	// Create context with timeout
	timeout := req.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	
	// Execute request through provider
	start := time.Now()
	response, err := provider.Execute(execCtx, req)
	if err != nil {
		return nil, fmt.Errorf("provider execution failed: %w", err)
	}
	
	// Update processing time
	response.ProcessingTime = time.Since(start).Seconds()
	
	// Publish response to NATS if pack ID is specified
	if req.PackID != "" {
		subject := fmt.Sprintf("canidae.response.%s", req.PackID)
		if err := r.publishResponse(subject, response); err != nil {
			log.Printf("Failed to publish response to NATS: %v", err)
		}
	}
	
	return response, nil
}

// publishResponse publishes a response to NATS
func (r *Ring) publishResponse(subject string, response *types.Response) error {
	// In a real implementation, we would marshal the response to JSON
	// For now, just log it
	log.Printf("Publishing response to %s: %+v", subject, response)
	return nil
}

// ListProviders returns all available providers
func (r *Ring) ListProviders() []types.Provider {
	return r.registry.List()
}

// GetProvider returns a specific provider
func (r *Ring) GetProvider(id string) (types.Provider, error) {
	return r.registry.Get(id)
}

// RegisterProvider adds a new provider
func (r *Ring) RegisterProvider(provider types.Provider) error {
	return r.registry.Register(provider)
}

// Start begins processing messages from NATS
func (r *Ring) Start(ctx context.Context) error {
	// Subscribe to request subjects
	subscription, err := r.nc.Subscribe("canidae.request.*", func(msg *nats.Msg) {
		// Parse pack ID from subject
		// Process request asynchronously
		go r.handleNatsMessage(ctx, msg)
	})
	if err != nil {
		return fmt.Errorf("failed to subscribe: %w", err)
	}
	
	log.Printf("Ring started, listening on canidae.request.*")
	
	// Wait for context cancellation
	<-ctx.Done()
	
	// Clean shutdown
	subscription.Unsubscribe()
	return nil
}

// handleNatsMessage processes a message from NATS
func (r *Ring) handleNatsMessage(ctx context.Context, msg *nats.Msg) {
	// In a real implementation, we would unmarshal the message
	// and process it through the appropriate provider
	log.Printf("Received message on %s: %d bytes", msg.Subject, len(msg.Data))
	
	// Send acknowledgment
	if err := msg.Ack(); err != nil {
		log.Printf("Failed to acknowledge message: %v", err)
	}
}

// Close cleanly shuts down the Ring
func (r *Ring) Close() error {
	// Close all providers
	for _, provider := range r.registry.List() {
		if err := provider.Close(); err != nil {
			log.Printf("Error closing provider %s: %v", provider.GetID(), err)
		}
	}
	
	// Close NATS connection
	r.nc.Close()
	
	log.Printf("Ring shut down cleanly")
	return nil
}

// HealthCheck verifies the Ring is operational
func (r *Ring) HealthCheck() error {
	if !r.nc.IsConnected() {
		return fmt.Errorf("NATS disconnected")
	}
	
	providers := r.registry.List()
	if len(providers) == 0 {
		return fmt.Errorf("no providers available")
	}
	
	return nil
}