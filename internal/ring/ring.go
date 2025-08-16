package ring

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/canidae/canidae/internal/chaos"
	"github.com/canidae/canidae/internal/providers"
	"github.com/nats-io/nats.go"
)

// Ring is the core orchestration engine for CANIDAE
type Ring struct {
	nc           *nats.Conn
	js           nats.JetStreamContext
	registry     *providers.Registry
	flowControl  *FlowController
	chaosMonkey  *chaos.ChaosMonkey
	mu           sync.RWMutex
	config       Config
	packRoutes   map[string]string // pack ID -> routing rules
	providers    map[string]providers.Provider // provider instances
}

// Config holds Ring configuration
type Config struct {
	NatsConn   *nats.Conn
	ConfigPath string
}

// Provider interface for AI providers
type Provider interface {
	Name() string
	Execute(ctx context.Context, req Request) (Response, error)
	HealthCheck(ctx context.Context) error
}

// Request represents an AI request
type Request struct {
	ID        string                 `json:"id"`
	PackID    string                 `json:"pack_id"`
	Provider  string                 `json:"provider"`
	Model     string                 `json:"model"`
	Prompt    string                 `json:"prompt"`
	Options   map[string]interface{} `json:"options,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
}

// Response represents an AI response
type Response struct {
	ID        string    `json:"id"`
	RequestID string    `json:"request_id"`
	Content   string    `json:"content"`
	Usage     Usage     `json:"usage"`
	Error     string    `json:"error,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// Usage tracks resource consumption
type Usage struct {
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	Cost             float64 `json:"cost"`
}

// New creates a new Ring instance
func New(cfg Config) (*Ring, error) {
	// Create JetStream context for durable messaging
	js, err := cfg.NatsConn.JetStream()
	if err != nil {
		return nil, fmt.Errorf("failed to create JetStream context: %w", err)
	}

	// Create or update the stream for requests
	streamCfg := &nats.StreamConfig{
		Name:      "CANIDAE_REQUESTS",
		Subjects:  []string{"canidae.request.>"},
		Retention: nats.WorkQueuePolicy,
		MaxAge:    24 * time.Hour,
		Storage:   nats.FileStorage,
	}
	
	if _, err := js.AddStream(streamCfg); err != nil {
		// Stream might already exist, try updating
		if _, err := js.UpdateStream(streamCfg); err != nil {
			log.Printf("Warning: Could not create/update stream: %v", err)
		}
	}

	r := &Ring{
		nc:         cfg.NatsConn,
		js:         js,
		providers:  make(map[string]Provider),
		config:     cfg,
		packRoutes: make(map[string]string),
	}

	// Initialize providers (will be loaded dynamically later)
	if err := r.initProviders(); err != nil {
		return nil, fmt.Errorf("failed to initialize providers: %w", err)
	}

	return r, nil
}

// Start begins processing requests
func (r *Ring) Start(ctx context.Context) error {
	log.Println("Ring orchestration engine starting...")

	// Subscribe to request queue
	sub, err := r.js.QueueSubscribe(
		"canidae.request.*",
		"ring-workers",
		r.handleRequest,
		nats.Durable("ring-worker"),
		nats.ManualAck(),
	)
	if err != nil {
		return fmt.Errorf("failed to subscribe to requests: %w", err)
	}
	defer sub.Unsubscribe()

	// Start health check routine
	go r.healthCheckLoop(ctx)

	log.Println("Ring is active and processing requests")
	
	// Wait for context cancellation
	<-ctx.Done()
	
	log.Println("Ring shutting down...")
	return nil
}

// handleRequest processes incoming AI requests
func (r *Ring) handleRequest(msg *nats.Msg) {
	var req Request
	if err := json.Unmarshal(msg.Data, &req); err != nil {
		log.Printf("Failed to unmarshal request: %v", err)
		msg.Nak()
		return
	}

	log.Printf("Processing request %s from pack %s to provider %s", 
		req.ID, req.PackID, req.Provider)

	// Get the provider
	r.mu.RLock()
	provider, exists := r.providers[req.Provider]
	r.mu.RUnlock()

	if !exists {
		r.sendError(req, fmt.Sprintf("provider %s not found", req.Provider))
		msg.Ack()
		return
	}

	// Execute the request
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := provider.Execute(ctx, req)
	if err != nil {
		r.sendError(req, err.Error())
		msg.Ack()
		return
	}

	// Publish response
	respData, _ := json.Marshal(resp)
	if err := r.nc.Publish(fmt.Sprintf("canidae.response.%s", req.PackID), respData); err != nil {
		log.Printf("Failed to publish response: %v", err)
	}

	msg.Ack()
}

// sendError sends an error response
func (r *Ring) sendError(req Request, errMsg string) {
	resp := Response{
		ID:        generateID(),
		RequestID: req.ID,
		Error:     errMsg,
		Timestamp: time.Now(),
	}
	
	respData, _ := json.Marshal(resp)
	r.nc.Publish(fmt.Sprintf("canidae.response.%s", req.PackID), respData)
}

// initProviders initializes available providers
func (r *Ring) initProviders() error {
	// TODO: Dynamically load providers from plugins or configuration
	// For now, we'll add them manually in the next iteration
	log.Println("Provider initialization complete (0 providers loaded)")
	return nil
}

// healthCheckLoop monitors provider health
func (r *Ring) healthCheckLoop(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			r.mu.RLock()
			providers := make(map[string]Provider)
			for k, v := range r.providers {
				providers[k] = v
			}
			r.mu.RUnlock()

			for name, provider := range providers {
				checkCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
				if err := provider.HealthCheck(checkCtx); err != nil {
					log.Printf("Provider %s health check failed: %v", name, err)
				}
				cancel()
			}
		}
	}
}

// generateID creates a unique identifier
func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}