package providers

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Provider defines the interface all AI providers must implement
type Provider interface {
	// Name returns the provider identifier
	Name() string
	
	// Execute sends a request to the AI provider
	Execute(ctx context.Context, req Request) (*Response, error)
	
	// HealthCheck verifies the provider is responsive
	HealthCheck(ctx context.Context) error
	
	// GetCapabilities returns what this provider can do
	GetCapabilities() Capabilities
	
	// EstimateCost calculates the expected cost for a request
	EstimateCost(req Request) Cost
}

// Request represents a unified AI request
type Request struct {
	ID        string                 `json:"id"`
	PackID    string                 `json:"pack_id"`
	Role      PackRole              `json:"role"`
	Model     string                 `json:"model"`
	Messages  []Message             `json:"messages"`
	Options   Options               `json:"options"`
	Priority  Priority              `json:"priority"`
	Timestamp time.Time             `json:"timestamp"`
}

// Message represents a conversation message
type Message struct {
	Role    string `json:"role"`    // system, user, assistant
	Content string `json:"content"`
}

// Response represents a unified AI response
type Response struct {
	ID        string    `json:"id"`
	RequestID string    `json:"request_id"`
	Provider  string    `json:"provider"`
	Model     string    `json:"model"`
	Content   string    `json:"content"`
	Usage     Usage     `json:"usage"`
	Metadata  Metadata  `json:"metadata"`
	Error     string    `json:"error,omitempty"`
	Timestamp time.Time `json:"timestamp"`
}

// Usage tracks token consumption
type Usage struct {
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	TotalTokens      int     `json:"total_tokens"`
	Cost             float64 `json:"cost"`
	Currency         string  `json:"currency"`
}

// Cost represents estimated costs
type Cost struct {
	Estimated float64 `json:"estimated"`
	Minimum   float64 `json:"minimum"`
	Maximum   float64 `json:"maximum"`
	Currency  string  `json:"currency"`
}

// Options for request customization
type Options struct {
	Temperature      float64           `json:"temperature,omitempty"`
	MaxTokens        int               `json:"max_tokens,omitempty"`
	TopP             float64           `json:"top_p,omitempty"`
	Stream           bool              `json:"stream,omitempty"`
	StopSequences    []string          `json:"stop_sequences,omitempty"`
	PresencePenalty  float64           `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64           `json:"frequency_penalty,omitempty"`
	Custom           map[string]interface{} `json:"custom,omitempty"`
}

// Capabilities describes what a provider can do
type Capabilities struct {
	Models           []ModelInfo `json:"models"`
	MaxTokens        int         `json:"max_tokens"`
	SupportsStream   bool        `json:"supports_stream"`
	SupportsFunction bool        `json:"supports_function"`
	SupportsVision   bool        `json:"supports_vision"`
	SupportsAudio    bool        `json:"supports_audio"`
}

// ModelInfo describes a specific model
type ModelInfo struct {
	ID           string  `json:"id"`
	Name         string  `json:"name"`
	ContextSize  int     `json:"context_size"`
	CostPerToken float64 `json:"cost_per_token"`
	Deprecated   bool    `json:"deprecated"`
}

// Metadata for additional response information
type Metadata struct {
	ProcessingTime time.Duration          `json:"processing_time"`
	Region         string                 `json:"region,omitempty"`
	Version        string                 `json:"version,omitempty"`
	Extra          map[string]interface{} `json:"extra,omitempty"`
}

// PackRole represents the role in the pack
type PackRole string

const (
	PackRoleAlpha   PackRole = "alpha"
	PackRoleHunter  PackRole = "hunter"
	PackRoleScout   PackRole = "scout"
	PackRoleSentry  PackRole = "sentry"
	PackRoleElder   PackRole = "elder"
	PackRolePup     PackRole = "pup"
)

// Priority levels for requests
type Priority int

const (
	PriorityCritical Priority = iota
	PriorityHigh
	PriorityMedium
	PriorityLow
)

// Registry manages available providers
type Registry struct {
	providers map[string]Provider
	factories map[string]Factory
	mu        sync.RWMutex
}

// Factory creates provider instances
type Factory func(config map[string]interface{}) (Provider, error)

// NewRegistry creates a provider registry
func NewRegistry() *Registry {
	return &Registry{
		providers: make(map[string]Provider),
		factories: make(map[string]Factory),
	}
}

// Register adds a provider factory
func (r *Registry) Register(name string, factory Factory) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.factories[name]; exists {
		return fmt.Errorf("provider %s already registered", name)
	}
	
	r.factories[name] = factory
	return nil
}

// Create instantiates a provider
func (r *Registry) Create(name string, config map[string]interface{}) (Provider, error) {
	r.mu.RLock()
	factory, exists := r.factories[name]
	r.mu.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("provider %s not found", name)
	}
	
	provider, err := factory(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create provider %s: %w", name, err)
	}
	
	r.mu.Lock()
	r.providers[name] = provider
	r.mu.Unlock()
	
	return provider, nil
}

// Get retrieves an active provider
func (r *Registry) Get(name string) (Provider, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	provider, exists := r.providers[name]
	return provider, exists
}

// List returns all registered provider names
func (r *Registry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	names := make([]string, 0, len(r.factories))
	for name := range r.factories {
		names = append(names, name)
	}
	
	return names
}

// BaseProvider provides common functionality
type BaseProvider struct {
	name         string
	capabilities Capabilities
	config       map[string]interface{}
}

// Name returns provider name
func (b *BaseProvider) Name() string {
	return b.name
}

// GetCapabilities returns provider capabilities
func (b *BaseProvider) GetCapabilities() Capabilities {
	return b.capabilities
}