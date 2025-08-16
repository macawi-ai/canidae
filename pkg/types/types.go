// Package types provides common types and interfaces for CANIDAE
package types

import (
	"context"
	"time"
)

// Provider defines the interface that all AI providers must implement
type Provider interface {
	// Execute processes a request through the provider
	Execute(ctx context.Context, req *Request) (*Response, error)
	
	// GetID returns the unique identifier for this provider
	GetID() string
	
	// GetStatus returns the current status of the provider
	GetStatus() ProviderStatus
	
	// Close cleanly shuts down the provider
	Close() error
}

// Request represents a request to an AI provider
type Request struct {
	ID         string                 `json:"id"`
	PackID     string                 `json:"pack_id"`
	ProviderID string                 `json:"provider_id"`
	Prompt     string                 `json:"prompt"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	Priority   Priority               `json:"priority"`
	Timeout    time.Duration          `json:"timeout"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	Timestamp  time.Time              `json:"timestamp"`
}

// Response represents a response from an AI provider
type Response struct {
	ID            string                 `json:"id"`
	RequestID     string                 `json:"request_id"`
	ProviderID    string                 `json:"provider_id"`
	Status        ResponseStatus         `json:"status"`
	Result        string                 `json:"result,omitempty"`
	Error         *Error                 `json:"error,omitempty"`
	Usage         *Usage                 `json:"usage,omitempty"`
	ProcessingTime float64               `json:"processing_time"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	Timestamp     time.Time              `json:"timestamp"`
}

// Error represents an error response
type Error struct {
	Code    string                 `json:"code"`
	Message string                 `json:"message"`
	Details map[string]interface{} `json:"details,omitempty"`
}

// Usage tracks token/resource usage
type Usage struct {
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
	Cost             float64 `json:"cost,omitempty"`
}

// Priority defines request priority levels
type Priority string

const (
	PriorityCritical Priority = "critical"
	PriorityHigh     Priority = "high"
	PriorityMedium   Priority = "medium"
	PriorityLow      Priority = "low"
)

// ResponseStatus defines possible response statuses
type ResponseStatus string

const (
	StatusSuccess     ResponseStatus = "success"
	StatusError       ResponseStatus = "error"
	StatusTimeout     ResponseStatus = "timeout"
	StatusRateLimited ResponseStatus = "rate_limited"
	StatusCancelled   ResponseStatus = "cancelled"
)

// ProviderStatus represents the health status of a provider
type ProviderStatus struct {
	Available    bool      `json:"available"`
	HealthScore  float64   `json:"health_score"` // 0-100
	Latency      float64   `json:"latency"`      // milliseconds
	ErrorRate    float64   `json:"error_rate"`   // percentage
	LastCheck    time.Time `json:"last_check"`
	Message      string    `json:"message,omitempty"`
}

// ProviderConfig holds provider configuration
type ProviderConfig struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Type        ProviderType           `json:"type"`
	APIKey      string                 `json:"api_key,omitempty"`
	Endpoint    string                 `json:"endpoint,omitempty"`
	RateLimit   int                    `json:"rate_limit"`   // requests per minute
	Timeout     time.Duration          `json:"timeout"`
	MaxRetries  int                    `json:"max_retries"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

// ProviderType defines types of AI providers
type ProviderType string

const (
	ProviderTypeLLM              ProviderType = "LLM"
	ProviderTypeImageGeneration  ProviderType = "ImageGeneration"
	ProviderTypeImageAnalysis    ProviderType = "ImageAnalysis"
	ProviderTypeSpeech           ProviderType = "Speech"
	ProviderTypeTranslation      ProviderType = "Translation"
)

// FlowControlEvent represents flow control system events
type FlowControlEvent struct {
	Type       FlowControlEventType `json:"type"`
	ProviderID string               `json:"provider_id"`
	PackID     string               `json:"pack_id,omitempty"`
	Details    map[string]interface{} `json:"details,omitempty"`
	Timestamp  time.Time            `json:"timestamp"`
}

// FlowControlEventType defines flow control event types
type FlowControlEventType string

const (
	EventRateLimitTriggered FlowControlEventType = "rate_limit_triggered"
	EventCircuitBreakerOpen FlowControlEventType = "circuit_breaker_open"
	EventCircuitBreakerClose FlowControlEventType = "circuit_breaker_close"
	EventBackpressureApplied FlowControlEventType = "backpressure_applied"
)

// Registry manages provider registration and lookup
type Registry interface {
	Register(provider Provider) error
	Get(id string) (Provider, error)
	List() []Provider
	Remove(id string) error
}