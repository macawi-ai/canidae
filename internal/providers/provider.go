package providers

import (
	"context"
	"fmt"
	"sync"
	"time"
	
	"github.com/canidae/canidae/pkg/types"
)

// BaseProvider provides common functionality for all providers
type BaseProvider struct {
	id     string
	name   string
	config types.ProviderConfig
	status types.ProviderStatus
	mu     sync.RWMutex
}

// NewBaseProvider creates a base provider instance
func NewBaseProvider(config types.ProviderConfig) *BaseProvider {
	return &BaseProvider{
		id:     config.ID,
		name:   config.Name,
		config: config,
		status: types.ProviderStatus{
			Available:   true,
			HealthScore: 100.0,
			LastCheck:   time.Now(),
		},
	}
}

// GetID returns the provider ID
func (p *BaseProvider) GetID() string {
	return p.id
}

// GetStatus returns the current provider status
func (p *BaseProvider) GetStatus() types.ProviderStatus {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.status
}

// UpdateStatus updates the provider status
func (p *BaseProvider) UpdateStatus(status types.ProviderStatus) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.status = status
}

// MockProvider is a simple mock provider for testing
type MockProvider struct {
	*BaseProvider
	responseDelay time.Duration
	errorRate     float64
}

// NewMockProvider creates a mock provider
func NewMockProvider(id string) *MockProvider {
	config := types.ProviderConfig{
		ID:        id,
		Name:      "Mock Provider " + id,
		Type:      types.ProviderTypeLLM,
		RateLimit: 100,
		Timeout:   30 * time.Second,
	}
	
	return &MockProvider{
		BaseProvider:  NewBaseProvider(config),
		responseDelay: 100 * time.Millisecond,
		errorRate:     0.0,
	}
}

// Execute processes a request through the mock provider
func (p *MockProvider) Execute(ctx context.Context, req *types.Request) (*types.Response, error) {
	// Simulate processing delay
	select {
	case <-time.After(p.responseDelay):
		// Continue processing
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	
	// Create mock response
	response := &types.Response{
		ID:         fmt.Sprintf("resp-%d", time.Now().UnixNano()),
		RequestID:  req.ID,
		ProviderID: p.GetID(),
		Status:     types.StatusSuccess,
		Result:     fmt.Sprintf("Mock response to: %s", req.Prompt),
		Usage: &types.Usage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
		ProcessingTime: p.responseDelay.Seconds(),
		Timestamp:      time.Now(),
	}
	
	return response, nil
}

// Close cleanly shuts down the provider
func (p *MockProvider) Close() error {
	p.UpdateStatus(types.ProviderStatus{
		Available: false,
		Message:   "Provider closed",
		LastCheck: time.Now(),
	})
	return nil
}

// Registry manages provider registration and lookup
type Registry struct {
	mu        sync.RWMutex
	providers map[string]types.Provider
}

// NewRegistry creates a new provider registry
func NewRegistry() *Registry {
	return &Registry{
		providers: make(map[string]types.Provider),
	}
}

// Register adds a provider to the registry
func (r *Registry) Register(provider types.Provider) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	id := provider.GetID()
	if _, exists := r.providers[id]; exists {
		return fmt.Errorf("provider %s already registered", id)
	}
	
	r.providers[id] = provider
	return nil
}

// Get retrieves a provider by ID
func (r *Registry) Get(id string) (types.Provider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	provider, ok := r.providers[id]
	if !ok {
		return nil, fmt.Errorf("provider %s not found", id)
	}
	return provider, nil
}

// List returns all registered providers
func (r *Registry) List() []types.Provider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	providers := make([]types.Provider, 0, len(r.providers))
	for _, p := range r.providers {
		providers = append(providers, p)
	}
	return providers
}

// Remove unregisters a provider
func (r *Registry) Remove(id string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if _, exists := r.providers[id]; !exists {
		return fmt.Errorf("provider %s not found", id)
	}
	
	// Close the provider before removing
	if provider, ok := r.providers[id]; ok {
		provider.Close()
	}
	
	delete(r.providers, id)
	return nil
}