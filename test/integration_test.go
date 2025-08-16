package test

import (
	"context"
	"testing"
	"time"
	
	"github.com/canidae/canidae/internal/providers"
	"github.com/canidae/canidae/internal/ring"
	"github.com/canidae/canidae/pkg/types"
	"github.com/nats-io/nats.go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNATSMessageRouting tests NATS message routing functionality
func TestNATSMessageRouting(t *testing.T) {
	// Skip if NATS not available
	nc, err := nats.Connect("nats://localhost:4222")
	if err != nil {
		t.Skip("NATS not available, skipping integration test")
		return
	}
	defer nc.Close()
	
	// Create Ring with NATS
	config := ring.Config{
		NatsURL:    "nats://localhost:4222",
		Name:       "test-ring",
		MaxRetries: 3,
		RetryDelay: time.Second,
	}
	
	r, err := ring.NewRing(config)
	require.NoError(t, err)
	defer r.Close()
	
	// Test message subscription
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	// Start Ring in background
	go func() {
		err := r.Start(ctx)
		assert.NoError(t, err)
	}()
	
	// Give it time to subscribe
	time.Sleep(100 * time.Millisecond)
	
	// Send test message
	testMsg := []byte(`{"test": "message"}`)
	err = nc.Publish("canidae.request.test-pack", testMsg)
	assert.NoError(t, err)
	
	// Verify no errors in short period
	time.Sleep(500 * time.Millisecond)
}

// TestProviderRegistrationDiscovery tests provider registration and discovery
func TestProviderRegistrationDiscovery(t *testing.T) {
	// Create registry
	registry := providers.NewRegistry()
	
	// Test registration
	provider1 := providers.NewMockProvider("test-provider-1")
	err := registry.Register(provider1)
	assert.NoError(t, err)
	
	provider2 := providers.NewMockProvider("test-provider-2")
	err = registry.Register(provider2)
	assert.NoError(t, err)
	
	// Test duplicate registration
	err = registry.Register(provider1)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "already registered")
	
	// Test discovery
	found, err := registry.Get("test-provider-1")
	assert.NoError(t, err)
	assert.NotNil(t, found)
	assert.Equal(t, "test-provider-1", found.GetID())
	
	// Test not found
	_, err = registry.Get("non-existent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
	
	// Test list
	providers := registry.List()
	assert.Len(t, providers, 2)
	
	// Test removal
	err = registry.Remove("test-provider-1")
	assert.NoError(t, err)
	
	providers = registry.List()
	assert.Len(t, providers, 1)
}

// TestProviderExecution tests basic provider execution
func TestProviderExecution(t *testing.T) {
	// Create mock provider
	provider := providers.NewMockProvider("test-provider")
	
	// Create request
	req := &types.Request{
		ID:         "test-req-1",
		PackID:     "test-pack",
		ProviderID: "test-provider",
		Prompt:     "Test prompt",
		Priority:   types.PriorityMedium,
		Timeout:    5 * time.Second,
		Timestamp:  time.Now(),
	}
	
	// Execute request
	ctx := context.Background()
	response, err := provider.Execute(ctx, req)
	
	// Verify response
	assert.NoError(t, err)
	assert.NotNil(t, response)
	assert.Equal(t, types.StatusSuccess, response.Status)
	assert.Contains(t, response.Result, "Mock response to: Test prompt")
	assert.NotNil(t, response.Usage)
	assert.Greater(t, response.Usage.TotalTokens, 0)
}

// TestRingOrchestration tests Ring orchestration with providers
func TestRingOrchestration(t *testing.T) {
	// Create Ring (will use mock NATS if real NATS not available)
	config := ring.Config{
		NatsURL:    "nats://localhost:4222",
		Name:       "test-ring",
		MaxRetries: 3,
		RetryDelay: time.Second,
	}
	
	// Allow failure if NATS not available - Ring will still work
	r, _ := ring.NewRing(config)
	if r == nil {
		t.Skip("Could not create Ring, skipping orchestration test")
		return
	}
	defer r.Close()
	
	// Verify providers are registered
	providers := r.ListProviders()
	assert.GreaterOrEqual(t, len(providers), 2, "Should have at least 2 mock providers")
	
	// Test processing a request
	req := &types.Request{
		ID:         "test-req-2",
		PackID:     "test-pack",
		ProviderID: "mock-1",
		Prompt:     "Test orchestration",
		Priority:   types.PriorityHigh,
		Timeout:    5 * time.Second,
		Timestamp:  time.Now(),
	}
	
	ctx := context.Background()
	response, err := r.Process(ctx, req)
	
	if err == nil {
		assert.NotNil(t, response)
		assert.Equal(t, types.StatusSuccess, response.Status)
		assert.Contains(t, response.Result, "Mock response")
	}
}

// TestProviderHealthStatus tests provider health status
func TestProviderHealthStatus(t *testing.T) {
	provider := providers.NewMockProvider("health-test")
	
	// Check initial status
	status := provider.GetStatus()
	assert.True(t, status.Available)
	assert.Equal(t, 100.0, status.HealthScore)
	
	// Close provider
	err := provider.Close()
	assert.NoError(t, err)
	
	// Check status after close
	status = provider.GetStatus()
	assert.False(t, status.Available)
	assert.Equal(t, "Provider closed", status.Message)
}