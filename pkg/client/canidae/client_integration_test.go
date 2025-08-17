package canidae_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/macawi-ai/canidae/pkg/client/canidae"
)

// TestExecuteAgentMethod tests the ExecuteAgent method
func TestExecuteAgentMethod(t *testing.T) {
	client, err := canidae.NewClient(
		canidae.WithServerEndpoint("localhost:50051"),
		canidae.WithAPIKey("test-key"),
	)
	assert.NoError(t, err)
	assert.NotNil(t, client)

	ctx := context.Background()

	// Test ExecuteAgent without connection
	req := &canidae.ExecuteRequest{
		Agent:  canidae.AgentTypeAnthropic,
		Prompt: "Test prompt",
	}

	// Should fail since we're not connected
	resp, err := client.ExecuteAgent(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "not connected")
}

// TestChainAgentsMethod tests the ChainAgents method
func TestChainAgentsMethod(t *testing.T) {
	client, err := canidae.NewClient(
		canidae.WithServerEndpoint("localhost:50051"),
		canidae.WithAPIKey("test-key"),
	)
	assert.NoError(t, err)
	assert.NotNil(t, client)

	ctx := context.Background()

	// Test ChainAgents without connection
	req := &canidae.ChainRequest{
		Steps: []canidae.ChainStep{
			{
				Agent:  canidae.AgentTypeAnthropic,
				Prompt: "Step 1",
			},
			{
				Agent:  canidae.AgentTypeOpenAI,
				Prompt: "Step 2",
			},
		},
	}

	// Should fail since we're not connected
	resp, err := client.ChainAgents(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "not connected")
}

// TestSummonPackMethod tests the SummonPack method
func TestSummonPackMethod(t *testing.T) {
	client, err := canidae.NewClient(
		canidae.WithServerEndpoint("localhost:50051"),
		canidae.WithAPIKey("test-key"),
	)
	assert.NoError(t, err)
	assert.NotNil(t, client)

	ctx := context.Background()

	// Test SummonPack without connection
	req := &canidae.PackRequest{
		Formation: canidae.PackFormation{
			Alpha: &canidae.PackMember{
				Role:      "coordinator",
				Agent:     canidae.AgentTypeAnthropic,
				Objective: "Coordinate tasks",
			},
		},
		Objective: "Complete complex task",
	}

	// Should fail since we're not connected
	resp, err := client.SummonPack(ctx, req)
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "not connected")
}

// TestStreamMethod tests the Stream method
func TestStreamMethod(t *testing.T) {
	client, err := canidae.NewClient(
		canidae.WithServerEndpoint("localhost:50051"),
		canidae.WithAPIKey("test-key"),
	)
	assert.NoError(t, err)
	assert.NotNil(t, client)

	ctx := context.Background()

	// Define a handler
	eventCount := 0
	handler := func(event canidae.StreamEvent) error {
		eventCount++
		return nil
	}

	// Should fail since we're not connected
	err = client.Stream(ctx, handler)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not connected")
	assert.Equal(t, 0, eventCount)
}

// TestConnectDisconnectFlow tests the connect and disconnect flow
func TestConnectDisconnectFlow(t *testing.T) {
	client, err := canidae.NewClient(
		canidae.WithServerEndpoint("localhost:50051"),
		canidae.WithAPIKey("test-key"),
		canidae.WithTimeout(100*time.Millisecond),
	)
	assert.NoError(t, err)
	assert.NotNil(t, client)

	ctx := context.Background()

	// Initial status should show not connected
	status := client.GetStatus()
	assert.False(t, status.Connected)
	assert.False(t, status.Authenticated)

	// Try to connect (will fail without server, but tests the flow)
	err = client.Connect(ctx)
	if err != nil {
		// Expected without running server
		assert.Contains(t, err.Error(), "connection")
	}

	// Disconnect should work even if not connected
	err = client.Disconnect(ctx)
	assert.NoError(t, err)

	// Status should still show not connected
	status = client.GetStatus()
	assert.False(t, status.Connected)
}

// TestRequestValidationInMethods tests that methods validate requests
func TestRequestValidationInMethods(t *testing.T) {
	client, err := canidae.NewClient(
		canidae.WithServerEndpoint("localhost:50051"),
	)
	assert.NoError(t, err)

	ctx := context.Background()

	t.Run("ExecuteAgent with invalid request", func(t *testing.T) {
		// Invalid request - missing agent
		req := &canidae.ExecuteRequest{
			Prompt: "Test",
		}
		
		resp, err := client.ExecuteAgent(ctx, req)
		assert.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "agent type is required")
	})

	t.Run("ChainAgents with invalid request", func(t *testing.T) {
		// Invalid request - empty steps
		req := &canidae.ChainRequest{
			Steps: []canidae.ChainStep{},
		}
		
		resp, err := client.ChainAgents(ctx, req)
		assert.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "at least one step is required")
	})

	t.Run("SummonPack with invalid request", func(t *testing.T) {
		// Invalid request - missing alpha
		req := &canidae.PackRequest{
			Objective: "Test",
		}
		
		resp, err := client.SummonPack(ctx, req)
		assert.Error(t, err)
		assert.Nil(t, resp)
		assert.Contains(t, err.Error(), "pack alpha is required")
	})
}

// TestNilRequests tests handling of nil requests
func TestNilRequests(t *testing.T) {
	client, err := canidae.NewClient()
	assert.NoError(t, err)

	ctx := context.Background()

	t.Run("ExecuteAgent with nil request", func(t *testing.T) {
		resp, err := client.ExecuteAgent(ctx, nil)
		assert.Error(t, err)
		assert.Nil(t, resp)
	})

	t.Run("ChainAgents with nil request", func(t *testing.T) {
		resp, err := client.ChainAgents(ctx, nil)
		assert.Error(t, err)
		assert.Nil(t, resp)
	})

	t.Run("SummonPack with nil request", func(t *testing.T) {
		resp, err := client.SummonPack(ctx, nil)
		assert.Error(t, err)
		assert.Nil(t, resp)
	})

	t.Run("Stream with nil handler", func(t *testing.T) {
		err := client.Stream(ctx, nil)
		assert.Error(t, err)
	})
}

// TestContextCancellationInMethods tests context cancellation handling
func TestContextCancellationInMethods(t *testing.T) {
	client, err := canidae.NewClient()
	assert.NoError(t, err)

	// Create a cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	t.Run("Connect with cancelled context", func(t *testing.T) {
		err := client.Connect(ctx)
		assert.Error(t, err)
	})

	t.Run("ExecuteAgent with cancelled context", func(t *testing.T) {
		req := &canidae.ExecuteRequest{
			Agent:  canidae.AgentTypeAnthropic,
			Prompt: "Test",
		}
		resp, err := client.ExecuteAgent(ctx, req)
		assert.Error(t, err)
		assert.Nil(t, resp)
	})

	t.Run("ChainAgents with cancelled context", func(t *testing.T) {
		req := &canidae.ChainRequest{
			Steps: []canidae.ChainStep{
				{Agent: canidae.AgentTypeAnthropic, Prompt: "Test"},
			},
		}
		resp, err := client.ChainAgents(ctx, req)
		assert.Error(t, err)
		assert.Nil(t, resp)
	})

	t.Run("SummonPack with cancelled context", func(t *testing.T) {
		req := &canidae.PackRequest{
			Formation: canidae.PackFormation{
				Alpha: &canidae.PackMember{
					Role:      "alpha",
					Agent:     canidae.AgentTypeAnthropic,
					Objective: "Lead",
				},
			},
			Objective: "Test",
		}
		resp, err := client.SummonPack(ctx, req)
		assert.Error(t, err)
		assert.Nil(t, resp)
	})
}

// TestTimeoutHandling tests timeout handling in methods
func TestTimeoutHandling(t *testing.T) {
	client, err := canidae.NewClient(
		canidae.WithTimeout(1 * time.Nanosecond), // Very short timeout
	)
	assert.NoError(t, err)

	ctx := context.Background()

	// All operations should timeout quickly
	t.Run("Connect with timeout", func(t *testing.T) {
		err := client.Connect(ctx)
		assert.Error(t, err)
	})
}

// TestConcurrentMethodCalls tests concurrent method calls
func TestConcurrentMethodCalls(t *testing.T) {
	client, err := canidae.NewClient()
	assert.NoError(t, err)

	ctx := context.Background()
	done := make(chan bool, 30)

	// Concurrent ExecuteAgent calls
	for i := 0; i < 10; i++ {
		go func() {
			req := &canidae.ExecuteRequest{
				Agent:  canidae.AgentTypeAnthropic,
				Prompt: "Test",
			}
			_, _ = client.ExecuteAgent(ctx, req)
			done <- true
		}()
	}

	// Concurrent ChainAgents calls
	for i := 0; i < 10; i++ {
		go func() {
			req := &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{Agent: canidae.AgentTypeAnthropic, Prompt: "Test"},
				},
			}
			_, _ = client.ChainAgents(ctx, req)
			done <- true
		}()
	}

	// Concurrent SummonPack calls
	for i := 0; i < 10; i++ {
		go func() {
			req := &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "alpha",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "Lead",
					},
				},
				Objective: "Test",
			}
			_, _ = client.SummonPack(ctx, req)
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 30; i++ {
		<-done
	}
}

// BenchmarkExecuteAgent benchmarks ExecuteAgent method
func BenchmarkExecuteAgent(b *testing.B) {
	client, _ := canidae.NewClient()
	ctx := context.Background()
	req := &canidae.ExecuteRequest{
		Agent:  canidae.AgentTypeAnthropic,
		Prompt: "Benchmark prompt",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.ExecuteAgent(ctx, req)
	}
}

// BenchmarkChainAgents benchmarks ChainAgents method
func BenchmarkChainAgents(b *testing.B) {
	client, _ := canidae.NewClient()
	ctx := context.Background()
	req := &canidae.ChainRequest{
		Steps: []canidae.ChainStep{
			{Agent: canidae.AgentTypeAnthropic, Prompt: "Step 1"},
			{Agent: canidae.AgentTypeOpenAI, Prompt: "Step 2"},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.ChainAgents(ctx, req)
	}
}

// BenchmarkSummonPack benchmarks SummonPack method
func BenchmarkSummonPack(b *testing.B) {
	client, _ := canidae.NewClient()
	ctx := context.Background()
	req := &canidae.PackRequest{
		Formation: canidae.PackFormation{
			Alpha: &canidae.PackMember{
				Role:      "alpha",
				Agent:     canidae.AgentTypeAnthropic,
				Objective: "Lead",
			},
		},
		Objective: "Benchmark",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = client.SummonPack(ctx, req)
	}
}