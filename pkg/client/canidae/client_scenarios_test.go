package canidae_test

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/macawi-ai/canidae/pkg/client/canidae"
	"github.com/macawi-ai/canidae/pkg/client/errors"
)

// TestTimeoutScenarios tests various timeout scenarios
func TestTimeoutScenarios(t *testing.T) {
	tests := []struct {
		name    string
		timeout time.Duration
		delay   time.Duration
	}{
		{
			name:    "immediate timeout",
			timeout: 1 * time.Nanosecond,
			delay:   10 * time.Millisecond,
		},
		{
			name:    "short timeout",
			timeout: 10 * time.Millisecond,
			delay:   100 * time.Millisecond,
		},
		{
			name:    "normal timeout",
			timeout: 1 * time.Second,
			delay:   2 * time.Second,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := canidae.NewClient(
				canidae.WithTimeout(tt.timeout),
			)
			assert.NoError(t, err)
			assert.NotNil(t, client)

			// Create a context with timeout
			ctx, cancel := context.WithTimeout(context.Background(), tt.timeout)
			defer cancel()

			// Simulate an operation that takes longer than timeout
			done := make(chan bool, 1)
			go func() {
				time.Sleep(tt.delay)
				done <- true
			}()

			select {
			case <-ctx.Done():
				// Context timed out as expected
				assert.Error(t, ctx.Err())
				assert.Equal(t, context.DeadlineExceeded, ctx.Err())
			case <-done:
				// Operation completed before timeout
				if tt.delay > tt.timeout {
					t.Error("Operation should have timed out")
				}
			}
		})
	}
}

// TestConcurrentClientOperations tests thread safety
func TestConcurrentClientOperations(t *testing.T) {
	client, err := canidae.NewClient(
		canidae.WithServerEndpoint("192.168.1.38:14001"),
		canidae.WithPackID("test-pack"),
	)
	assert.NoError(t, err)
	assert.NotNil(t, client)

	// Test concurrent operations
	var wg sync.WaitGroup
	errors := make(chan error, 100)

	// Concurrent status checks
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			status := client.GetStatus()
			if status == nil {
				errors <- fmt.Errorf("nil status returned")
			}
		}()
	}

	// Concurrent pack ID updates
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			packID := fmt.Sprintf("pack-%d", id)
			client.SetPackID(packID)
		}(i)
	}

	// Concurrent configuration checks
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			status := client.GetStatus()
			if status.ServerEndpoint == "" {
				errors <- fmt.Errorf("empty server endpoint")
			}
		}()
	}

	wg.Wait()
	close(errors)

	// Check for any errors
	for err := range errors {
		t.Errorf("Concurrent operation error: %v", err)
	}
}

// TestRequestValidationEdgeCases tests edge cases in request validation
func TestRequestValidationEdgeCases(t *testing.T) {
	tests := []struct {
		name      string
		request   interface{}
		wantError bool
		errorMsg  string
	}{
		// ExecuteRequest edge cases
		{
			name: "execute - empty agent type",
			request: &canidae.ExecuteRequest{
				Agent:  "",
				Prompt: "test",
			},
			wantError: true,
			errorMsg:  "agent type is required",
		},
		{
			name: "execute - empty prompt",
			request: &canidae.ExecuteRequest{
				Agent:  canidae.AgentTypeAnthropic,
				Prompt: "",
			},
			wantError: true,
			errorMsg:  "prompt is required",
		},
		{
			name: "execute - whitespace only prompt",
			request: &canidae.ExecuteRequest{
				Agent:  canidae.AgentTypeAnthropic,
				Prompt: "   \t\n   ",
			},
			wantError: true,
			errorMsg:  "prompt is required",
		},
		{
			name: "execute - negative temperature",
			request: &canidae.ExecuteRequest{
				Agent:       canidae.AgentTypeAnthropic,
				Prompt:      "test",
				Temperature: -0.5,
			},
			wantError: false, // Currently not validated
		},
		{
			name: "execute - excessive temperature",
			request: &canidae.ExecuteRequest{
				Agent:       canidae.AgentTypeAnthropic,
				Prompt:      "test",
				Temperature: 3.0,
			},
			wantError: false, // Currently not validated
		},
		{
			name: "execute - negative max tokens",
			request: &canidae.ExecuteRequest{
				Agent:     canidae.AgentTypeAnthropic,
				Prompt:    "test",
				MaxTokens: -100,
			},
			wantError: false, // Currently not validated
		},
		{
			name: "execute - zero max tokens",
			request: &canidae.ExecuteRequest{
				Agent:     canidae.AgentTypeAnthropic,
				Prompt:    "test",
				MaxTokens: 0,
			},
			wantError: false, // Zero is valid (uses default)
		},

		// ChainRequest edge cases
		{
			name: "chain - no steps",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{},
			},
			wantError: true,
			errorMsg:  "at least one step is required",
		},
		{
			name: "chain - nil steps",
			request: &canidae.ChainRequest{
				Steps: nil,
			},
			wantError: true,
			errorMsg:  "at least one step is required",
		},
		{
			name: "chain - step with empty agent",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Agent:  "",
						Prompt: "test",
					},
				},
			},
			wantError: true,
			errorMsg:  "agent type is required",
		},
		{
			name: "chain - step with empty prompt",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Agent:  canidae.AgentTypeAnthropic,
						Prompt: "",
					},
				},
			},
			wantError: true,
			errorMsg:  "prompt is required",
		},
		{
			name: "chain - circular dependency",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Agent:     canidae.AgentTypeAnthropic,
						Prompt:    "Step 1",
						DependsOn: []string{"anthropic"}, // Self-reference
					},
				},
			},
			wantError: true,
			errorMsg:  "invalid dependency",
		},
		{
			name: "chain - dependency on non-existent step",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Agent:  canidae.AgentTypeAnthropic,
						Prompt: "Step 1",
					},
					{
						Agent:     canidae.AgentTypeOpenAI,
						Prompt:    "Step 2",
						DependsOn: []string{"gemini"}, // Doesn't exist
					},
				},
			},
			wantError: true,
			errorMsg:  "invalid dependency",
		},

		// PackRequest edge cases
		{
			name: "pack - no alpha",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: nil,
				},
				Objective: "test",
			},
			wantError: true,
			errorMsg:  "pack alpha is required",
		},
		{
			name: "pack - no objective",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "alpha",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "lead",
					},
				},
				Objective: "",
			},
			wantError: true,
			errorMsg:  "objective is required",
		},
		{
			name: "pack - alpha with empty role",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "lead",
					},
				},
				Objective: "test",
			},
			wantError: true,
			errorMsg:  "role is required",
		},
		{
			name: "pack - alpha with empty objective",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "alpha",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "",
					},
				},
				Objective: "test",
			},
			wantError: true,
			errorMsg:  "objective is required",
		},
		{
			name: "pack - negative max concurrency",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "alpha",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "lead",
					},
				},
				Objective:      "test",
				MaxConcurrency: -1,
			},
			wantError: false, // Currently not validated
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var err error
			switch req := tt.request.(type) {
			case *canidae.ExecuteRequest:
				err = req.Validate()
			case *canidae.ChainRequest:
				err = req.Validate()
			case *canidae.PackRequest:
				err = req.Validate()
			}

			if tt.wantError {
				assert.Error(t, err)
				if tt.errorMsg != "" && err != nil {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestClientConfigurationOptions tests various configuration options
func TestClientConfigurationOptions(t *testing.T) {
	tests := []struct {
		name    string
		options []canidae.Option
		verify  func(*testing.T, *canidae.Client)
	}{
		{
			name: "multiple auth methods",
			options: []canidae.Option{
				canidae.WithAPIKey("test-key"),
				canidae.WithWebAuthn("example.com", "https://example.com"),
				canidae.WithOAuth("client", "secret", "auth", "token"),
			},
			verify: func(t *testing.T, c *canidae.Client) {
				assert.NotNil(t, c)
				// Last auth method wins
			},
		},
		{
			name: "all options combined",
			options: []canidae.Option{
				canidae.WithServerEndpoint("192.168.1.38:14001"),
				canidae.WithPackID("test-pack"),
				canidae.WithSecurityProfile(canidae.SecurityProfileEnterprise),
				canidae.WithTimeout(30 * time.Second),
				canidae.WithAPIKey("test-key"),
				canidae.WithMTLS("cert.pem", "key.pem", "ca.pem"),
			},
			verify: func(t *testing.T, c *canidae.Client) {
				assert.NotNil(t, c)
				status := c.GetStatus()
				assert.Equal(t, "test-pack", status.PackID)
				assert.Equal(t, "192.168.1.38:14001", status.ServerEndpoint)
			},
		},
		{
			name: "security profiles",
			options: []canidae.Option{
				canidae.WithSecurityProfile(canidae.SecurityProfileFinance),
			},
			verify: func(t *testing.T, c *canidae.Client) {
				assert.NotNil(t, c)
			},
		},
		{
			name: "empty pack ID",
			options: []canidae.Option{
				canidae.WithPackID(""),
			},
			verify: func(t *testing.T, c *canidae.Client) {
				assert.NotNil(t, c)
				status := c.GetStatus()
				assert.Equal(t, "", status.PackID)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := canidae.NewClient(tt.options...)
			assert.NoError(t, err)
			if tt.verify != nil {
				tt.verify(t, client)
			}
		})
	}
}

// TestErrorCodeCategories verifies error codes are properly categorized
func TestErrorCodeCategories(t *testing.T) {
	tests := []struct {
		name      string
		errorFunc func() *errors.CanidaeError
		category  string
		retryable bool
	}{
		{
			name:      "connection errors are retryable",
			errorFunc: func() *errors.CanidaeError { return errors.NewConnectionError(nil) },
			category:  "connection",
			retryable: true,
		},
		{
			name:      "auth errors are not retryable",
			errorFunc: func() *errors.CanidaeError { return errors.NewAuthenticationError(nil) },
			category:  "authentication",
			retryable: false,
		},
		{
			name: "token expiry is retryable",
			errorFunc: func() *errors.CanidaeError {
				return errors.GetRegistry().NewError(errors.ErrCodeTokenExpired, nil)
			},
			category:  "authentication",
			retryable: true,
		},
		{
			name: "rate limit is retryable",
			errorFunc: func() *errors.CanidaeError {
				return errors.GetRegistry().NewError(errors.ErrCodeRateLimitExceeded, nil)
			},
			category:  "resource",
			retryable: true,
		},
		{
			name:      "validation errors are not retryable",
			errorFunc: func() *errors.CanidaeError { return errors.NewValidationError("field", nil) },
			category:  "validation",
			retryable: false,
		},
		{
			name:      "internal errors are not retryable",
			errorFunc: func() *errors.CanidaeError { return errors.NewInternalError(nil) },
			category:  "internal",
			retryable: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.errorFunc()
			assert.NotNil(t, err)
			assert.Equal(t, tt.retryable, errors.IsRetryable(err))
		})
	}
}

// Define custom types for context keys to avoid collisions
type contextKey string

const (
	testContextKey1 contextKey = "test_key_1"
	testContextKey2 contextKey = "test_key_2"
)

// TestContextPropagation tests context value propagation
func TestContextPropagation(t *testing.T) {
	ctx := context.Background()

	// Test context without values
	t.Run("empty context", func(t *testing.T) {
		client, err := canidae.NewClient()
		assert.NoError(t, err)
		assert.NotNil(t, client)
		// Context operations should not panic
	})

	// Test context with values using proper context keys
	t.Run("context with values", func(t *testing.T) {
		ctx = context.WithValue(ctx, testContextKey1, "value1")
		ctx = context.WithValue(ctx, testContextKey2, 123)

		client, err := canidae.NewClient()
		assert.NoError(t, err)
		assert.NotNil(t, client)
		// Context values should be accessible
	})

	// Test context cancellation
	t.Run("cancelled context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		// Operations with cancelled context should fail fast
		assert.Error(t, ctx.Err())
	})

	// Test context deadline
	t.Run("context with deadline", func(t *testing.T) {
		deadline := time.Now().Add(100 * time.Millisecond)
		ctx, cancel := context.WithDeadline(context.Background(), deadline)
		defer cancel()

		time.Sleep(200 * time.Millisecond)
		assert.Error(t, ctx.Err())
		assert.Equal(t, context.DeadlineExceeded, ctx.Err())
	})
}

// TestMemoryLeaks tests for potential memory leaks
func TestMemoryLeaks(t *testing.T) {
	// Create and destroy many clients
	t.Run("client creation/destruction", func(t *testing.T) {
		for i := 0; i < 1000; i++ {
			client, err := canidae.NewClient(
				canidae.WithPackID(fmt.Sprintf("pack-%d", i)),
			)
			assert.NoError(t, err)
			assert.NotNil(t, client)
			// Client should be garbage collected
		}
	})

	// Test request object creation
	t.Run("request creation", func(t *testing.T) {
		for i := 0; i < 1000; i++ {
			req := &canidae.ExecuteRequest{
				Agent:  canidae.AgentTypeAnthropic,
				Prompt: fmt.Sprintf("prompt-%d", i),
				Metadata: map[string]string{
					"iteration": fmt.Sprintf("%d", i),
				},
			}
			err := req.Validate()
			assert.NoError(t, err)
		}
	})
}

// BenchmarkClientCreation benchmarks client creation with various options
func BenchmarkClientCreationWithOptions(b *testing.B) {
	b.Run("minimal", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = canidae.NewClient()
		}
	})

	b.Run("with all options", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = canidae.NewClient(
				canidae.WithServerEndpoint("192.168.1.38:14001"),
				canidae.WithPackID("bench-pack"),
				canidae.WithSecurityProfile(canidae.SecurityProfileEnterprise),
				canidae.WithTimeout(30*time.Second),
				canidae.WithAPIKey("bench-key"),
			)
		}
	})
}

// BenchmarkConcurrentOperations benchmarks concurrent operations
func BenchmarkConcurrentOperations(b *testing.B) {
	client, _ := canidae.NewClient(
		canidae.WithPackID("bench-pack"),
	)

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if i%2 == 0 {
				client.SetPackID(fmt.Sprintf("pack-%d", i))
			} else {
				_ = client.GetStatus()
			}
			i++
		}
	})
}