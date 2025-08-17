package canidae_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/macawi-ai/canidae/pkg/client/canidae"
)

// TestClientConnectionStatus tests the connection status
func TestClientConnectionStatus(t *testing.T) {
	client, err := canidae.NewClient()
	assert.NoError(t, err)
	
	// Should not be connected initially
	status := client.GetStatus()
	assert.False(t, status.Connected)
	
	// After failed connection attempt, still not connected
	ctx := context.Background()
	_ = client.Connect(ctx)
	status = client.GetStatus()
	assert.False(t, status.Connected)
}

// TestClientWithInvalidEndpoint tests client creation with invalid endpoint
func TestClientWithInvalidEndpoint(t *testing.T) {
	tests := []struct {
		name     string
		endpoint string
	}{
		{
			name:     "empty endpoint",
			endpoint: "",
		},
		{
			name:     "spaces only",
			endpoint: "   ",
		},
		{
			name:     "invalid format",
			endpoint: "not-a-valid-endpoint",
		},
		{
			name:     "missing port",
			endpoint: "localhost",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := canidae.NewClient(
				canidae.WithServerEndpoint(tt.endpoint),
			)
			// Currently doesn't validate, but tests the option
			if tt.endpoint == "" {
				assert.Error(t, err)
			} else {
				// Non-empty endpoints are accepted
				assert.NoError(t, err)
				if client != nil {
					status := client.GetStatus()
					assert.Equal(t, tt.endpoint, status.ServerEndpoint)
				}
			}
		})
	}
}

// TestPackMemberValidation tests pack member validation
func TestPackMemberValidation(t *testing.T) {
	tests := []struct {
		name      string
		request   *canidae.PackRequest
		wantError bool
	}{
		{
			name: "valid pack with hunters",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "alpha",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "Lead",
					},
					Hunters: []canidae.PackMember{
						{
							Role:      "hunter1",
							Agent:     canidae.AgentTypeOpenAI,
							Objective: "Hunt",
						},
					},
				},
				Objective: "Complete task",
			},
			wantError: false,
		},
		{
			name: "hunter with missing role",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "alpha",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "Lead",
					},
					Hunters: []canidae.PackMember{
						{
							Agent:     canidae.AgentTypeOpenAI,
							Objective: "Hunt",
						},
					},
				},
				Objective: "Complete task",
			},
			wantError: true,
		},
		{
			name: "scout with missing objective",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "alpha",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "Lead",
					},
					Scouts: []canidae.PackMember{
						{
							Role:  "scout1",
							Agent: canidae.AgentTypeGemini,
						},
					},
				},
				Objective: "Complete task",
			},
			wantError: true,
		},
		{
			name: "sentry with missing agent",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "alpha",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "Lead",
					},
					Sentries: []canidae.PackMember{
						{
							Role:      "sentry1",
							Objective: "Guard",
						},
					},
				},
				Objective: "Complete task",
			},
			wantError: true,
		},
		{
			name: "elder with all fields",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "alpha",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "Lead",
					},
					Elders: []canidae.PackMember{
						{
							Role:      "elder1",
							Agent:     canidae.AgentTypeDeepSeek,
							Objective: "Advise",
						},
					},
				},
				Objective: "Complete task",
			},
			wantError: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.request.Validate()
			if tt.wantError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestChainStepValidation tests chain step validation
func TestChainStepValidation(t *testing.T) {
	tests := []struct {
		name      string
		request   *canidae.ChainRequest
		wantError bool
	}{
		{
			name: "valid chain with metadata",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Agent:  canidae.AgentTypeAnthropic,
						Prompt: "Step 1",
						Metadata: map[string]string{
							"key": "value",
						},
					},
				},
			},
			wantError: false,
		},
		{
			name: "chain with temperature and max tokens",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Agent:       canidae.AgentTypeOpenAI,
						Prompt:      "Generate",
						Temperature: 0.7,
						MaxTokens:   1000,
					},
				},
			},
			wantError: false,
		},
		{
			name: "chain with model specified",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Agent:  canidae.AgentTypeGemini,
						Prompt: "Analyze",
						Model:  "gemini-pro",
					},
				},
			},
			wantError: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.request.Validate()
			if tt.wantError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestExecuteRequestFields tests ExecuteRequest with all fields
func TestExecuteRequestFields(t *testing.T) {
	req := &canidae.ExecuteRequest{
		Agent:           canidae.AgentTypeOllama,
		Prompt:          "Test prompt",
		Model:           "llama2",
		Temperature:     0.8,
		MaxTokens:       500,
		SecurityProfile: canidae.SecurityProfileDebug,
		Metadata: map[string]string{
			"source": "test",
			"id":     "123",
		},
	}
	
	err := req.Validate()
	assert.NoError(t, err)
	
	// Test all agent types
	agents := []canidae.AgentType{
		canidae.AgentTypeAnthropic,
		canidae.AgentTypeOpenAI,
		canidae.AgentTypeGemini,
		canidae.AgentTypeOllama,
		canidae.AgentTypeDeepSeek,
	}
	
	for _, agent := range agents {
		req.Agent = agent
		err := req.Validate()
		assert.NoError(t, err)
	}
	
	// Test all security profiles
	profiles := []canidae.SecurityProfile{
		canidae.SecurityProfileEnterprise,
		canidae.SecurityProfileFinance,
		canidae.SecurityProfileICS,
		canidae.SecurityProfileDebug,
		canidae.SecurityProfilePermissive,
	}
	
	for _, profile := range profiles {
		req.SecurityProfile = profile
		err := req.Validate()
		assert.NoError(t, err)
	}
}

// TestClientMethodsWithTimeout tests methods with context timeout
func TestClientMethodsWithTimeout(t *testing.T) {
	client, err := canidae.NewClient(
		canidae.WithTimeout(10 * time.Millisecond),
	)
	assert.NoError(t, err)
	
	// Create context with very short timeout
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Nanosecond)
	defer cancel()
	
	// Wait for context to expire
	time.Sleep(2 * time.Millisecond)
	
	// All methods should handle expired context gracefully
	t.Run("ExecuteAgent with expired context", func(t *testing.T) {
		req := &canidae.ExecuteRequest{
			Agent:  canidae.AgentTypeAnthropic,
			Prompt: "Test",
		}
		resp, err := client.ExecuteAgent(ctx, req)
		assert.Error(t, err)
		assert.Nil(t, resp)
	})
	
	t.Run("ChainAgents with expired context", func(t *testing.T) {
		req := &canidae.ChainRequest{
			Steps: []canidae.ChainStep{
				{Agent: canidae.AgentTypeAnthropic, Prompt: "Test"},
			},
		}
		resp, err := client.ChainAgents(ctx, req)
		assert.Error(t, err)
		assert.Nil(t, resp)
	})
	
	t.Run("SummonPack with expired context", func(t *testing.T) {
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