package canidae_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/macawi-ai/canidae/pkg/client/canidae"
)

func TestClientCreation(t *testing.T) {
	tests := []struct {
		name    string
		opts    []canidae.Option
		wantErr bool
	}{
		{
			name:    "valid client with defaults",
			opts:    []canidae.Option{},
			wantErr: false,
		},
		{
			name: "client with server endpoint",
			opts: []canidae.Option{
				canidae.WithServerEndpoint("192.168.1.38:14001"),
			},
			wantErr: false,
		},
		{
			name: "client with full configuration",
			opts: []canidae.Option{
				canidae.WithServerEndpoint("192.168.1.38:14001"),
				canidae.WithPackID("test-pack"),
				canidae.WithSecurityProfile(canidae.SecurityProfileEnterprise),
				canidae.WithTimeout(30 * time.Second),
				canidae.WithAPIKey("test-api-key"),
			},
			wantErr: false,
		},
		{
			name: "client with mTLS",
			opts: []canidae.Option{
				canidae.WithMTLS("cert.pem", "key.pem", "ca.pem"),
			},
			wantErr: false,
		},
		{
			name: "client with WebAuthn",
			opts: []canidae.Option{
				canidae.WithWebAuthn("canidae.example.com", "https://canidae.example.com"),
			},
			wantErr: false,
		},
		{
			name: "client with OAuth",
			opts: []canidae.Option{
				canidae.WithOAuth(
					"client-id",
					"client-secret",
					"https://auth.example.com/authorize",
					"https://auth.example.com/token",
				),
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := canidae.NewClient(tt.opts...)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, client)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, client)

				// Check status
				status := client.GetStatus()
				assert.NotNil(t, status)
				assert.False(t, status.Connected)
				assert.False(t, status.Authenticated)
			}
		})
	}
}

func TestExecuteRequest(t *testing.T) {
	tests := []struct {
		name    string
		request *canidae.ExecuteRequest
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid request",
			request: &canidae.ExecuteRequest{
				Agent:  canidae.AgentTypeAnthropic,
				Prompt: "Test prompt",
			},
			wantErr: false,
		},
		{
			name: "request with all fields",
			request: &canidae.ExecuteRequest{
				Agent:           canidae.AgentTypeOpenAI,
				Prompt:          "Test prompt",
				Model:           "gpt-4",
				Temperature:     0.7,
				MaxTokens:       500,
				SecurityProfile: canidae.SecurityProfileEnterprise,
				Metadata: map[string]string{
					"test": "value",
				},
			},
			wantErr: false,
		},
		{
			name: "missing agent",
			request: &canidae.ExecuteRequest{
				Prompt: "Test prompt",
			},
			wantErr: true,
			errMsg:  "agent type is required",
		},
		{
			name: "missing prompt",
			request: &canidae.ExecuteRequest{
				Agent: canidae.AgentTypeGemini,
			},
			wantErr: true,
			errMsg:  "prompt is required",
		},
		{
			name:    "empty request",
			request: &canidae.ExecuteRequest{},
			wantErr: true,
			errMsg:  "agent type is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.request.Validate()
			if tt.wantErr {
				assert.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestChainRequest(t *testing.T) {
	tests := []struct {
		name    string
		request *canidae.ChainRequest
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid chain with single step",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Agent:  canidae.AgentTypeAnthropic,
						Prompt: "Step 1",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "chain with dependencies",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Agent:  canidae.AgentTypeOpenAI,
						Prompt: "Step 1",
					},
					{
						Agent:     canidae.AgentTypeAnthropic,
						Prompt:    "Step 2",
						DependsOn: []string{"openai"},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "chain with multiple steps",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Agent:       canidae.AgentTypeOpenAI,
						Prompt:      "Step 1",
						Model:       "gpt-4",
						Temperature: 0.7,
					},
					{
						Agent:  canidae.AgentTypeAnthropic,
						Prompt: "Step 2",
						Model:  "claude-3-opus",
					},
					{
						Agent:  canidae.AgentTypeGemini,
						Prompt: "Step 3",
						Model:  "gemini-pro",
					},
				},
				SecurityProfile: canidae.SecurityProfileEnterprise,
				ContinueOnError: true,
			},
			wantErr: false,
		},
		{
			name: "empty chain",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{},
			},
			wantErr: true,
			errMsg:  "at least one step is required",
		},
		{
			name: "step missing agent",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Prompt: "Step 1",
					},
				},
			},
			wantErr: true,
			errMsg:  "agent type is required",
		},
		{
			name: "step missing prompt",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Agent: canidae.AgentTypeOpenAI,
					},
				},
			},
			wantErr: true,
			errMsg:  "prompt is required",
		},
		{
			name: "invalid dependency",
			request: &canidae.ChainRequest{
				Steps: []canidae.ChainStep{
					{
						Agent:  canidae.AgentTypeOpenAI,
						Prompt: "Step 1",
					},
					{
						Agent:     canidae.AgentTypeAnthropic,
						Prompt:    "Step 2",
						DependsOn: []string{"nonexistent"},
					},
				},
			},
			wantErr: true,
			errMsg:  "invalid dependency",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.request.Validate()
			if tt.wantErr {
				assert.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestPackRequest(t *testing.T) {
	tests := []struct {
		name    string
		request *canidae.PackRequest
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid pack with alpha only",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "coordinator",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "Coordinate tasks",
					},
				},
				Objective: "Complete complex task",
			},
			wantErr: false,
		},
		{
			name: "full pack formation",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "coordinator",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "Coordinate",
						Model:     "claude-3-opus",
					},
					Hunters: []canidae.PackMember{
						{
							Role:      "researcher",
							Agent:     canidae.AgentTypeOpenAI,
							Objective: "Research",
							Model:     "gpt-4",
						},
						{
							Role:      "analyzer",
							Agent:     canidae.AgentTypeGemini,
							Objective: "Analyze",
							Model:     "gemini-pro",
						},
					},
					Scouts: []canidae.PackMember{
						{
							Role:      "explorer",
							Agent:     canidae.AgentTypeOllama,
							Objective: "Explore",
							Model:     "llama2",
						},
					},
					Sentries: []canidae.PackMember{
						{
							Role:      "guardian",
							Agent:     canidae.AgentTypeAnthropic,
							Objective: "Guard",
						},
					},
					Elders: []canidae.PackMember{
						{
							Role:      "wisdom",
							Agent:     canidae.AgentTypeOpenAI,
							Objective: "Advise",
						},
					},
				},
				Objective:       "Solve complex problem",
				SecurityProfile: canidae.SecurityProfileEnterprise,
				MaxConcurrency:  5,
				Timeout:         60 * time.Second,
			},
			wantErr: false,
		},
		{
			name: "missing objective",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Alpha: &canidae.PackMember{
						Role:      "coordinator",
						Agent:     canidae.AgentTypeAnthropic,
						Objective: "Coordinate",
					},
				},
			},
			wantErr: true,
			errMsg:  "objective is required",
		},
		{
			name: "missing alpha",
			request: &canidae.PackRequest{
				Formation: canidae.PackFormation{
					Hunters: []canidae.PackMember{
						{
							Role:      "researcher",
							Agent:     canidae.AgentTypeOpenAI,
							Objective: "Research",
						},
					},
				},
				Objective: "Complete task",
			},
			wantErr: true,
			errMsg:  "pack alpha is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.request.Validate()
			if tt.wantErr {
				assert.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestClientStatus(t *testing.T) {
	client, err := canidae.NewClient(
		canidae.WithServerEndpoint("192.168.1.38:14001"),
		canidae.WithPackID("test-pack"),
	)
	require.NoError(t, err)
	require.NotNil(t, client)

	// Check initial status
	status := client.GetStatus()
	assert.False(t, status.Connected)
	assert.False(t, status.Authenticated)
	assert.Equal(t, "test-pack", status.PackID)
	assert.Equal(t, "192.168.1.38:14001", status.ServerEndpoint)
	assert.WithinDuration(t, time.Now(), status.LastActivity, 1*time.Second)

	// Set pack ID
	client.SetPackID("new-pack")
	status = client.GetStatus()
	assert.Equal(t, "new-pack", status.PackID)
}

func TestSecurityProfiles(t *testing.T) {
	profiles := []canidae.SecurityProfile{
		canidae.SecurityProfileEnterprise,
		canidae.SecurityProfileFinance,
		canidae.SecurityProfileICS,
		canidae.SecurityProfileDebug,
		canidae.SecurityProfilePermissive,
	}

	for _, profile := range profiles {
		t.Run(string(profile), func(t *testing.T) {
			client, err := canidae.NewClient(
				canidae.WithSecurityProfile(profile),
			)
			assert.NoError(t, err)
			assert.NotNil(t, client)
		})
	}
}

func TestAgentTypes(t *testing.T) {
	agents := []canidae.AgentType{
		canidae.AgentTypeAnthropic,
		canidae.AgentTypeOpenAI,
		canidae.AgentTypeGemini,
		canidae.AgentTypeOllama,
		canidae.AgentTypeDeepSeek,
	}

	for _, agent := range agents {
		t.Run(string(agent), func(t *testing.T) {
			req := &canidae.ExecuteRequest{
				Agent:  agent,
				Prompt: "Test",
			}
			assert.NoError(t, req.Validate())
		})
	}
}

// TestConnectionLifecycle tests connection and disconnection
// This test requires a mock server or will be skipped
func TestConnectionLifecycle(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping connection test in short mode")
	}

	client, err := canidae.NewClient(
		canidae.WithServerEndpoint("localhost:50051"),
		canidae.WithAPIKey("test-key"),
		canidae.WithTimeout(5*time.Second),
	)
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// This will fail without a running server, which is expected
	err = client.Connect(ctx)
	if err != nil {
		t.Logf("Expected connection error without server: %v", err)
	}

	// Disconnect should work even if not connected
	err = client.Disconnect(ctx)
	assert.NoError(t, err)
}

// BenchmarkClientCreation benchmarks client creation
func BenchmarkClientCreation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = canidae.NewClient(
			canidae.WithServerEndpoint("192.168.1.38:14001"),
			canidae.WithPackID("bench-pack"),
			canidae.WithAPIKey("bench-key"),
		)
	}
}

// BenchmarkRequestValidation benchmarks request validation
func BenchmarkRequestValidation(b *testing.B) {
	req := &canidae.ExecuteRequest{
		Agent:  canidae.AgentTypeAnthropic,
		Prompt: "Benchmark prompt",
		Model:  "claude-3-opus",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = req.Validate()
	}
}
