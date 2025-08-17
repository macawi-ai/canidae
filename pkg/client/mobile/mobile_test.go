package mobile_test

import (
	"encoding/json"
	"testing"
	
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	
	"github.com/macawi-ai/canidae/pkg/client/mobile"
)

func TestNewCanidaeClient(t *testing.T) {
	client := mobile.NewCanidaeClient()
	assert.NotNil(t, client)
}

func TestClientInitialization(t *testing.T) {
	tests := []struct {
		name      string
		configJSON string
		wantErr   bool
		errMsg    string
	}{
		{
			name: "valid configuration",
			configJSON: `{
				"serverEndpoint": "192.168.1.38:14001",
				"packID": "test-pack",
				"apiKey": "test-key",
				"securityProfile": "enterprise"
			}`,
			wantErr: false,
		},
		{
			name: "minimal configuration",
			configJSON: `{
				"serverEndpoint": "localhost:8080"
			}`,
			wantErr: false,
		},
		{
			name:      "invalid JSON",
			configJSON: `{invalid json}`,
			wantErr:   true,
			errMsg:    "invalid configuration JSON",
		},
		{
			name:      "empty configuration",
			configJSON: `{}`,
			wantErr:   false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := mobile.NewCanidaeClient()
			err := client.Initialize(tt.configJSON)
			
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

func TestHelperFunctions(t *testing.T) {
	t.Run("GetAgentTypes", func(t *testing.T) {
		agentsJSON := mobile.GetAgentTypes()
		
		var agents []string
		err := json.Unmarshal([]byte(agentsJSON), &agents)
		require.NoError(t, err)
		
		assert.Contains(t, agents, "anthropic")
		assert.Contains(t, agents, "openai")
		assert.Contains(t, agents, "gemini")
		assert.Contains(t, agents, "ollama")
		assert.Contains(t, agents, "deepseek")
	})
	
	t.Run("GetSecurityProfiles", func(t *testing.T) {
		profilesJSON := mobile.GetSecurityProfiles()
		
		var profiles []string
		err := json.Unmarshal([]byte(profilesJSON), &profiles)
		require.NoError(t, err)
		
		assert.Contains(t, profiles, "enterprise")
		assert.Contains(t, profiles, "finance")
		assert.Contains(t, profiles, "ics_iot")
		assert.Contains(t, profiles, "debug")
		assert.Contains(t, profiles, "permissive")
	})
	
	t.Run("CreateExecuteRequest", func(t *testing.T) {
		requestJSON := mobile.CreateExecuteRequest(
			"anthropic",
			"Test prompt",
			"claude-3-opus",
			0.7,
			500,
		)
		
		var request map[string]interface{}
		err := json.Unmarshal([]byte(requestJSON), &request)
		require.NoError(t, err)
		
		assert.Equal(t, "anthropic", request["agent"])
		assert.Equal(t, "Test prompt", request["prompt"])
		assert.Equal(t, "claude-3-opus", request["model"])
		assert.Equal(t, float64(0.7), request["temperature"])
		assert.Equal(t, float64(500), request["maxTokens"])
	})
	
	t.Run("CreateChainStep", func(t *testing.T) {
		stepJSON := mobile.CreateChainStep(
			"openai",
			"Step prompt",
			"gpt-4",
		)
		
		var step map[string]interface{}
		err := json.Unmarshal([]byte(stepJSON), &step)
		require.NoError(t, err)
		
		assert.Equal(t, "openai", step["agent"])
		assert.Equal(t, "Step prompt", step["prompt"])
		assert.Equal(t, "gpt-4", step["model"])
	})
}

func TestClientOperations(t *testing.T) {
	client := mobile.NewCanidaeClient()
	
	// Initialize with valid config
	config := `{
		"serverEndpoint": "localhost:50051",
		"apiKey": "test-key"
	}`
	
	err := client.Initialize(config)
	require.NoError(t, err)
	
	t.Run("GetStatus", func(t *testing.T) {
		// Status should work even without connection
		statusJSON, err := client.GetStatus()
		assert.NoError(t, err)
		
		var status map[string]interface{}
		err = json.Unmarshal([]byte(statusJSON), &status)
		assert.NoError(t, err)
		
		assert.Equal(t, false, status["connected"])
		assert.Equal(t, false, status["authenticated"])
	})
	
	t.Run("SetPackID", func(t *testing.T) {
		err := client.SetPackID("mobile-pack")
		assert.NoError(t, err)
		
		// Verify pack ID was set
		statusJSON, err := client.GetStatus()
		require.NoError(t, err)
		
		var status map[string]interface{}
		err = json.Unmarshal([]byte(statusJSON), &status)
		require.NoError(t, err)
		
		assert.Equal(t, "mobile-pack", status["pack_id"])
	})
	
	t.Run("Close", func(t *testing.T) {
		err := client.Close()
		assert.NoError(t, err)
	})
}

func TestUninitializedClient(t *testing.T) {
	client := mobile.NewCanidaeClient()
	
	// Operations should fail on uninitialized client
	t.Run("Connect", func(t *testing.T) {
		err := client.Connect()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "client not initialized")
	})
	
	t.Run("Disconnect", func(t *testing.T) {
		err := client.Disconnect()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "client not initialized")
	})
	
	t.Run("ExecuteAgent", func(t *testing.T) {
		_, err := client.ExecuteAgent(`{"agent": "anthropic", "prompt": "test"}`)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "client not initialized")
	})
	
	t.Run("ChainAgents", func(t *testing.T) {
		_, err := client.ChainAgents(`{"steps": []}`)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "client not initialized")
	})
	
	t.Run("GetStatus", func(t *testing.T) {
		_, err := client.GetStatus()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "client not initialized")
	})
	
	t.Run("SetPackID", func(t *testing.T) {
		err := client.SetPackID("test-pack")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "client not initialized")
	})
}

func TestJSONValidation(t *testing.T) {
	client := mobile.NewCanidaeClient()
	
	// Initialize client
	config := `{"serverEndpoint": "localhost:50051", "apiKey": "test"}`
	err := client.Initialize(config)
	require.NoError(t, err)
	
	t.Run("ExecuteAgent invalid JSON", func(t *testing.T) {
		_, err := client.ExecuteAgent(`{invalid}`)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid request JSON")
	})
	
	t.Run("ChainAgents invalid JSON", func(t *testing.T) {
		_, err := client.ChainAgents(`{invalid}`)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid request JSON")
	})
}

// BenchmarkClientCreation benchmarks client creation
func BenchmarkClientCreation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = mobile.NewCanidaeClient()
	}
}

// BenchmarkInitialization benchmarks client initialization
func BenchmarkInitialization(b *testing.B) {
	config := `{
		"serverEndpoint": "192.168.1.38:14001",
		"packID": "bench-pack",
		"apiKey": "bench-key",
		"securityProfile": "enterprise"
	}`
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		client := mobile.NewCanidaeClient()
		_ = client.Initialize(config)
	}
}

// BenchmarkHelpers benchmarks helper functions
func BenchmarkHelpers(b *testing.B) {
	b.Run("GetAgentTypes", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = mobile.GetAgentTypes()
		}
	})
	
	b.Run("CreateExecuteRequest", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = mobile.CreateExecuteRequest("anthropic", "test", "claude", 0.7, 500)
		}
	})
}