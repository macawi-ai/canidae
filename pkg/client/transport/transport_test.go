package transport_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/macawi-ai/canidae/pkg/client/config"
	"github.com/macawi-ai/canidae/pkg/client/transport"
)

func TestRequestCreation(t *testing.T) {
	req := &transport.Request{
		ID:   "test-123",
		Type: transport.RequestTypeExecute,
		Payload: map[string]interface{}{
			"prompt": "test prompt",
			"model":  "test-model",
		},
		Headers: map[string]string{
			"X-Request-ID": "test-123",
		},
		Metadata: map[string]string{
			"source": "test",
		},
		Timeout: 30 * time.Second,
	}

	assert.Equal(t, "test-123", req.ID)
	assert.Equal(t, transport.RequestTypeExecute, req.Type)
	assert.NotNil(t, req.Payload)
	assert.NotNil(t, req.Headers)
	assert.NotNil(t, req.Metadata)
	assert.Equal(t, 30*time.Second, req.Timeout)
}

func TestResponseParsing(t *testing.T) {
	type TestData struct {
		Message string `json:"message"`
		Value   int    `json:"value"`
	}

	testData := TestData{
		Message: "test message",
		Value:   42,
	}

	data, err := json.Marshal(testData)
	require.NoError(t, err)

	resp := &transport.Response{
		ID:      "resp-123",
		Success: true,
		Data:    json.RawMessage(data),
		Headers: map[string]string{
			"X-Response-ID": "resp-123",
		},
		Metadata: map[string]string{
			"processed": "true",
		},
	}

	// Test unmarshaling
	var result TestData
	err = resp.UnmarshalTo(&result)
	assert.NoError(t, err)
	assert.Equal(t, "test message", result.Message)
	assert.Equal(t, 42, result.Value)

	// Test error response
	errResp := &transport.Response{
		ID:      "err-123",
		Success: false,
		Error: &transport.Error{
			Code:    "INVALID_REQUEST",
			Message: "Invalid request format",
			Details: "Missing required field: prompt",
		},
	}

	assert.False(t, errResp.Success)
	assert.NotNil(t, errResp.Error)
	assert.Equal(t, "INVALID_REQUEST", errResp.Error.Code)
	assert.Contains(t, errResp.Error.Error(), "Invalid request format")
	assert.Contains(t, errResp.Error.Error(), "Missing required field")
}

func TestRequestTypes(t *testing.T) {
	types := []transport.RequestType{
		transport.RequestTypeExecute,
		transport.RequestTypeChain,
		transport.RequestTypePack,
		transport.RequestTypeStream,
		transport.RequestTypeControl,
	}

	for _, reqType := range types {
		t.Run(string(reqType), func(t *testing.T) {
			assert.NotEmpty(t, reqType)
			assert.IsType(t, transport.RequestType(""), reqType)
		})
	}
}

func TestStreamEventTypes(t *testing.T) {
	types := []transport.StreamEventType{
		transport.StreamEventTypeData,
		transport.StreamEventTypeError,
		transport.StreamEventTypeComplete,
		transport.StreamEventTypeHeartbeat,
	}

	for _, eventType := range types {
		t.Run(string(eventType), func(t *testing.T) {
			event := &transport.StreamEvent{
				Type: eventType,
				Data: json.RawMessage(`{"test": "data"}`),
				Metadata: map[string]string{
					"event_type": string(eventType),
				},
			}

			assert.Equal(t, eventType, event.Type)
			assert.NotNil(t, event.Data)
			assert.NotNil(t, event.Metadata)
		})
	}
}

func TestMetrics(t *testing.T) {
	metrics := &transport.Metrics{
		RequestsSent:      10,
		ResponsesReceived: 9,
		ErrorCount:        1,
		BytesSent:         1024,
		BytesReceived:     2048,
		AverageLatency:    100 * time.Millisecond,
		LastActivity:      time.Now(),
	}

	assert.Equal(t, int64(10), metrics.RequestsSent)
	assert.Equal(t, int64(9), metrics.ResponsesReceived)
	assert.Equal(t, int64(1), metrics.ErrorCount)
	assert.Equal(t, int64(1024), metrics.BytesSent)
	assert.Equal(t, int64(2048), metrics.BytesReceived)
	assert.Equal(t, 100*time.Millisecond, metrics.AverageLatency)
	assert.WithinDuration(t, time.Now(), metrics.LastActivity, 1*time.Second)
}

func TestTransportConfig(t *testing.T) {
	cfg := config.TransportConfig{
		Type: "grpc",
		TLS: config.TLSConfig{
			Enabled:    true,
			CertFile:   "cert.pem",
			KeyFile:    "key.pem",
			CAFile:     "ca.pem",
			ServerName: "canidae.example.com",
		},
		Retry: config.RetryConfig{
			Enabled:     true,
			MaxAttempts: 3,
			Backoff:     1 * time.Second,
			MaxBackoff:  30 * time.Second,
		},
		StreamBufferSize: 1024,
		MaxMessageSize:   4 * 1024 * 1024,
	}

	assert.Equal(t, "grpc", cfg.Type)
	assert.True(t, cfg.TLS.Enabled)
	assert.Equal(t, "cert.pem", cfg.TLS.CertFile)
	assert.True(t, cfg.Retry.Enabled)
	assert.Equal(t, 3, cfg.Retry.MaxAttempts)
	assert.Equal(t, 1024, cfg.StreamBufferSize)
	assert.Equal(t, 4*1024*1024, cfg.MaxMessageSize)
}

func TestTransportCreation(t *testing.T) {
	tests := []struct {
		name        string
		cfg         config.TransportConfig
		shouldError bool
		errorMsg    string
	}{
		{
			name: "gRPC transport",
			cfg: config.TransportConfig{
				Type: "grpc",
			},
			shouldError: false,
		},
		{
			name: "gRPC-Web transport",
			cfg: config.TransportConfig{
				Type: "grpcweb",
			},
			shouldError: true,
			errorMsg:    "gRPC-Web transport not yet implemented",
		},
		{
			name: "HTTP transport",
			cfg: config.TransportConfig{
				Type: "http",
			},
			shouldError: true,
			errorMsg:    "HTTP transport not yet implemented",
		},
		{
			name: "unsupported transport",
			cfg: config.TransportConfig{
				Type: "invalid",
			},
			shouldError: true,
			errorMsg:    "unsupported transport type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := transport.New(tt.cfg)
			if tt.shouldError {
				assert.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
				assert.Nil(t, client)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, client)
			}
		})
	}
}

// TestStreamHandler tests the stream handler function
func TestStreamHandler(t *testing.T) {
	called := false
	var receivedEvent *transport.StreamEvent

	handler := func(event *transport.StreamEvent) error {
		called = true
		receivedEvent = event
		return nil
	}

	testEvent := &transport.StreamEvent{
		Type: transport.StreamEventTypeData,
		Data: json.RawMessage(`{"message": "test"}`),
		Metadata: map[string]string{
			"test": "true",
		},
	}

	err := handler(testEvent)
	assert.NoError(t, err)
	assert.True(t, called)
	assert.NotNil(t, receivedEvent)
	assert.Equal(t, transport.StreamEventTypeData, receivedEvent.Type)
	assert.NotNil(t, receivedEvent.Data)
}

// TestConnectionMethods tests that connection methods are properly defined
func TestConnectionMethods(t *testing.T) {
	cfg := config.TransportConfig{
		Type: "grpc",
	}

	client, err := transport.New(cfg)
	require.NoError(t, err)
	require.NotNil(t, client)

	// Test that all interface methods exist
	ctx := context.Background()

	// These will fail without a server, but we're just testing the interface
	_ = client.Connect(ctx)
	_ = client.Disconnect(ctx)
	_ = client.IsConnected()

	client.SetSession("test-session")
	client.SetHeader("X-Test", "value")

	metrics := client.GetMetrics()
	assert.NotNil(t, metrics)
}

// BenchmarkRequestCreation benchmarks request creation
func BenchmarkRequestCreation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = &transport.Request{
			ID:   "bench-123",
			Type: transport.RequestTypeExecute,
			Payload: map[string]interface{}{
				"prompt": "benchmark",
			},
		}
	}
}

// BenchmarkResponseUnmarshal benchmarks response unmarshaling
func BenchmarkResponseUnmarshal(b *testing.B) {
	type TestData struct {
		Message string `json:"message"`
		Value   int    `json:"value"`
	}

	data, _ := json.Marshal(TestData{
		Message: "benchmark",
		Value:   42,
	})

	resp := &transport.Response{
		ID:      "bench-123",
		Success: true,
		Data:    json.RawMessage(data),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var result TestData
		_ = resp.UnmarshalTo(&result)
	}
}
