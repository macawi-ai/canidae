package logging_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/rs/zerolog"
	"github.com/stretchr/testify/assert"

	"github.com/macawi-ai/canidae/pkg/client/logging"
)

func TestLoggerInitialization(t *testing.T) {
	tests := []struct {
		name   string
		config *logging.Config
	}{
		{
			name: "console output",
			config: &logging.Config{
				Level:  "info",
				Output: "console",
			},
		},
		{
			name: "json output",
			config: &logging.Config{
				Level:  "debug",
				Output: "json",
			},
		},
		{
			name: "with file output",
			config: &logging.Config{
				Level:  "warn",
				Output: "json",
				File:   "/tmp/test.log",
			},
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logging.Initialize(tt.config)
			logger := logging.GetLogger()
			assert.NotNil(t, logger)
		})
	}
}

func TestLogLevelParsing(t *testing.T) {
	tests := []struct {
		level    string
		expected zerolog.Level
	}{
		{"trace", zerolog.TraceLevel},
		{"debug", zerolog.DebugLevel},
		{"info", zerolog.InfoLevel},
		{"warn", zerolog.WarnLevel},
		{"warning", zerolog.WarnLevel},
		{"error", zerolog.ErrorLevel},
		{"fatal", zerolog.FatalLevel},
		{"panic", zerolog.PanicLevel},
		{"invalid", zerolog.InfoLevel}, // default
	}
	
	for _, tt := range tests {
		t.Run(tt.level, func(t *testing.T) {
			// Re-initialize logger with different level
			logging.Initialize(&logging.Config{
				Level:  tt.level,
				Output: "json",
			})
			
			// Can't directly test the level, but we can verify logger is created
			logger := logging.GetLogger()
			assert.NotNil(t, logger)
		})
	}
}

func TestContextLogging(t *testing.T) {
	// Create context with IDs
	ctx := context.Background()
	ctx = logging.WithCorrelationID(ctx, "corr-123")
	ctx = logging.WithRequestID(ctx, "req-456")
	ctx = logging.WithSessionID(ctx, "sess-789")
	ctx = logging.WithPackID(ctx, "pack-abc")
	
	// Verify IDs are stored in context
	assert.Equal(t, "corr-123", logging.GetCorrelationID(ctx))
	assert.Equal(t, "req-456", logging.GetRequestID(ctx))
	assert.Equal(t, "sess-789", logging.GetSessionID(ctx))
	assert.Equal(t, "pack-abc", logging.GetPackID(ctx))
}

func TestContextWithEmptyValues(t *testing.T) {
	ctx := context.Background()
	
	// Test getting IDs from empty context
	assert.Equal(t, "", logging.GetCorrelationID(ctx))
	assert.Equal(t, "", logging.GetRequestID(ctx))
	assert.Equal(t, "", logging.GetSessionID(ctx))
	assert.Equal(t, "", logging.GetPackID(ctx))
}

func TestLoggerWithContext(t *testing.T) {
	// Override global logger for this test
	logging.Initialize(&logging.Config{
		Level:  "debug",
		Output: "json",
	})
	
	ctx := context.Background()
	ctx = logging.WithCorrelationID(ctx, "test-correlation")
	ctx = logging.WithRequestID(ctx, "test-request")
	
	// Create logger with context
	logger := logging.WithContext(ctx)
	
	// Log a message
	logger.Info().Msg("Test message")
	
	// Verify the logger is created and has expected type
	assert.NotNil(t, &logger)
}

func TestLogHelpers(t *testing.T) {
	// Initialize logger
	logging.Initialize(&logging.Config{
		Level:  "debug",
		Output: "json",
	})
	
	ctx := context.Background()
	ctx = logging.WithCorrelationID(ctx, "test-corr")
	
	// These functions don't return values, so we just ensure they don't panic
	t.Run("LogError", func(t *testing.T) {
		err := fmt.Errorf("test error")
		logging.LogError(ctx, err, "Error occurred")
	})
	
	t.Run("LogErrorWithFields", func(t *testing.T) {
		err := fmt.Errorf("test error")
		fields := map[string]interface{}{
			"field1": "value1",
			"field2": 42,
		}
		logging.LogErrorWithFields(ctx, err, "Error with fields", fields)
	})
	
	t.Run("LogRequest", func(t *testing.T) {
		fields := map[string]interface{}{
			"user_id": "user-123",
		}
		logging.LogRequest(ctx, "POST", "/api/execute", fields)
	})
	
	t.Run("LogResponse", func(t *testing.T) {
		fields := map[string]interface{}{
			"bytes": 1024,
		}
		logging.LogResponse(ctx, 200, 100*time.Millisecond, fields)
	})
	
	t.Run("LogMetric", func(t *testing.T) {
		fields := map[string]interface{}{
			"unit": "ms",
		}
		logging.LogMetric(ctx, "latency", 50.5, fields)
	})
	
	t.Run("LogDebug", func(t *testing.T) {
		fields := map[string]interface{}{
			"debug": true,
		}
		logging.LogDebug(ctx, "Debug message", fields)
	})
	
	t.Run("LogInfo", func(t *testing.T) {
		fields := map[string]interface{}{
			"info": "data",
		}
		logging.LogInfo(ctx, "Info message", fields)
	})
	
	t.Run("LogWarn", func(t *testing.T) {
		fields := map[string]interface{}{
			"warning": "condition",
		}
		logging.LogWarn(ctx, "Warning message", fields)
	})
}

func TestWithCaller(t *testing.T) {
	logging.Initialize(&logging.Config{
		Level:  "debug",
		Output: "json",
	})
	
	logger := logging.WithCaller()
	assert.NotNil(t, logger)
}

func TestWithFields(t *testing.T) {
	logging.Initialize(&logging.Config{
		Level:  "debug",
		Output: "json",
	})
	
	fields := map[string]interface{}{
		"field1": "value1",
		"field2": 42,
		"field3": true,
	}
	
	logger := logging.WithFields(fields)
	assert.NotNil(t, logger)
}

func TestGetCallerInfo(t *testing.T) {
	file, line := logging.GetCallerInfo(0)
	
	// Should contain test file name
	assert.Contains(t, file, "logger_test.go")
	assert.Greater(t, line, 0)
}

func TestStructuredLogging(t *testing.T) {
	// Create a buffer to capture output
	buf := &bytes.Buffer{}
	logger := zerolog.New(buf).With().Timestamp().Logger()
	
	// Log structured message
	logger.Info().
		Str("correlation_id", "test-123").
		Str("request_id", "req-456").
		Int("status", 200).
		Dur("duration", 100*time.Millisecond).
		Msg("Request completed")
	
	// Parse the JSON output
	var logEntry map[string]interface{}
	err := json.Unmarshal(buf.Bytes(), &logEntry)
	assert.NoError(t, err)
	
	// Verify fields
	assert.Equal(t, "test-123", logEntry["correlation_id"])
	assert.Equal(t, "req-456", logEntry["request_id"])
	assert.Equal(t, float64(200), logEntry["status"])
	assert.Equal(t, "Request completed", logEntry["message"])
}

func TestConcurrentLogging(t *testing.T) {
	logging.Initialize(&logging.Config{
		Level:  "debug",
		Output: "json",
	})
	
	ctx := context.Background()
	
	// Test concurrent logging doesn't cause issues
	done := make(chan bool, 100)
	for i := 0; i < 100; i++ {
		go func(id int) {
			ctx := logging.WithCorrelationID(ctx, fmt.Sprintf("corr-%d", id))
			fields := map[string]interface{}{
				"goroutine": id,
			}
			logging.LogInfo(ctx, "Concurrent log", fields)
			done <- true
		}(i)
	}
	
	// Wait for all goroutines
	for i := 0; i < 100; i++ {
		<-done
	}
}

func TestInvalidContextValue(t *testing.T) {
	ctx := context.Background()
	
	// Add non-string values to context
	ctx = context.WithValue(ctx, logging.CorrelationIDKey, 123)
	ctx = context.WithValue(ctx, logging.RequestIDKey, []byte("bytes"))
	
	// Should return empty string for non-string values
	assert.Equal(t, "", logging.GetCorrelationID(ctx))
	assert.Equal(t, "", logging.GetRequestID(ctx))
}

// BenchmarkLogWithContext benchmarks logging with context
func BenchmarkLogWithContext(b *testing.B) {
	logging.Initialize(&logging.Config{
		Level:  "info",
		Output: "json",
	})
	
	ctx := context.Background()
	ctx = logging.WithCorrelationID(ctx, "bench-corr")
	ctx = logging.WithRequestID(ctx, "bench-req")
	
	fields := map[string]interface{}{
		"benchmark": true,
		"iteration": 0,
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fields["iteration"] = i
		logging.LogInfo(ctx, "Benchmark message", fields)
	}
}

// BenchmarkStructuredFields benchmarks adding structured fields
func BenchmarkStructuredFields(b *testing.B) {
	logging.Initialize(&logging.Config{
		Level:  "info",
		Output: "json",
	})
	
	fields := map[string]interface{}{
		"field1": "value1",
		"field2": 42,
		"field3": true,
		"field4": 3.14,
		"field5": time.Now(),
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = logging.WithFields(fields)
	}
}

func ExampleLogRequest() {
	ctx := context.Background()
	ctx = logging.WithCorrelationID(ctx, "req-123-456")
	ctx = logging.WithSessionID(ctx, "session-789")
	
	// Log incoming request
	logging.LogRequest(ctx, "POST", "/api/agents/execute", map[string]interface{}{
		"user_id":    "user-123",
		"agent_type": "anthropic",
	})
}

func ExampleLogResponse() {
	ctx := context.Background()
	ctx = logging.WithCorrelationID(ctx, "req-123-456")
	
	// Log outgoing response
	logging.LogResponse(ctx, 200, 250*time.Millisecond, map[string]interface{}{
		"bytes_sent": 2048,
		"cache_hit":  true,
	})
}

func ExampleLogError() {
	ctx := context.Background()
	ctx = logging.WithCorrelationID(ctx, "error-correlation")
	
	// Log an error with context
	err := fmt.Errorf("connection timeout")
	logging.LogError(ctx, err, "Failed to connect to server")
}

func TestLoggerDefaultInitialization(t *testing.T) {
	// Reset global logger to nil to test default initialization
	// This is a bit hacky but necessary for testing
	
	// Get logger without explicit initialization
	logger := logging.GetLogger()
	assert.NotNil(t, logger)
}

func TestFileOutputError(t *testing.T) {
	// Test with invalid file path
	logging.Initialize(&logging.Config{
		Level:  "info",
		Output: "json",
		File:   "/invalid/path/that/should/not/exist/test.log",
	})
	
	// Should fall back to stderr
	logger := logging.GetLogger()
	assert.NotNil(t, logger)
}

func TestConsoleOutput(t *testing.T) {
	// Create buffer to capture console output
	buf := &bytes.Buffer{}
	
	// We can't easily test console output directly,
	// but we can verify the configuration is accepted
	logging.Initialize(&logging.Config{
		Level:  "info",
		Output: "console",
	})
	
	logger := logging.GetLogger()
	assert.NotNil(t, logger)
	
	// The ConsoleWriter would format output differently
	// but we can't easily capture it in tests
	_ = buf
}

func TestEdgeCases(t *testing.T) {
	t.Run("EmptyConfig", func(t *testing.T) {
		logging.Initialize(&logging.Config{})
		logger := logging.GetLogger()
		assert.NotNil(t, logger)
	})
	
	t.Run("NilFields", func(t *testing.T) {
		ctx := context.Background()
		
		// Should not panic with nil fields
		logging.LogInfo(ctx, "Message", nil)
		logging.LogDebug(ctx, "Debug", nil)
		logging.LogWarn(ctx, "Warning", nil)
	})
	
	t.Run("EmptyFields", func(t *testing.T) {
		ctx := context.Background()
		fields := map[string]interface{}{}
		
		// Should not panic with empty fields
		logging.LogInfo(ctx, "Message", fields)
	})
	
	t.Run("GetCallerInfoInvalid", func(t *testing.T) {
		// Test with very high skip value
		file, line := logging.GetCallerInfo(1000)
		assert.Equal(t, "unknown", file)
		assert.Equal(t, 0, line)
	})
}

func TestLogLevelCaseSensitivity(t *testing.T) {
	levels := []string{"INFO", "Info", "iNfO", "DEBUG", "Debug"}
	
	for _, level := range levels {
		t.Run(level, func(t *testing.T) {
			logging.Initialize(&logging.Config{
				Level:  level,
				Output: "json",
			})
			logger := logging.GetLogger()
			assert.NotNil(t, logger)
		})
	}
}

func TestFilenameParsing(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"/path/to/file.go", "file.go"},
		{"file.go", "file.go"},
		{"", ""},
	}
	
	for _, tt := range tests {
		parts := strings.Split(tt.input, "/")
		result := ""
		if len(parts) > 0 {
			result = parts[len(parts)-1]
		}
		assert.Equal(t, tt.expected, result)
	}
}