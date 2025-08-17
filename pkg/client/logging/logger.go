package logging

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/pkgerrors"
)

// ContextKey for storing values in context
type ContextKey string

const (
	// CorrelationIDKey is the context key for correlation ID
	CorrelationIDKey ContextKey = "correlation_id"
	
	// RequestIDKey is the context key for request ID
	RequestIDKey ContextKey = "request_id"
	
	// SessionIDKey is the context key for session ID
	SessionIDKey ContextKey = "session_id"
	
	// PackIDKey is the context key for pack ID
	PackIDKey ContextKey = "pack_id"
)

var (
	// Global logger instance
	globalLogger *zerolog.Logger
	loggerOnce   sync.Once
	
	// Default log level
	defaultLevel = zerolog.InfoLevel
)

// Config represents logger configuration
type Config struct {
	Level      string `json:"level" yaml:"level"`
	Output     string `json:"output" yaml:"output"` // console, json
	File       string `json:"file" yaml:"file"`
	MaxSize    int    `json:"max_size" yaml:"max_size"`       // MB
	MaxBackups int    `json:"max_backups" yaml:"max_backups"`
	MaxAge     int    `json:"max_age" yaml:"max_age"` // days
	Compress   bool   `json:"compress" yaml:"compress"`
}

// Initialize sets up the global logger
func Initialize(cfg *Config) {
	loggerOnce.Do(func() {
		globalLogger = createLogger(cfg)
	})
}

// GetLogger returns the global logger instance
func GetLogger() *zerolog.Logger {
	if globalLogger == nil {
		// Initialize with defaults if not already done
		Initialize(&Config{
			Level:  "info",
			Output: "console",
		})
	}
	return globalLogger
}

// createLogger creates a new logger instance
func createLogger(cfg *Config) *zerolog.Logger {
	// Configure error stack marshaler
	zerolog.ErrorStackMarshaler = pkgerrors.MarshalStack
	
	// Parse log level
	level := parseLevel(cfg.Level)
	
	// Create output writer
	var output io.Writer
	if cfg.File != "" {
		// TODO: Add file rotation support
		// SECURITY: Use 0600 permissions - readable/writable by owner only
		// This prevents other users from reading potentially sensitive log data
		file, err := os.OpenFile(cfg.File, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0600)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to open log file: %v\n", err)
			output = os.Stderr
		} else {
			output = file
		}
	} else {
		output = os.Stderr
	}
	
	// Configure output format
	if cfg.Output == "console" {
		output = zerolog.ConsoleWriter{
			Out:        output,
			TimeFormat: time.RFC3339,
			FieldsExclude: []string{
				"hostname",
			},
		}
	}
	
	// Create logger
	logger := zerolog.New(output).
		Level(level).
		With().
		Timestamp().
		Str("service", "canidae-sdk").
		Str("version", "0.1.0").
		Logger()
	
	return &logger
}

// parseLevel parses string log level to zerolog level
func parseLevel(level string) zerolog.Level {
	switch strings.ToLower(level) {
	case "trace":
		return zerolog.TraceLevel
	case "debug":
		return zerolog.DebugLevel
	case "info":
		return zerolog.InfoLevel
	case "warn", "warning":
		return zerolog.WarnLevel
	case "error":
		return zerolog.ErrorLevel
	case "fatal":
		return zerolog.FatalLevel
	case "panic":
		return zerolog.PanicLevel
	default:
		return defaultLevel
	}
}

// WithContext returns a logger with context values
func WithContext(ctx context.Context) zerolog.Logger {
	logger := GetLogger().With()
	
	// Add correlation ID if present, auto-generate if not
	correlationID := GetCorrelationID(ctx)
	if correlationID == "" {
		correlationID = GenerateCorrelationID()
	}
	logger = logger.Str("correlation_id", correlationID)
	
	// Add request ID if present
	if requestID := GetRequestID(ctx); requestID != "" {
		logger = logger.Str("request_id", requestID)
	}
	
	// Add session ID if present
	if sessionID := GetSessionID(ctx); sessionID != "" {
		logger = logger.Str("session_id", sessionID)
	}
	
	// Add pack ID if present
	if packID := GetPackID(ctx); packID != "" {
		logger = logger.Str("pack_id", packID)
	}
	
	return logger.Logger()
}

// WithCorrelationID adds correlation ID to context
func WithCorrelationID(ctx context.Context, correlationID string) context.Context {
	return context.WithValue(ctx, CorrelationIDKey, correlationID)
}

// GetCorrelationID retrieves correlation ID from context
func GetCorrelationID(ctx context.Context) string {
	if v := ctx.Value(CorrelationIDKey); v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// WithRequestID adds request ID to context
func WithRequestID(ctx context.Context, requestID string) context.Context {
	return context.WithValue(ctx, RequestIDKey, requestID)
}

// GetRequestID retrieves request ID from context
func GetRequestID(ctx context.Context) string {
	if v := ctx.Value(RequestIDKey); v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// WithSessionID adds session ID to context
func WithSessionID(ctx context.Context, sessionID string) context.Context {
	return context.WithValue(ctx, SessionIDKey, sessionID)
}

// GetSessionID retrieves session ID from context
func GetSessionID(ctx context.Context) string {
	if v := ctx.Value(SessionIDKey); v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// WithPackID adds pack ID to context
func WithPackID(ctx context.Context, packID string) context.Context {
	return context.WithValue(ctx, PackIDKey, packID)
}

// GetPackID retrieves pack ID from context
func GetPackID(ctx context.Context) string {
	if v := ctx.Value(PackIDKey); v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// LogError logs an error with context
func LogError(ctx context.Context, err error, msg string) {
	logger := WithContext(ctx)
	
	// Add stack trace if available
	logger.Error().
		Stack().
		Err(err).
		Msg(msg)
}

// LogErrorWithFields logs an error with additional fields
func LogErrorWithFields(ctx context.Context, err error, msg string, fields map[string]interface{}) {
	logger := WithContext(ctx)
	
	event := logger.Error().Stack().Err(err)
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	event.Msg(msg)
}

// LogRequest logs an incoming request
func LogRequest(ctx context.Context, method, path string, fields map[string]interface{}) {
	logger := WithContext(ctx)
	
	event := logger.Info().
		Str("method", method).
		Str("path", path)
	
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	
	event.Msg("Request received")
}

// LogResponse logs an outgoing response
func LogResponse(ctx context.Context, statusCode int, duration time.Duration, fields map[string]interface{}) {
	logger := WithContext(ctx)
	
	event := logger.Info().
		Int("status_code", statusCode).
		Dur("duration", duration)
	
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	
	event.Msg("Response sent")
}

// LogMetric logs a metric
func LogMetric(ctx context.Context, name string, value interface{}, fields map[string]interface{}) {
	logger := WithContext(ctx)
	
	event := logger.Info().
		Str("metric_name", name).
		Interface("metric_value", value)
	
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	
	event.Msg("Metric recorded")
}

// LogDebug logs a debug message
func LogDebug(ctx context.Context, msg string, fields map[string]interface{}) {
	logger := WithContext(ctx)
	
	event := logger.Debug()
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	
	event.Msg(msg)
}

// LogInfo logs an info message
func LogInfo(ctx context.Context, msg string, fields map[string]interface{}) {
	logger := WithContext(ctx)
	
	event := logger.Info()
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	
	event.Msg(msg)
}

// LogWarn logs a warning message
func LogWarn(ctx context.Context, msg string, fields map[string]interface{}) {
	logger := WithContext(ctx)
	
	event := logger.Warn()
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	
	event.Msg(msg)
}

// LogFatal logs a fatal message and exits
func LogFatal(ctx context.Context, msg string, fields map[string]interface{}) {
	logger := WithContext(ctx)
	
	event := logger.Fatal()
	for k, v := range fields {
		event = event.Interface(k, v)
	}
	
	event.Msg(msg)
}

// WithCaller adds caller information to the logger
func WithCaller() zerolog.Logger {
	return GetLogger().With().Caller().Logger()
}

// WithFields returns a logger with additional fields
func WithFields(fields map[string]interface{}) zerolog.Logger {
	logger := GetLogger().With()
	
	for k, v := range fields {
		logger = logger.Interface(k, v)
	}
	
	return logger.Logger()
}

// GetCallerInfo returns caller file and line number
func GetCallerInfo(skip int) (string, int) {
	_, file, line, ok := runtime.Caller(skip + 1)
	if !ok {
		return "unknown", 0
	}
	
	// Extract just the filename
	parts := strings.Split(file, "/")
	if len(parts) > 0 {
		file = parts[len(parts)-1]
	}
	
	return file, line
}

// GenerateCorrelationID generates a unique correlation ID
func GenerateCorrelationID() string {
	// Generate 8 random bytes
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		// Fallback to timestamp if random generation fails
		return fmt.Sprintf("canidae-%d", time.Now().UnixNano())
	}
	
	// Format as hex string with prefix
	return fmt.Sprintf("canidae-%s", hex.EncodeToString(b))
}

// EnsureCorrelationID ensures the context has a correlation ID
func EnsureCorrelationID(ctx context.Context) context.Context {
	if GetCorrelationID(ctx) == "" {
		return WithCorrelationID(ctx, GenerateCorrelationID())
	}
	return ctx
}