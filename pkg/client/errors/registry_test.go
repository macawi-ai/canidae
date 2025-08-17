package errors_test

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/macawi-ai/canidae/pkg/client/errors"
)

func TestErrorRegistry(t *testing.T) {
	registry := errors.GetRegistry()
	assert.NotNil(t, registry)
	
	// Test singleton pattern
	registry2 := errors.GetRegistry()
	assert.Same(t, registry, registry2)
}

func TestCanidaeError(t *testing.T) {
	originalErr := fmt.Errorf("original error")
	err := &errors.CanidaeError{
		Code:        errors.ErrCodeConnectionFailed,
		Message:     "Connection failed",
		Details:     "Could not reach server",
		Severity:    errors.SeverityError,
		Retryable:   true,
		OriginalErr: originalErr,
	}
	
	// Test Error() method
	assert.Contains(t, err.Error(), "CANIDAE-1001")
	assert.Contains(t, err.Error(), "Connection failed")
	assert.Contains(t, err.Error(), "Could not reach server")
	
	// Test Unwrap
	assert.Equal(t, originalErr, err.Unwrap())
	
	// Test WithContext
	err.WithContext("server", "192.168.1.38:14001")
	err.WithContext("attempt", 3)
	assert.Equal(t, "192.168.1.38:14001", err.Context["server"])
	assert.Equal(t, 3, err.Context["attempt"])
	
	// Test WithDetails
	err.WithDetails("Additional details")
	assert.Equal(t, "Additional details", err.Details)
}

func TestErrorCreation(t *testing.T) {
	tests := []struct {
		name      string
		code      errors.ErrorCode
		retryable bool
		severity  errors.ErrorSeverity
	}{
		{
			name:      "connection error",
			code:      errors.ErrCodeConnectionFailed,
			retryable: true,
			severity:  errors.SeverityError,
		},
		{
			name:      "authentication error",
			code:      errors.ErrCodeAuthenticationFailed,
			retryable: false,
			severity:  errors.SeverityError,
		},
		{
			name:      "token expired",
			code:      errors.ErrCodeTokenExpired,
			retryable: true,
			severity:  errors.SeverityWarning,
		},
		{
			name:      "rate limit",
			code:      errors.ErrCodeRateLimitExceeded,
			retryable: true,
			severity:  errors.SeverityWarning,
		},
		{
			name:      "internal error",
			code:      errors.ErrCodeInternalError,
			retryable: false,
			severity:  errors.SeverityCritical,
		},
	}
	
	registry := errors.GetRegistry()
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalErr := fmt.Errorf("test error")
			err := registry.NewError(tt.code, originalErr)
			
			assert.Equal(t, tt.code, err.Code)
			assert.Equal(t, tt.retryable, err.Retryable)
			assert.Equal(t, tt.severity, err.Severity)
			assert.Equal(t, originalErr, err.OriginalErr)
		})
	}
}

func TestHelperFunctions(t *testing.T) {
	originalErr := fmt.Errorf("test error")
	
	t.Run("NewConnectionError", func(t *testing.T) {
		err := errors.NewConnectionError(originalErr)
		assert.Equal(t, errors.ErrCodeConnectionFailed, err.Code)
		assert.True(t, err.Retryable)
		assert.Equal(t, errors.SeverityError, err.Severity)
	})
	
	t.Run("NewAuthenticationError", func(t *testing.T) {
		err := errors.NewAuthenticationError(originalErr)
		assert.Equal(t, errors.ErrCodeAuthenticationFailed, err.Code)
		assert.False(t, err.Retryable)
		assert.Equal(t, errors.SeverityError, err.Severity)
	})
	
	t.Run("NewValidationError", func(t *testing.T) {
		err := errors.NewValidationError("username", originalErr)
		assert.Equal(t, errors.ErrCodeInvalidFieldValue, err.Code)
		assert.False(t, err.Retryable)
		assert.Equal(t, "username", err.Context["field"])
	})
	
	t.Run("NewResourceError", func(t *testing.T) {
		err := errors.NewResourceError("agent", originalErr)
		assert.Equal(t, errors.ErrCodeResourceNotFound, err.Code)
		assert.False(t, err.Retryable)
		assert.Equal(t, "agent", err.Context["resource"])
	})
	
	t.Run("NewInternalError", func(t *testing.T) {
		err := errors.NewInternalError(originalErr)
		assert.Equal(t, errors.ErrCodeInternalError, err.Code)
		assert.False(t, err.Retryable)
		assert.Equal(t, errors.SeverityCritical, err.Severity)
	})
}

func TestIsRetryable(t *testing.T) {
	tests := []struct {
		name      string
		err       error
		retryable bool
	}{
		{
			name:      "nil error",
			err:       nil,
			retryable: false,
		},
		{
			name:      "retryable error",
			err:       errors.NewConnectionError(fmt.Errorf("test")),
			retryable: true,
		},
		{
			name:      "non-retryable error",
			err:       errors.NewAuthenticationError(fmt.Errorf("test")),
			retryable: false,
		},
		{
			name:      "non-canidae error",
			err:       fmt.Errorf("regular error"),
			retryable: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.retryable, errors.IsRetryable(tt.err))
		})
	}
}

func TestGetErrorCode(t *testing.T) {
	tests := []struct {
		name string
		err  error
		code errors.ErrorCode
	}{
		{
			name: "nil error",
			err:  nil,
			code: "",
		},
		{
			name: "canidae error",
			err:  errors.NewConnectionError(fmt.Errorf("test")),
			code: errors.ErrCodeConnectionFailed,
		},
		{
			name: "non-canidae error",
			err:  fmt.Errorf("regular error"),
			code: "",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.code, errors.GetErrorCode(tt.err))
		})
	}
}

func TestCustomErrorRegistration(t *testing.T) {
	registry := errors.GetRegistry()
	
	// Register a custom error
	customCode := errors.ErrorCode("CUSTOM-001")
	registry.Register(customCode, "Custom error", errors.SeverityWarning, true)
	
	// Create error with custom code
	err := registry.NewError(customCode, fmt.Errorf("test"))
	assert.Equal(t, customCode, err.Code)
	assert.Equal(t, "Custom error", err.Message)
	assert.Equal(t, errors.SeverityWarning, err.Severity)
	assert.True(t, err.Retryable)
}

func TestUnknownErrorCode(t *testing.T) {
	registry := errors.GetRegistry()
	
	// Try to create error with unknown code
	unknownCode := errors.ErrorCode("UNKNOWN-999")
	err := registry.NewError(unknownCode, fmt.Errorf("test"))
	
	// Should fall back to internal error
	assert.Equal(t, errors.ErrCodeInternalError, err.Code)
	assert.Equal(t, "Unknown error code", err.Message)
	assert.Equal(t, errors.SeverityError, err.Severity)
	assert.False(t, err.Retryable)
}

func TestErrorCategories(t *testing.T) {
	// Test that error codes follow the category pattern
	tests := []struct {
		category string
		codes    []errors.ErrorCode
		prefix   string
	}{
		{
			category: "connection",
			codes: []errors.ErrorCode{
				errors.ErrCodeConnectionFailed,
				errors.ErrCodeConnectionTimeout,
				errors.ErrCodeTLSHandshakeFailed,
			},
			prefix: "CANIDAE-1",
		},
		{
			category: "authentication",
			codes: []errors.ErrorCode{
				errors.ErrCodeAuthenticationFailed,
				errors.ErrCodeInvalidCredentials,
				errors.ErrCodeTokenExpired,
			},
			prefix: "CANIDAE-2",
		},
		{
			category: "validation",
			codes: []errors.ErrorCode{
				errors.ErrCodeInvalidRequest,
				errors.ErrCodeMissingRequiredField,
				errors.ErrCodeInvalidFieldValue,
			},
			prefix: "CANIDAE-3",
		},
		{
			category: "resource",
			codes: []errors.ErrorCode{
				errors.ErrCodeResourceNotFound,
				errors.ErrCodeResourceUnavailable,
				errors.ErrCodeRateLimitExceeded,
			},
			prefix: "CANIDAE-4",
		},
		{
			category: "protocol",
			codes: []errors.ErrorCode{
				errors.ErrCodeProtocolViolation,
				errors.ErrCodeUnsupportedOperation,
				errors.ErrCodeMalformedMessage,
			},
			prefix: "CANIDAE-5",
		},
		{
			category: "internal",
			codes: []errors.ErrorCode{
				errors.ErrCodeInternalError,
				errors.ErrCodeConfigurationError,
				errors.ErrCodeSerializationError,
			},
			prefix: "CANIDAE-6",
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.category, func(t *testing.T) {
			for _, code := range tt.codes {
				assert.Contains(t, string(code), tt.prefix,
					"Error code %s should have prefix %s", code, tt.prefix)
			}
		})
	}
}

func TestErrorWithoutDetails(t *testing.T) {
	err := &errors.CanidaeError{
		Code:      errors.ErrCodeConnectionFailed,
		Message:   "Connection failed",
		Severity:  errors.SeverityError,
		Retryable: true,
	}
	
	// Test Error() method without details
	errStr := err.Error()
	assert.Contains(t, errStr, "CANIDAE-1001")
	assert.Contains(t, errStr, "Connection failed")
	assert.NotContains(t, errStr, ":")
}

func TestConcurrentErrorCreation(t *testing.T) {
	registry := errors.GetRegistry()
	
	// Test concurrent error creation
	done := make(chan bool, 100)
	for i := 0; i < 100; i++ {
		go func() {
			err := registry.NewError(errors.ErrCodeConnectionFailed, fmt.Errorf("test"))
			assert.NotNil(t, err)
			done <- true
		}()
	}
	
	// Wait for all goroutines
	for i := 0; i < 100; i++ {
		<-done
	}
}

// BenchmarkErrorCreation benchmarks error creation
func BenchmarkErrorCreation(b *testing.B) {
	registry := errors.GetRegistry()
	originalErr := fmt.Errorf("test error")
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = registry.NewError(errors.ErrCodeConnectionFailed, originalErr)
	}
}

// BenchmarkIsRetryable benchmarks retryable check
func BenchmarkIsRetryable(b *testing.B) {
	err := errors.NewConnectionError(fmt.Errorf("test"))
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = errors.IsRetryable(err)
	}
}

// BenchmarkGetErrorCode benchmarks error code extraction
func BenchmarkGetErrorCode(b *testing.B) {
	err := errors.NewConnectionError(fmt.Errorf("test"))
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = errors.GetErrorCode(err)
	}
}

func ExampleCanidaeError() {
	// Create a connection error
	err := errors.NewConnectionError(fmt.Errorf("dial tcp: connection refused"))
	
	// Add context
	err.WithContext("server", "192.168.1.38:14001").
		WithContext("attempt", 3).
		WithDetails("Failed after 3 retry attempts")
	
	// Check if retryable
	if errors.IsRetryable(err) {
		fmt.Println("Error is retryable")
	}
	
	// Get error code
	code := errors.GetErrorCode(err)
	fmt.Printf("Error code: %s\n", code)
}

func ExampleErrorRegistry_Register() {
	registry := errors.GetRegistry()
	
	// Register a custom error
	customCode := errors.ErrorCode("CUSTOM-001")
	registry.Register(
		customCode,
		"Custom operation failed",
		errors.SeverityWarning,
		true, // retryable
	)
	
	// Use the custom error
	err := registry.NewError(customCode, fmt.Errorf("underlying cause"))
	fmt.Printf("Custom error: %s\n", err.Error())
}