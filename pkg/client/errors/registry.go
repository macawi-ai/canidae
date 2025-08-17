package errors

import (
	"fmt"
	"sync"
)

// ErrorCode represents a unique error code
type ErrorCode string

// Error codes - organized by category
const (
	// Connection errors (1xxx)
	ErrCodeConnectionFailed    ErrorCode = "CANIDAE-1001"
	ErrCodeConnectionTimeout   ErrorCode = "CANIDAE-1002"
	ErrCodeConnectionRefused   ErrorCode = "CANIDAE-1003"
	ErrCodeTLSHandshakeFailed  ErrorCode = "CANIDAE-1004"
	ErrCodeDNSResolutionFailed ErrorCode = "CANIDAE-1005"
	ErrCodeNetworkUnreachable  ErrorCode = "CANIDAE-1006"
	
	// Authentication errors (2xxx)
	ErrCodeAuthenticationFailed   ErrorCode = "CANIDAE-2001"
	ErrCodeInvalidCredentials      ErrorCode = "CANIDAE-2002"
	ErrCodeTokenExpired            ErrorCode = "CANIDAE-2003"
	ErrCodeInsufficientPermissions ErrorCode = "CANIDAE-2004"
	ErrCodeMFARequired             ErrorCode = "CANIDAE-2005"
	ErrCodeWebAuthnChallengeFailed ErrorCode = "CANIDAE-2006"
	ErrCodeOAuthFlowFailed         ErrorCode = "CANIDAE-2007"
	
	// Validation errors (3xxx)
	ErrCodeInvalidRequest     ErrorCode = "CANIDAE-3001"
	ErrCodeMissingRequiredField ErrorCode = "CANIDAE-3002"
	ErrCodeInvalidFieldValue   ErrorCode = "CANIDAE-3003"
	ErrCodeRequestTooLarge     ErrorCode = "CANIDAE-3004"
	ErrCodeInvalidFormat       ErrorCode = "CANIDAE-3005"
	
	// Resource errors (4xxx)
	ErrCodeResourceNotFound    ErrorCode = "CANIDAE-4001"
	ErrCodeResourceUnavailable ErrorCode = "CANIDAE-4002"
	ErrCodeResourceExhausted   ErrorCode = "CANIDAE-4003"
	ErrCodeRateLimitExceeded   ErrorCode = "CANIDAE-4004"
	ErrCodeQuotaExceeded       ErrorCode = "CANIDAE-4005"
	
	// Protocol errors (5xxx)
	ErrCodeProtocolViolation    ErrorCode = "CANIDAE-5001"
	ErrCodeUnsupportedOperation  ErrorCode = "CANIDAE-5002"
	ErrCodeIncompatibleVersion   ErrorCode = "CANIDAE-5003"
	ErrCodeMalformedMessage      ErrorCode = "CANIDAE-5004"
	ErrCodeStreamInterrupted     ErrorCode = "CANIDAE-5005"
	
	// Internal errors (6xxx)
	ErrCodeInternalError      ErrorCode = "CANIDAE-6001"
	ErrCodeConfigurationError ErrorCode = "CANIDAE-6002"
	ErrCodeSerializationError ErrorCode = "CANIDAE-6003"
	ErrCodeStateInconsistent  ErrorCode = "CANIDAE-6004"
	ErrCodeNotImplemented     ErrorCode = "CANIDAE-6005"
)

// ErrorSeverity represents the severity of an error
type ErrorSeverity string

const (
	SeverityCritical ErrorSeverity = "CRITICAL" // System failure, immediate action required
	SeverityError    ErrorSeverity = "ERROR"    // Operation failed, manual intervention may be needed
	SeverityWarning  ErrorSeverity = "WARNING"  // Operation succeeded with issues
	SeverityInfo     ErrorSeverity = "INFO"     // Informational, no action required
)

// CanidaeError represents a structured error with metadata
type CanidaeError struct {
	Code        ErrorCode              `json:"code"`
	Message     string                 `json:"message"`
	Details     string                 `json:"details,omitempty"`
	Severity    ErrorSeverity          `json:"severity"`
	Retryable   bool                   `json:"retryable"`
	Context     map[string]interface{} `json:"context,omitempty"`
	OriginalErr error                  `json:"-"`
	StackTrace  string                 `json:"stack_trace,omitempty"`
}

// Error implements the error interface
func (e *CanidaeError) Error() string {
	if e.Details != "" {
		return fmt.Sprintf("[%s] %s: %s", e.Code, e.Message, e.Details)
	}
	return fmt.Sprintf("[%s] %s", e.Code, e.Message)
}

// Unwrap returns the original error
func (e *CanidaeError) Unwrap() error {
	return e.OriginalErr
}

// WithContext adds context to the error
func (e *CanidaeError) WithContext(key string, value interface{}) *CanidaeError {
	if e.Context == nil {
		e.Context = make(map[string]interface{})
	}
	e.Context[key] = value
	return e
}

// WithDetails adds details to the error
func (e *CanidaeError) WithDetails(details string) *CanidaeError {
	e.Details = details
	return e
}

// ErrorRegistry manages error definitions
type ErrorRegistry struct {
	mu         sync.RWMutex
	errors     map[ErrorCode]*ErrorDefinition
}

// ErrorDefinition defines an error template
type ErrorDefinition struct {
	Code      ErrorCode
	Message   string
	Severity  ErrorSeverity
	Retryable bool
}

var (
	// Global registry instance
	registry *ErrorRegistry
	once     sync.Once
)

// GetRegistry returns the global error registry
func GetRegistry() *ErrorRegistry {
	once.Do(func() {
		registry = &ErrorRegistry{
			errors: make(map[ErrorCode]*ErrorDefinition),
		}
		registry.registerDefaultErrors()
	})
	return registry
}

// registerDefaultErrors registers all default error definitions
func (r *ErrorRegistry) registerDefaultErrors() {
	// Connection errors
	r.Register(ErrCodeConnectionFailed, "Connection failed", SeverityError, true)
	r.Register(ErrCodeConnectionTimeout, "Connection timeout", SeverityError, true)
	r.Register(ErrCodeConnectionRefused, "Connection refused", SeverityError, true)
	r.Register(ErrCodeTLSHandshakeFailed, "TLS handshake failed", SeverityError, false)
	r.Register(ErrCodeDNSResolutionFailed, "DNS resolution failed", SeverityError, true)
	r.Register(ErrCodeNetworkUnreachable, "Network unreachable", SeverityError, true)
	
	// Authentication errors
	r.Register(ErrCodeAuthenticationFailed, "Authentication failed", SeverityError, false)
	r.Register(ErrCodeInvalidCredentials, "Invalid credentials", SeverityError, false)
	r.Register(ErrCodeTokenExpired, "Token expired", SeverityWarning, true)
	r.Register(ErrCodeInsufficientPermissions, "Insufficient permissions", SeverityError, false)
	r.Register(ErrCodeMFARequired, "MFA required", SeverityWarning, false)
	r.Register(ErrCodeWebAuthnChallengeFailed, "WebAuthn challenge failed", SeverityError, false)
	r.Register(ErrCodeOAuthFlowFailed, "OAuth flow failed", SeverityError, false)
	
	// Validation errors
	r.Register(ErrCodeInvalidRequest, "Invalid request", SeverityError, false)
	r.Register(ErrCodeMissingRequiredField, "Missing required field", SeverityError, false)
	r.Register(ErrCodeInvalidFieldValue, "Invalid field value", SeverityError, false)
	r.Register(ErrCodeRequestTooLarge, "Request too large", SeverityError, false)
	r.Register(ErrCodeInvalidFormat, "Invalid format", SeverityError, false)
	
	// Resource errors
	r.Register(ErrCodeResourceNotFound, "Resource not found", SeverityError, false)
	r.Register(ErrCodeResourceUnavailable, "Resource unavailable", SeverityError, true)
	r.Register(ErrCodeResourceExhausted, "Resource exhausted", SeverityError, true)
	r.Register(ErrCodeRateLimitExceeded, "Rate limit exceeded", SeverityWarning, true)
	r.Register(ErrCodeQuotaExceeded, "Quota exceeded", SeverityError, false)
	
	// Protocol errors
	r.Register(ErrCodeProtocolViolation, "Protocol violation", SeverityError, false)
	r.Register(ErrCodeUnsupportedOperation, "Unsupported operation", SeverityError, false)
	r.Register(ErrCodeIncompatibleVersion, "Incompatible version", SeverityError, false)
	r.Register(ErrCodeMalformedMessage, "Malformed message", SeverityError, false)
	r.Register(ErrCodeStreamInterrupted, "Stream interrupted", SeverityError, true)
	
	// Internal errors
	r.Register(ErrCodeInternalError, "Internal error", SeverityCritical, false)
	r.Register(ErrCodeConfigurationError, "Configuration error", SeverityCritical, false)
	r.Register(ErrCodeSerializationError, "Serialization error", SeverityError, false)
	r.Register(ErrCodeStateInconsistent, "State inconsistent", SeverityCritical, false)
	r.Register(ErrCodeNotImplemented, "Not implemented", SeverityError, false)
}

// Register adds a new error definition to the registry
func (r *ErrorRegistry) Register(code ErrorCode, message string, severity ErrorSeverity, retryable bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	r.errors[code] = &ErrorDefinition{
		Code:      code,
		Message:   message,
		Severity:  severity,
		Retryable: retryable,
	}
}

// NewError creates a new error from a code
func (r *ErrorRegistry) NewError(code ErrorCode, originalErr error) *CanidaeError {
	r.mu.RLock()
	def, exists := r.errors[code]
	r.mu.RUnlock()
	
	if !exists {
		return &CanidaeError{
			Code:        ErrCodeInternalError,
			Message:     "Unknown error code",
			Severity:    SeverityError,
			Retryable:   false,
			OriginalErr: originalErr,
		}
	}
	
	return &CanidaeError{
		Code:        def.Code,
		Message:     def.Message,
		Severity:    def.Severity,
		Retryable:   def.Retryable,
		OriginalErr: originalErr,
	}
}

// Helper functions for common error creation

// NewConnectionError creates a connection error
func NewConnectionError(originalErr error) *CanidaeError {
	return GetRegistry().NewError(ErrCodeConnectionFailed, originalErr)
}

// NewAuthenticationError creates an authentication error
func NewAuthenticationError(originalErr error) *CanidaeError {
	return GetRegistry().NewError(ErrCodeAuthenticationFailed, originalErr)
}

// NewValidationError creates a validation error
func NewValidationError(field string, originalErr error) *CanidaeError {
	err := GetRegistry().NewError(ErrCodeInvalidFieldValue, originalErr)
	return err.WithContext("field", field)
}

// NewResourceError creates a resource error
func NewResourceError(resource string, originalErr error) *CanidaeError {
	err := GetRegistry().NewError(ErrCodeResourceNotFound, originalErr)
	return err.WithContext("resource", resource)
}

// NewInternalError creates an internal error
func NewInternalError(originalErr error) *CanidaeError {
	return GetRegistry().NewError(ErrCodeInternalError, originalErr)
}

// IsRetryable checks if an error is retryable
func IsRetryable(err error) bool {
	if err == nil {
		return false
	}
	
	canidaeErr, ok := err.(*CanidaeError)
	if !ok {
		return false
	}
	
	return canidaeErr.Retryable
}

// GetErrorCode extracts the error code from an error
func GetErrorCode(err error) ErrorCode {
	if err == nil {
		return ""
	}
	
	canidaeErr, ok := err.(*CanidaeError)
	if !ok {
		return ""
	}
	
	return canidaeErr.Code
}