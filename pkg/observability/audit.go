package observability

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"go.opentelemetry.io/otel/trace"
)

// AuditEvent represents an auditable event in the system
type AuditEvent struct {
	Timestamp   time.Time              `json:"timestamp"`
	EventType   string                 `json:"event_type"`
	Actor       string                 `json:"actor"`
	PackID      string                 `json:"pack_id,omitempty"`
	Resource    string                 `json:"resource"`
	Action      string                 `json:"action"`
	Result      string                 `json:"result"`
	TraceID     string                 `json:"trace_id,omitempty"`
	SpanID      string                 `json:"span_id,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	IPAddress   string                 `json:"ip_address,omitempty"`
	UserAgent   string                 `json:"user_agent,omitempty"`
	Severity    string                 `json:"severity"`
	Compliance  []string               `json:"compliance_tags,omitempty"`
}

// AuditLogger handles secure audit logging
type AuditLogger struct {
	file       *os.File
	encoder    *json.Encoder
	mu         sync.Mutex
	rotateSize int64
	currentSize int64
}

// NewAuditLogger creates a new audit logger
func NewAuditLogger(path string) (*AuditLogger, error) {
	// Ensure audit log directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0700); err != nil {
		return nil, fmt.Errorf("failed to create audit log directory: %w", err)
	}

	// Open audit log file with secure permissions
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return nil, fmt.Errorf("failed to open audit log: %w", err)
	}

	// Get current file size for rotation
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to stat audit log: %w", err)
	}

	return &AuditLogger{
		file:        file,
		encoder:     json.NewEncoder(file),
		rotateSize:  100 * 1024 * 1024, // 100MB default rotation size
		currentSize: info.Size(),
	}, nil
}

// Log writes an audit event
func (al *AuditLogger) Log(ctx context.Context, event AuditEvent) {
	al.mu.Lock()
	defer al.mu.Unlock()

	// Add trace context if available
	if span := trace.SpanFromContext(ctx); span.SpanContext().IsValid() {
		event.TraceID = span.SpanContext().TraceID().String()
		event.SpanID = span.SpanContext().SpanID().String()
	}

	// Set timestamp if not provided
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now().UTC()
	}

	// Write event
	if err := al.encoder.Encode(event); err != nil {
		// Log to stderr if audit logging fails (critical)
		fmt.Fprintf(os.Stderr, "AUDIT LOG FAILURE: %v - Event: %+v\n", err, event)
		return
	}

	// Update size counter
	eventSize := int64(len(event.EventType) + len(event.Actor) + 100) // Approximate
	al.currentSize += eventSize

	// Check if rotation is needed
	if al.currentSize >= al.rotateSize {
		al.rotate()
	}
}

// rotate performs audit log rotation
func (al *AuditLogger) rotate() {
	// Close current file
	al.file.Close()

	// Rename current file with timestamp
	timestamp := time.Now().Format("20060102-150405")
	oldPath := al.file.Name()
	newPath := fmt.Sprintf("%s.%s", oldPath, timestamp)
	
	if err := os.Rename(oldPath, newPath); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to rotate audit log: %v\n", err)
		return
	}

	// Open new file
	file, err := os.OpenFile(oldPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create new audit log: %v\n", err)
		return
	}

	al.file = file
	al.encoder = json.NewEncoder(file)
	al.currentSize = 0
}

// Close closes the audit logger
func (al *AuditLogger) Close() error {
	al.mu.Lock()
	defer al.mu.Unlock()
	return al.file.Close()
}

// Critical audit events that MUST be logged
const (
	EventAuthentication = "AUTHENTICATION"
	EventAuthorization  = "AUTHORIZATION"
	EventDataAccess     = "DATA_ACCESS"
	EventDataModify     = "DATA_MODIFY"
	EventConfigChange   = "CONFIG_CHANGE"
	EventSecurityAlert  = "SECURITY_ALERT"
	EventPackCreate     = "PACK_CREATE"
	EventPackDelete     = "PACK_DELETE"
	EventBillingChange  = "BILLING_CHANGE"
	EventAPIAccess      = "API_ACCESS"
)

// Severity levels for audit events
const (
	SeverityInfo     = "INFO"
	SeverityWarning  = "WARNING"
	SeverityError    = "ERROR"
	SeverityCritical = "CRITICAL"
)

// Compliance tags for regulatory requirements
const (
	ComplianceSOC2   = "SOC2"
	ComplianceHIPAA  = "HIPAA"
	ComplianceGDPR   = "GDPR"
	CompliancePCI    = "PCI-DSS"
	ComplianceISO27001 = "ISO27001"
)