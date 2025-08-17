package transport

import (
	"context"
	"encoding/json"
	"errors"
	"time"
	
	"github.com/macawi-ai/canidae/pkg/client/config"
)

// Common errors
var (
	ErrNotConnected    = errors.New("transport not connected")
	ErrTimeout         = errors.New("request timeout")
	ErrInvalidResponse = errors.New("invalid response")
)

// RequestType represents the type of request
type RequestType string

const (
	RequestTypeExecute RequestType = "execute"
	RequestTypeChain   RequestType = "chain"
	RequestTypePack    RequestType = "pack"
	RequestTypeStream  RequestType = "stream"
	RequestTypeControl RequestType = "control"
)

// Client represents the transport client interface
type Client interface {
	// Connect establishes a connection to the server
	Connect(ctx context.Context) error
	
	// Disconnect closes the connection
	Disconnect(ctx context.Context) error
	
	// Send sends a request and waits for response
	Send(ctx context.Context, req *Request) (*Response, error)
	
	// Stream opens a streaming connection
	Stream(ctx context.Context, handler StreamHandler) error
	
	// IsConnected returns true if connected
	IsConnected() bool
	
	// SetSession sets the session token
	SetSession(session interface{})
	
	// SetHeader sets a custom header
	SetHeader(key, value string)
	
	// GetMetrics returns transport metrics
	GetMetrics() *Metrics
}

// Request represents a transport request
type Request struct {
	ID       string            `json:"id"`
	Type     RequestType       `json:"type"`
	Payload  interface{}       `json:"payload"`
	Headers  map[string]string `json:"headers,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
	Timeout  time.Duration     `json:"timeout,omitempty"`
}

// Response represents a transport response
type Response struct {
	ID       string            `json:"id"`
	Success  bool              `json:"success"`
	Data     json.RawMessage   `json:"data,omitempty"`
	Error    *Error            `json:"error,omitempty"`
	Headers  map[string]string `json:"headers,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// UnmarshalTo unmarshals the response data to the given target
func (r *Response) UnmarshalTo(target interface{}) error {
	if r.Data == nil {
		return errors.New("no data in response")
	}
	return json.Unmarshal(r.Data, target)
}

// Error represents a transport error
type Error struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
}

// Error implements the error interface
func (e *Error) Error() string {
	if e.Details != "" {
		return e.Message + ": " + e.Details
	}
	return e.Message
}

// StreamHandler handles streaming events
type StreamHandler func(event *StreamEvent) error

// StreamEvent represents a streaming event
type StreamEvent struct {
	Type     StreamEventType   `json:"type"`
	Data     json.RawMessage   `json:"data"`
	Error    *Error            `json:"error,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// StreamEventType represents the type of streaming event
type StreamEventType string

const (
	StreamEventTypeData     StreamEventType = "data"
	StreamEventTypeError    StreamEventType = "error"
	StreamEventTypeComplete StreamEventType = "complete"
	StreamEventTypeHeartbeat StreamEventType = "heartbeat"
)

// Metrics represents transport metrics
type Metrics struct {
	RequestsSent     int64         `json:"requests_sent"`
	ResponsesReceived int64        `json:"responses_received"`
	ErrorCount       int64         `json:"error_count"`
	BytesSent        int64         `json:"bytes_sent"`
	BytesReceived    int64         `json:"bytes_received"`
	AverageLatency   time.Duration `json:"average_latency"`
	LastActivity     time.Time     `json:"last_activity"`
}

// New creates a new transport client based on configuration
func New(cfg config.TransportConfig) (Client, error) {
	switch cfg.Type {
	case "grpc":
		return NewGRPCClient(cfg)
	case "grpcweb":
		return NewGRPCWebClient(cfg)
	case "http":
		return NewHTTPClient(cfg)
	default:
		return nil, errors.New("unsupported transport type: " + cfg.Type)
	}
}

// NewGRPCClient creates a new gRPC client (to be implemented)
func NewGRPCClient(cfg config.TransportConfig) (Client, error) {
	// This will be implemented in grpc.go
	return nil, errors.New("gRPC transport not yet implemented")
}

// NewGRPCWebClient creates a new gRPC-Web client (to be implemented)
func NewGRPCWebClient(cfg config.TransportConfig) (Client, error) {
	// This will be implemented in grpcweb.go
	return nil, errors.New("gRPC-Web transport not yet implemented")
}

// NewHTTPClient creates a new HTTP client (to be implemented)
func NewHTTPClient(cfg config.TransportConfig) (Client, error) {
	// This will be implemented in http.go
	return nil, errors.New("HTTP transport not yet implemented")
}