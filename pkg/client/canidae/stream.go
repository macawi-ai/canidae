package canidae

// StreamEventType represents the type of streaming event
type StreamEventType string

const (
	StreamEventTypeConnect    StreamEventType = "connect"
	StreamEventTypeData       StreamEventType = "data"
	StreamEventTypeError      StreamEventType = "error"
	StreamEventTypeComplete   StreamEventType = "complete"
	StreamEventTypeHeartbeat  StreamEventType = "heartbeat"
	StreamEventTypeProgress   StreamEventType = "progress"
)