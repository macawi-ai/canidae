package transport

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"log"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"

	howlv1 "github.com/macawi-ai/canidae/api/howl/v1"
	"github.com/macawi-ai/canidae/pkg/client/config"
)

// grpcClient implements the Client interface using gRPC
type grpcClient struct {
	config  config.TransportConfig
	conn    *grpc.ClientConn
	client  howlv1.CanidaeServiceClient
	session interface{}
	headers map[string]string
	metrics *Metrics

	connected atomic.Bool
	mu        sync.RWMutex

	// Stream management
	streamCtx    context.Context
	streamCancel context.CancelFunc
}

// NewGRPCTransport creates a new gRPC transport client
func NewGRPCTransport(cfg config.TransportConfig) (Client, error) {
	return &grpcClient{
		config:  cfg,
		headers: make(map[string]string),
		metrics: &Metrics{
			LastActivity: time.Now(),
		},
	}, nil
}

// Connect establishes a gRPC connection to the server
func (c *grpcClient) Connect(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.connected.Load() {
		return nil // Already connected
	}

	// Build dial options
	opts, err := c.buildDialOptions()
	if err != nil {
		return fmt.Errorf("failed to build dial options: %w", err)
	}

	// Extract server address from config
	// Assuming config has the server endpoint
	serverAddr := c.getServerAddress()

	// Establish connection using the new API (grpc.NewClient replaces grpc.DialContext)
	conn, err := grpc.NewClient(serverAddr, opts...)
	if err != nil {
		return fmt.Errorf("failed to create gRPC client: %w", err)
	}

	// Wait for connection to be ready
	connCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	for {
		state := conn.GetState()
		if state == connectivity.Ready {
			break
		}
		if state == connectivity.TransientFailure || state == connectivity.Shutdown {
			if err := conn.Close(); err != nil {
				log.Printf("Error closing connection after failure: %v", err)
			}
			return fmt.Errorf("failed to establish connection: %s", state)
		}
		if !conn.WaitForStateChange(connCtx, state) {
			if err := conn.Close(); err != nil {
				log.Printf("Error closing connection after timeout: %v", err)
			}
			return fmt.Errorf("timeout waiting for connection")
		}
	}

	c.conn = conn
	c.client = howlv1.NewCanidaeServiceClient(conn)
	c.connected.Store(true)

	atomic.AddInt64(&c.metrics.RequestsSent, 1)

	return nil
}

// Disconnect closes the gRPC connection
func (c *grpcClient) Disconnect(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.connected.Load() {
		return nil // Already disconnected
	}

	if c.streamCancel != nil {
		c.streamCancel()
		c.streamCancel = nil
	}

	if c.conn != nil {
		err := c.conn.Close()
		c.conn = nil
		c.client = nil
		c.connected.Store(false)
		return err
	}

	return nil
}

// Send sends a request and waits for response
func (c *grpcClient) Send(ctx context.Context, req *Request) (*Response, error) {
	if !c.connected.Load() {
		return nil, ErrNotConnected
	}

	// Add metadata from headers
	ctx = c.attachMetadata(ctx)

	// Apply timeout if specified
	if req.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, req.Timeout)
		defer cancel()
	}

	// Convert to protobuf request
	pbReq, err := c.toPBRequest(req)
	if err != nil {
		return nil, fmt.Errorf("failed to convert request: %w", err)
	}

	// Record metrics
	startTime := time.Now()
	atomic.AddInt64(&c.metrics.RequestsSent, 1)

	// Send request based on type
	var pbResp *howlv1.Response
	switch req.Type {
	case RequestTypeExecute:
		pbResp, err = c.client.Execute(ctx, pbReq)
	case RequestTypeChain:
		pbResp, err = c.client.Chain(ctx, pbReq)
	case RequestTypePack:
		pbResp, err = c.client.SummonPack(ctx, pbReq)
	default:
		return nil, fmt.Errorf("unsupported request type: %s", req.Type)
	}

	// Update metrics
	duration := time.Since(startTime)
	c.updateMetrics(duration, err)

	if err != nil {
		atomic.AddInt64(&c.metrics.ErrorCount, 1)
		return nil, fmt.Errorf("gRPC call failed: %w", err)
	}

	atomic.AddInt64(&c.metrics.ResponsesReceived, 1)

	// Convert from protobuf response
	return c.fromPBResponse(pbResp)
}

// Stream opens a streaming connection
func (c *grpcClient) Stream(ctx context.Context, handler StreamHandler) error {
	if !c.connected.Load() {
		return ErrNotConnected
	}

	// Cancel any existing stream
	if c.streamCancel != nil {
		c.streamCancel()
	}

	// Create stream context
	c.streamCtx, c.streamCancel = context.WithCancel(ctx)
	ctx = c.attachMetadata(c.streamCtx)

	// Open bidirectional stream
	stream, err := c.client.Stream(ctx)
	if err != nil {
		return fmt.Errorf("failed to open stream: %w", err)
	}

	// Handle stream events
	go c.handleStream(stream, handler)

	// Send initial connect event
	if err := handler(&StreamEvent{
		Type: StreamEventTypeData,
		Metadata: map[string]string{
			"connected": "true",
		},
	}); err != nil {
		return err
	}

	return nil
}

// IsConnected returns true if connected
func (c *grpcClient) IsConnected() bool {
	return c.connected.Load() && c.conn != nil && c.conn.GetState() == connectivity.Ready
}

// SetSession sets the session token
func (c *grpcClient) SetSession(session interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.session = session

	// Add session token to headers
	if token, ok := session.(string); ok {
		c.headers["authorization"] = "Bearer " + token
	}
}

// SetHeader sets a custom header
func (c *grpcClient) SetHeader(key, value string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.headers[key] = value
}

// GetMetrics returns transport metrics
func (c *grpcClient) GetMetrics() *Metrics {
	return c.metrics
}

// buildDialOptions builds gRPC dial options
func (c *grpcClient) buildDialOptions() ([]grpc.DialOption, error) {
	opts := []grpc.DialOption{
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                30 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(c.config.MaxMessageSize),
			grpc.MaxCallSendMsgSize(c.config.MaxMessageSize),
		),
	}

	// Configure TLS
	if c.config.TLS.Enabled {
		tlsConfig := &tls.Config{
			ServerName: c.config.TLS.ServerName,
			MinVersion: tls.VersionTLS12, // Enforce minimum TLS 1.2
		}
		
		// SECURITY: Only allow InsecureSkipVerify in development/debug mode
		// This is a HIGH security risk and should never be used in production
		if c.config.TLS.InsecureSkipVerify {
			// Log a warning about the security risk
			log.Printf("WARNING: TLS certificate verification is DISABLED. This is a security risk and should only be used in development.")
			log.Printf("WARNING: Set TLS.InsecureSkipVerify to false for production use.")
			
			// Only allow in debug mode or with explicit environment variable
			debugMode := os.Getenv("CANIDAE_DEBUG_MODE") == "true"
			allowInsecure := os.Getenv("CANIDAE_ALLOW_INSECURE_TLS") == "true"
			
			if !debugMode && !allowInsecure {
				return nil, fmt.Errorf("TLS InsecureSkipVerify is not allowed in production. Set CANIDAE_DEBUG_MODE=true or CANIDAE_ALLOW_INSECURE_TLS=true to override (NOT RECOMMENDED)")
			}
			
			tlsConfig.InsecureSkipVerify = true
		}

		// Load client certificates for mTLS
		if c.config.TLS.CertFile != "" && c.config.TLS.KeyFile != "" {
			cert, err := tls.LoadX509KeyPair(c.config.TLS.CertFile, c.config.TLS.KeyFile)
			if err != nil {
				return nil, fmt.Errorf("failed to load client certificates: %w", err)
			}
			tlsConfig.Certificates = []tls.Certificate{cert}
		}
		
		// Load CA certificate if provided
		if c.config.TLS.CAFile != "" {
			caCert, err := os.ReadFile(c.config.TLS.CAFile)
			if err != nil {
				return nil, fmt.Errorf("failed to read CA certificate: %w", err)
			}
			
			caCertPool := x509.NewCertPool()
			if !caCertPool.AppendCertsFromPEM(caCert) {
				return nil, fmt.Errorf("failed to parse CA certificate")
			}
			tlsConfig.RootCAs = caCertPool
		}

		opts = append(opts, grpc.WithTransportCredentials(credentials.NewTLS(tlsConfig)))
	} else {
		// Log warning about using insecure connection
		log.Printf("WARNING: TLS is DISABLED. Connection is not encrypted. Use TLS in production.")
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	}

	// Add retry interceptor if enabled
	if c.config.Retry.Enabled {
		opts = append(opts, grpc.WithUnaryInterceptor(c.retryInterceptor))
	}

	return opts, nil
}

// retryInterceptor implements retry logic
func (c *grpcClient) retryInterceptor(
	ctx context.Context,
	method string,
	req, reply interface{},
	cc *grpc.ClientConn,
	invoker grpc.UnaryInvoker,
	opts ...grpc.CallOption,
) error {
	var lastErr error
	backoff := c.config.Retry.Backoff

	for attempt := 0; attempt < c.config.Retry.MaxAttempts; attempt++ {
		err := invoker(ctx, method, req, reply, cc, opts...)
		if err == nil {
			return nil
		}

		lastErr = err

		// Don't retry on certain errors
		if !isRetryableError(err) {
			return err
		}

		// Wait before retry
		if attempt < c.config.Retry.MaxAttempts-1 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(backoff):
				// Exponential backoff
				backoff *= 2
				if backoff > c.config.Retry.MaxBackoff {
					backoff = c.config.Retry.MaxBackoff
				}
			}
		}
	}

	return lastErr
}

// attachMetadata attaches headers as gRPC metadata
func (c *grpcClient) attachMetadata(ctx context.Context) context.Context {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.headers) > 0 {
		md := metadata.New(c.headers)
		ctx = metadata.NewOutgoingContext(ctx, md)
	}

	return ctx
}

// handleStream handles streaming events
func (c *grpcClient) handleStream(stream howlv1.CanidaeService_StreamClient, handler StreamHandler) {
	defer func() {
		if c.streamCancel != nil {
			c.streamCancel()
			c.streamCancel = nil
		}
	}()

	for {
		msg, err := stream.Recv()
		if err != nil {
			// Check if context was cancelled
			if errors.Is(err, context.Canceled) {
				if handlerErr := handler(&StreamEvent{
					Type: StreamEventTypeComplete,
				}); handlerErr != nil {
					// Log the handler error but don't propagate since stream is closing
					log.Printf("Error in stream handler during completion: %v", handlerErr)
				}
				return
			}

			// Send error event
			if handlerErr := handler(&StreamEvent{
				Type: StreamEventTypeError,
				Error: &Error{
					Message: err.Error(),
				},
			}); handlerErr != nil {
				// Log the handler error
				log.Printf("Error in stream handler during error event: %v", handlerErr)
			}
			return
		}

		// Convert and handle message
		event := c.pbToStreamEvent(msg)
		if err := handler(event); err != nil {
			return
		}
	}
}

// getServerAddress extracts server address from config
func (c *grpcClient) getServerAddress() string {
	// This would typically come from config
	// For now, return a default
	return "192.168.1.38:14001"
}

// updateMetrics updates transport metrics
func (c *grpcClient) updateMetrics(duration time.Duration, err error) {
	c.metrics.LastActivity = time.Now()

	// Update average latency
	if c.metrics.AverageLatency == 0 {
		c.metrics.AverageLatency = duration
	} else {
		// Simple moving average
		c.metrics.AverageLatency = (c.metrics.AverageLatency + duration) / 2
	}

	if err != nil {
		atomic.AddInt64(&c.metrics.ErrorCount, 1)
	}
}

// Helper functions for conversion between internal types and protobuf
// These would need to be implemented based on your protobuf definitions

func (c *grpcClient) toPBRequest(req *Request) (*howlv1.Request, error) {
	// Convert internal Request to protobuf Request
	// This is a placeholder implementation
	return &howlv1.Request{
		Id:      req.ID,
		Type:    string(req.Type),
		Headers: req.Headers,
	}, nil
}

func (c *grpcClient) fromPBResponse(pbResp *howlv1.Response) (*Response, error) {
	// Convert protobuf Response to internal Response
	// This is a placeholder implementation
	return &Response{
		ID:      pbResp.Id,
		Success: pbResp.Success,
		Headers: pbResp.Headers,
	}, nil
}

func (c *grpcClient) pbToStreamEvent(msg *howlv1.StreamMessage) *StreamEvent {
	// Convert protobuf StreamMessage to StreamEvent
	// This is a placeholder implementation
	return &StreamEvent{
		Type:     StreamEventTypeData,
		Metadata: msg.Metadata,
	}
}

func isRetryableError(err error) bool {
	// Define which errors are retryable
	// This is a simplified check
	return true
}
