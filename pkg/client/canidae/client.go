// Package canidae provides the main client SDK for CANIDAE AI orchestration platform
package canidae

import (
	"context"
	"fmt"
	"time"

	"github.com/macawi-ai/canidae/pkg/client/auth"
	"github.com/macawi-ai/canidae/pkg/client/config"
	"github.com/macawi-ai/canidae/pkg/client/session"
	"github.com/macawi-ai/canidae/pkg/client/transport"
)

// Client represents the main CANIDAE client SDK
type Client struct {
	config    *config.Config
	transport transport.Client
	auth      auth.Provider
	session   session.Manager
	packID    string
}

// NewClient creates a new CANIDAE client instance
func NewClient(opts ...Option) (*Client, error) {
	cfg := config.DefaultConfig()
	
	// Apply options
	for _, opt := range opts {
		if err := opt(cfg); err != nil {
			return nil, fmt.Errorf("failed to apply option: %w", err)
		}
	}
	
	// Validate configuration
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}
	
	// Initialize transport
	transportClient, err := transport.New(cfg.TransportConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize transport: %w", err)
	}
	
	// Initialize auth provider
	authProvider, err := auth.NewProvider(cfg.AuthConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize auth: %w", err)
	}
	
	// Initialize session manager
	sessionManager, err := session.NewManager(cfg.SessionConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize session manager: %w", err)
	}
	
	return &Client{
		config:    cfg,
		transport: transportClient,
		auth:      authProvider,
		session:   sessionManager,
	}, nil
}

// Connect establishes a connection to the CANIDAE server
func (c *Client) Connect(ctx context.Context) error {
	// Authenticate
	token, err := c.auth.Authenticate(ctx)
	if err != nil {
		return fmt.Errorf("authentication failed: %w", err)
	}
	
	// Create session
	sess, err := c.session.CreateSession(ctx, token)
	if err != nil {
		return fmt.Errorf("failed to create session: %w", err)
	}
	
	// Set session in transport
	c.transport.SetSession(sess)
	
	// Connect transport
	if err := c.transport.Connect(ctx); err != nil {
		return fmt.Errorf("transport connection failed: %w", err)
	}
	
	return nil
}

// Disconnect closes the connection to the CANIDAE server
func (c *Client) Disconnect(ctx context.Context) error {
	// Revoke session
	if err := c.session.RevokeSession(ctx); err != nil {
		// Log error but continue disconnect
		_ = err
	}
	
	// Disconnect transport
	return c.transport.Disconnect(ctx)
}

// ExecuteAgent sends a request to execute a specific AI agent
func (c *Client) ExecuteAgent(ctx context.Context, req *ExecuteRequest) (*ExecuteResponse, error) {
	// Validate request
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("invalid request: %w", err)
	}
	
	// Send via transport
	resp, err := c.transport.Send(ctx, &transport.Request{
		Type:    transport.RequestTypeExecute,
		Payload: req,
	})
	if err != nil {
		return nil, fmt.Errorf("transport error: %w", err)
	}
	
	// Parse response
	result := &ExecuteResponse{}
	if err := resp.UnmarshalTo(result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	return result, nil
}

// ChainAgents executes multiple agents in sequence
func (c *Client) ChainAgents(ctx context.Context, req *ChainRequest) (*ChainResponse, error) {
	// Validate request
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("invalid chain request: %w", err)
	}
	
	// Send via transport
	resp, err := c.transport.Send(ctx, &transport.Request{
		Type:    transport.RequestTypeChain,
		Payload: req,
	})
	if err != nil {
		return nil, fmt.Errorf("transport error: %w", err)
	}
	
	// Parse response
	result := &ChainResponse{}
	if err := resp.UnmarshalTo(result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	return result, nil
}

// SummonPack creates a new pack formation for parallel processing
func (c *Client) SummonPack(ctx context.Context, req *PackRequest) (*PackResponse, error) {
	// Validate request
	if err := req.Validate(); err != nil {
		return nil, fmt.Errorf("invalid pack request: %w", err)
	}
	
	// Send via transport
	resp, err := c.transport.Send(ctx, &transport.Request{
		Type:    transport.RequestTypePack,
		Payload: req,
	})
	if err != nil {
		return nil, fmt.Errorf("transport error: %w", err)
	}
	
	// Parse response
	result := &PackResponse{}
	if err := resp.UnmarshalTo(result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	return result, nil
}

// Stream opens a streaming connection for real-time agent communication
func (c *Client) Stream(ctx context.Context, handler StreamHandler) error {
	return c.transport.Stream(ctx, handler)
}

// GetStatus returns the current client and connection status
func (c *Client) GetStatus() *Status {
	return &Status{
		Connected:   c.transport.IsConnected(),
		Authenticated: c.session.IsValid(),
		PackID:      c.packID,
		ServerEndpoint: c.config.ServerEndpoint,
		LastActivity: time.Now(),
	}
}

// SetPackID sets the pack identifier for multi-tenant isolation
func (c *Client) SetPackID(packID string) {
	c.packID = packID
	c.transport.SetHeader("X-Pack-ID", packID)
}