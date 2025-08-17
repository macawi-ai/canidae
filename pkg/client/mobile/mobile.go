// Package mobile provides a simplified interface for mobile platforms (iOS/Android)
// This package is designed to be used with gomobile bind
package mobile

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"time"
	
	"github.com/macawi-ai/canidae/pkg/client/canidae"
)

// CanidaeClient is a mobile-friendly wrapper around the CANIDAE SDK
type CanidaeClient struct {
	client *canidae.Client
	ctx    context.Context
	cancel context.CancelFunc
	mu     sync.RWMutex
}

// NewCanidaeClient creates a new mobile client instance
func NewCanidaeClient() *CanidaeClient {
	ctx, cancel := context.WithCancel(context.Background())
	return &CanidaeClient{
		ctx:    ctx,
		cancel: cancel,
	}
}

// Initialize initializes the client with configuration
// Config should be a JSON string with the following structure:
// {
//   "serverEndpoint": "192.168.1.38:14001",
//   "packID": "mobile-pack",
//   "apiKey": "your-api-key",
//   "securityProfile": "enterprise"
// }
func (c *CanidaeClient) Initialize(configJSON string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	var config struct {
		ServerEndpoint  string `json:"serverEndpoint"`
		PackID          string `json:"packID"`
		APIKey          string `json:"apiKey"`
		SecurityProfile string `json:"securityProfile"`
	}
	
	if err := json.Unmarshal([]byte(configJSON), &config); err != nil {
		return errors.New("invalid configuration JSON: " + err.Error())
	}
	
	// Build options
	opts := []canidae.Option{
		canidae.WithTimeout(30 * time.Second),
	}
	
	if config.ServerEndpoint != "" {
		opts = append(opts, canidae.WithServerEndpoint(config.ServerEndpoint))
	}
	
	if config.PackID != "" {
		opts = append(opts, canidae.WithPackID(config.PackID))
	}
	
	if config.APIKey != "" {
		opts = append(opts, canidae.WithAPIKey(config.APIKey))
	}
	
	if config.SecurityProfile != "" {
		opts = append(opts, canidae.WithSecurityProfile(canidae.SecurityProfile(config.SecurityProfile)))
	}
	
	// Create client
	client, err := canidae.NewClient(opts...)
	if err != nil {
		return err
	}
	
	c.client = client
	return nil
}

// Connect establishes connection to the CANIDAE server
func (c *CanidaeClient) Connect() error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if c.client == nil {
		return errors.New("client not initialized")
	}
	
	return c.client.Connect(c.ctx)
}

// Disconnect closes the connection
func (c *CanidaeClient) Disconnect() error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if c.client == nil {
		return errors.New("client not initialized")
	}
	
	return c.client.Disconnect(c.ctx)
}

// ExecuteAgent executes a single AI agent
// Request should be a JSON string with the following structure:
// {
//   "agent": "anthropic",
//   "prompt": "Your prompt here",
//   "model": "claude-3-opus",
//   "temperature": 0.7,
//   "maxTokens": 500
// }
func (c *CanidaeClient) ExecuteAgent(requestJSON string) (string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if c.client == nil {
		return "", errors.New("client not initialized")
	}
	
	var request struct {
		Agent       string  `json:"agent"`
		Prompt      string  `json:"prompt"`
		Model       string  `json:"model"`
		Temperature float32 `json:"temperature"`
		MaxTokens   int     `json:"maxTokens"`
	}
	
	if err := json.Unmarshal([]byte(requestJSON), &request); err != nil {
		return "", errors.New("invalid request JSON: " + err.Error())
	}
	
	// Create execute request
	execReq := &canidae.ExecuteRequest{
		Agent:       canidae.AgentType(request.Agent),
		Prompt:      request.Prompt,
		Model:       request.Model,
		Temperature: request.Temperature,
		MaxTokens:   request.MaxTokens,
	}
	
	// Execute
	resp, err := c.client.ExecuteAgent(c.ctx, execReq)
	if err != nil {
		return "", err
	}
	
	// Convert response to JSON
	responseJSON, err := json.Marshal(resp)
	if err != nil {
		return "", err
	}
	
	return string(responseJSON), nil
}

// ChainAgents executes multiple agents in sequence
// Request should be a JSON string with an array of steps
func (c *CanidaeClient) ChainAgents(requestJSON string) (string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if c.client == nil {
		return "", errors.New("client not initialized")
	}
	
	var request struct {
		Steps []struct {
			Agent       string   `json:"agent"`
			Prompt      string   `json:"prompt"`
			Model       string   `json:"model"`
			Temperature float32  `json:"temperature"`
			MaxTokens   int      `json:"maxTokens"`
			DependsOn   []string `json:"dependsOn"`
		} `json:"steps"`
		ContinueOnError bool `json:"continueOnError"`
	}
	
	if err := json.Unmarshal([]byte(requestJSON), &request); err != nil {
		return "", errors.New("invalid request JSON: " + err.Error())
	}
	
	// Build chain steps
	steps := make([]canidae.ChainStep, len(request.Steps))
	for i, step := range request.Steps {
		steps[i] = canidae.ChainStep{
			Agent:       canidae.AgentType(step.Agent),
			Prompt:      step.Prompt,
			Model:       step.Model,
			Temperature: step.Temperature,
			MaxTokens:   step.MaxTokens,
			DependsOn:   step.DependsOn,
		}
	}
	
	// Create chain request
	chainReq := &canidae.ChainRequest{
		Steps:           steps,
		ContinueOnError: request.ContinueOnError,
	}
	
	// Execute chain
	resp, err := c.client.ChainAgents(c.ctx, chainReq)
	if err != nil {
		return "", err
	}
	
	// Convert response to JSON
	responseJSON, err := json.Marshal(resp)
	if err != nil {
		return "", err
	}
	
	return string(responseJSON), nil
}

// GetStatus returns the current client status as JSON
func (c *CanidaeClient) GetStatus() (string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if c.client == nil {
		return "", errors.New("client not initialized")
	}
	
	status := c.client.GetStatus()
	
	// Convert to JSON
	statusJSON, err := json.Marshal(status)
	if err != nil {
		return "", err
	}
	
	return string(statusJSON), nil
}

// SetPackID sets the pack identifier
func (c *CanidaeClient) SetPackID(packID string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	if c.client == nil {
		return errors.New("client not initialized")
	}
	
	c.client.SetPackID(packID)
	return nil
}

// Close cleans up resources
func (c *CanidaeClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if c.cancel != nil {
		c.cancel()
	}
	
	if c.client != nil {
		return c.client.Disconnect(context.Background())
	}
	
	return nil
}

// Simple data structures for mobile platforms

// AgentType represents supported AI agents
type AgentType string

const (
	AgentTypeAnthropic AgentType = "anthropic"
	AgentTypeOpenAI    AgentType = "openai"
	AgentTypeGemini    AgentType = "gemini"
	AgentTypeOllama    AgentType = "ollama"
	AgentTypeDeepSeek  AgentType = "deepseek"
)

// SecurityProfile represents security profiles
type SecurityProfile string

const (
	SecurityProfileEnterprise SecurityProfile = "enterprise"
	SecurityProfileFinance    SecurityProfile = "finance"
	SecurityProfileICS        SecurityProfile = "ics_iot"
	SecurityProfileDebug      SecurityProfile = "debug"
	SecurityProfilePermissive SecurityProfile = "permissive"
)

// Helper functions for mobile platforms

// GetAgentTypes returns available agent types as JSON array
func GetAgentTypes() string {
	agents := []string{
		string(AgentTypeAnthropic),
		string(AgentTypeOpenAI),
		string(AgentTypeGemini),
		string(AgentTypeOllama),
		string(AgentTypeDeepSeek),
	}
	
	data, _ := json.Marshal(agents)
	return string(data)
}

// GetSecurityProfiles returns available security profiles as JSON array
func GetSecurityProfiles() string {
	profiles := []string{
		string(SecurityProfileEnterprise),
		string(SecurityProfileFinance),
		string(SecurityProfileICS),
		string(SecurityProfileDebug),
		string(SecurityProfilePermissive),
	}
	
	data, _ := json.Marshal(profiles)
	return string(data)
}

// CreateExecuteRequest is a helper to create an execute request JSON
func CreateExecuteRequest(agent, prompt, model string, temperature float32, maxTokens int) string {
	request := map[string]interface{}{
		"agent":       agent,
		"prompt":      prompt,
		"model":       model,
		"temperature": temperature,
		"maxTokens":   maxTokens,
	}
	
	data, _ := json.Marshal(request)
	return string(data)
}

// CreateChainStep is a helper to create a chain step JSON
func CreateChainStep(agent, prompt, model string) string {
	step := map[string]interface{}{
		"agent":  agent,
		"prompt": prompt,
		"model":  model,
	}
	
	data, _ := json.Marshal(step)
	return string(data)
}