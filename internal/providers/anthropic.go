package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// AnthropicProvider implements the Provider interface for Claude
type AnthropicProvider struct {
	BaseProvider
	apiKey     string
	baseURL    string
	httpClient *http.Client
	version    string
}

// AnthropicRequest represents the API request format
type AnthropicRequest struct {
	Model       string                   `json:"model"`
	Messages    []AnthropicMessage       `json:"messages"`
	MaxTokens   int                     `json:"max_tokens"`
	Temperature float64                  `json:"temperature,omitempty"`
	System      string                   `json:"system,omitempty"`
	Metadata    map[string]interface{}   `json:"metadata,omitempty"`
}

// AnthropicMessage represents a message in the conversation
type AnthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// AnthropicResponse represents the API response
type AnthropicResponse struct {
	ID           string       `json:"id"`
	Type         string       `json:"type"`
	Model        string       `json:"model"`
	Role         string       `json:"role"`
	Content      []Content    `json:"content"`
	StopReason   string       `json:"stop_reason"`
	StopSequence string       `json:"stop_sequence,omitempty"`
	Usage        AnthropicUsage `json:"usage"`
}

// Content represents response content
type Content struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// AnthropicUsage represents token usage
type AnthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// NewAnthropicProvider creates an Anthropic provider instance
func NewAnthropicProvider(config map[string]interface{}) (Provider, error) {
	apiKey, ok := config["api_key"].(string)
	if !ok || apiKey == "" {
		return nil, fmt.Errorf("api_key is required for Anthropic provider")
	}
	
	baseURL := "https://api.anthropic.com"
	if url, ok := config["base_url"].(string); ok {
		baseURL = url
	}
	
	p := &AnthropicProvider{
		BaseProvider: BaseProvider{
			name:   "anthropic",
			config: config,
			capabilities: Capabilities{
				Models: []ModelInfo{
					{
						ID:           "claude-3-opus-20240229",
						Name:         "Claude 3 Opus",
						ContextSize:  200000,
						CostPerToken: 0.000015, // $15 per 1M input tokens
					},
					{
						ID:           "claude-3-sonnet-20240229",
						Name:         "Claude 3 Sonnet",
						ContextSize:  200000,
						CostPerToken: 0.000003, // $3 per 1M input tokens
					},
					{
						ID:           "claude-3-haiku-20240307",
						Name:         "Claude 3 Haiku",
						ContextSize:  200000,
						CostPerToken: 0.00000025, // $0.25 per 1M input tokens
					},
				},
				MaxTokens:        4096,
				SupportsStream:   true,
				SupportsFunction: false,
				SupportsVision:   true,
				SupportsAudio:    false,
			},
		},
		apiKey:     apiKey,
		baseURL:    baseURL,
		httpClient: &http.Client{Timeout: 30 * time.Second},
		version:    "2023-06-01",
	}
	
	return p, nil
}

// Execute sends a request to the Anthropic API
func (p *AnthropicProvider) Execute(ctx context.Context, req Request) (*Response, error) {
	start := time.Now()
	
	// Convert to Anthropic format
	anthropicReq := AnthropicRequest{
		Model:       req.Model,
		MaxTokens:   req.Options.MaxTokens,
		Temperature: req.Options.Temperature,
	}
	
	// Set default max tokens if not specified
	if anthropicReq.MaxTokens == 0 {
		anthropicReq.MaxTokens = 1024
	}
	
	// Convert messages
	for _, msg := range req.Messages {
		if msg.Role == "system" {
			anthropicReq.System = msg.Content
		} else {
			anthropicReq.Messages = append(anthropicReq.Messages, AnthropicMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}
	
	// Marshal request
	reqBody, err := json.Marshal(anthropicReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", 
		fmt.Sprintf("%s/v1/messages", p.baseURL), 
		bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", p.apiKey)
	httpReq.Header.Set("anthropic-version", p.version)
	
	// Send request
	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()
	
	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}
	
	// Check for errors
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}
	
	// Parse response
	var anthropicResp AnthropicResponse
	if err := json.Unmarshal(body, &anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	// Extract content
	content := ""
	for _, c := range anthropicResp.Content {
		if c.Type == "text" {
			content += c.Text
		}
	}
	
	// Calculate cost (simplified)
	totalTokens := anthropicResp.Usage.InputTokens + anthropicResp.Usage.OutputTokens
	costPerToken := 0.000003 // Default to Sonnet pricing
	for _, model := range p.capabilities.Models {
		if model.ID == req.Model {
			costPerToken = model.CostPerToken
			break
		}
	}
	cost := float64(totalTokens) * costPerToken
	
	// Build response
	return &Response{
		ID:        anthropicResp.ID,
		RequestID: req.ID,
		Provider:  p.name,
		Model:     req.Model,
		Content:   content,
		Usage: Usage{
			PromptTokens:     anthropicResp.Usage.InputTokens,
			CompletionTokens: anthropicResp.Usage.OutputTokens,
			TotalTokens:      totalTokens,
			Cost:             cost,
			Currency:         "USD",
		},
		Metadata: Metadata{
			ProcessingTime: time.Since(start),
			Version:        p.version,
		},
		Timestamp: time.Now(),
	}, nil
}

// HealthCheck verifies the Anthropic API is accessible
func (p *AnthropicProvider) HealthCheck(ctx context.Context) error {
	// Simple health check - try to get models list
	req, err := http.NewRequestWithContext(ctx, "GET", 
		fmt.Sprintf("%s/v1/models", p.baseURL), nil)
	if err != nil {
		return err
	}
	
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", p.version)
	
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned status %d", resp.StatusCode)
	}
	
	return nil
}

// EstimateCost calculates the expected cost for a request
func (p *AnthropicProvider) EstimateCost(req Request) Cost {
	// Find the model
	var costPerToken float64 = 0.000003 // Default to Sonnet
	for _, model := range p.capabilities.Models {
		if model.ID == req.Model {
			costPerToken = model.CostPerToken
			break
		}
	}
	
	// Estimate tokens (rough approximation: 1 token ≈ 4 characters)
	estimatedInputTokens := 0
	for _, msg := range req.Messages {
		estimatedInputTokens += len(msg.Content) / 4
	}
	
	// Estimate output tokens
	maxTokens := req.Options.MaxTokens
	if maxTokens == 0 {
		maxTokens = 1024
	}
	
	// Calculate costs
	minCost := float64(estimatedInputTokens) * costPerToken
	maxCost := float64(estimatedInputTokens+maxTokens) * costPerToken
	avgCost := (minCost + maxCost) / 2
	
	return Cost{
		Estimated: avgCost,
		Minimum:   minCost,
		Maximum:   maxCost,
		Currency:  "USD",
	}
}

// Register the provider factory
func init() {
	// This will be called by the main application
	// to register the Anthropic provider factory
}