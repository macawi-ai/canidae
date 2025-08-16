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

// OpenAIProvider implements the Provider interface for GPT models
type OpenAIProvider struct {
	BaseProvider
	apiKey     string
	baseURL    string
	httpClient *http.Client
	orgID      string
}

// OpenAIRequest represents the API request format
type OpenAIRequest struct {
	Model            string            `json:"model"`
	Messages         []OpenAIMessage   `json:"messages"`
	MaxTokens        int              `json:"max_tokens,omitempty"`
	Temperature      float64          `json:"temperature,omitempty"`
	TopP             float64          `json:"top_p,omitempty"`
	N                int              `json:"n,omitempty"`
	Stream           bool             `json:"stream,omitempty"`
	Stop             []string         `json:"stop,omitempty"`
	PresencePenalty  float64          `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64          `json:"frequency_penalty,omitempty"`
	User             string           `json:"user,omitempty"`
}

// OpenAIMessage represents a message in the conversation
type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OpenAIResponse represents the API response
type OpenAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   OpenAIUsage    `json:"usage"`
}

// OpenAIChoice represents a response choice
type OpenAIChoice struct {
	Index        int           `json:"index"`
	Message      OpenAIMessage `json:"message"`
	FinishReason string        `json:"finish_reason"`
}

// OpenAIUsage represents token usage
type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// NewOpenAIProvider creates an OpenAI provider instance
func NewOpenAIProvider(config map[string]interface{}) (Provider, error) {
	apiKey, ok := config["api_key"].(string)
	if !ok || apiKey == "" {
		return nil, fmt.Errorf("api_key is required for OpenAI provider")
	}
	
	baseURL := "https://api.openai.com"
	if url, ok := config["base_url"].(string); ok {
		baseURL = url
	}
	
	orgID := ""
	if org, ok := config["organization_id"].(string); ok {
		orgID = org
	}
	
	p := &OpenAIProvider{
		BaseProvider: BaseProvider{
			name:   "openai",
			config: config,
			capabilities: Capabilities{
				Models: []ModelInfo{
					{
						ID:           "gpt-4-turbo-preview",
						Name:         "GPT-4 Turbo",
						ContextSize:  128000,
						CostPerToken: 0.00001, // $10 per 1M input tokens
					},
					{
						ID:           "gpt-4",
						Name:         "GPT-4",
						ContextSize:  8192,
						CostPerToken: 0.00003, // $30 per 1M input tokens
					},
					{
						ID:           "gpt-3.5-turbo",
						Name:         "GPT-3.5 Turbo",
						ContextSize:  16385,
						CostPerToken: 0.0000005, // $0.50 per 1M input tokens
					},
				},
				MaxTokens:        4096,
				SupportsStream:   true,
				SupportsFunction: true,
				SupportsVision:   true,
				SupportsAudio:    false,
			},
		},
		apiKey:     apiKey,
		baseURL:    baseURL,
		httpClient: &http.Client{Timeout: 30 * time.Second},
		orgID:      orgID,
	}
	
	return p, nil
}

// Execute sends a request to the OpenAI API
func (p *OpenAIProvider) Execute(ctx context.Context, req Request) (*Response, error) {
	start := time.Now()
	
	// Convert to OpenAI format
	openAIReq := OpenAIRequest{
		Model:            req.Model,
		MaxTokens:        req.Options.MaxTokens,
		Temperature:      req.Options.Temperature,
		TopP:             req.Options.TopP,
		Stream:           req.Options.Stream,
		Stop:             req.Options.StopSequences,
		PresencePenalty:  req.Options.PresencePenalty,
		FrequencyPenalty: req.Options.FrequencyPenalty,
	}
	
	// Convert messages
	for _, msg := range req.Messages {
		openAIReq.Messages = append(openAIReq.Messages, OpenAIMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}
	
	// Marshal request
	reqBody, err := json.Marshal(openAIReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	
	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", 
		fmt.Sprintf("%s/v1/chat/completions", p.baseURL), 
		bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	// Set headers
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", p.apiKey))
	if p.orgID != "" {
		httpReq.Header.Set("OpenAI-Organization", p.orgID)
	}
	
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
	var openAIResp OpenAIResponse
	if err := json.Unmarshal(body, &openAIResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}
	
	// Extract content
	content := ""
	if len(openAIResp.Choices) > 0 {
		content = openAIResp.Choices[0].Message.Content
	}
	
	// Calculate cost
	costPerToken := 0.00001 // Default to GPT-4 Turbo pricing
	for _, model := range p.capabilities.Models {
		if model.ID == req.Model {
			costPerToken = model.CostPerToken
			break
		}
	}
	cost := float64(openAIResp.Usage.TotalTokens) * costPerToken
	
	// Build response
	return &Response{
		ID:        openAIResp.ID,
		RequestID: req.ID,
		Provider:  p.name,
		Model:     req.Model,
		Content:   content,
		Usage: Usage{
			PromptTokens:     openAIResp.Usage.PromptTokens,
			CompletionTokens: openAIResp.Usage.CompletionTokens,
			TotalTokens:      openAIResp.Usage.TotalTokens,
			Cost:             cost,
			Currency:         "USD",
		},
		Metadata: Metadata{
			ProcessingTime: time.Since(start),
			Extra: map[string]interface{}{
				"created": openAIResp.Created,
				"object":  openAIResp.Object,
			},
		},
		Timestamp: time.Now(),
	}, nil
}

// HealthCheck verifies the OpenAI API is accessible
func (p *OpenAIProvider) HealthCheck(ctx context.Context) error {
	// Use the models endpoint for health check
	req, err := http.NewRequestWithContext(ctx, "GET", 
		fmt.Sprintf("%s/v1/models", p.baseURL), nil)
	if err != nil {
		return err
	}
	
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", p.apiKey))
	if p.orgID != "" {
		req.Header.Set("OpenAI-Organization", p.orgID)
	}
	
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
func (p *OpenAIProvider) EstimateCost(req Request) Cost {
	// Find the model
	var costPerToken float64 = 0.00001 // Default to GPT-4 Turbo
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