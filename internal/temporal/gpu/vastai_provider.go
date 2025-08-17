package gpu

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// VastAIProvider implements GPU provider for Vast.ai
type VastAIProvider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
	
	// Cached data
	lastCheck time.Time
	available bool
}

// NewVastAIProvider creates a new Vast.ai provider
func NewVastAIProvider(apiKey string) *VastAIProvider {
	return &VastAIProvider{
		apiKey:  apiKey,
		baseURL: "https://vast.ai/api/v0",
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Name returns the provider name
func (v *VastAIProvider) Name() string {
	return "Vast.ai"
}

// GetCostPerHour returns typical RTX 3090 cost
func (v *VastAIProvider) GetCostPerHour() float64 {
	return 0.20 // $0.20/hour for RTX 3090
}

// GetLatency returns typical startup latency
func (v *VastAIProvider) GetLatency() time.Duration {
	return 30 * time.Second // Instance startup time
}

// Available checks if GPUs are available
func (v *VastAIProvider) Available(ctx context.Context) (bool, error) {
	// Cache availability for 1 minute
	if time.Since(v.lastCheck) < time.Minute {
		return v.available, nil
	}
	
	// Query available instances
	req, err := v.newRequest(ctx, "GET", "/instances", nil)
	if err != nil {
		return false, err
	}
	
	resp, err := v.httpClient.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("API returned status %d", resp.StatusCode)
	}
	
	var result struct {
		Instances []struct {
			GPUName      string  `json:"gpu_name"`
			Available    bool    `json:"rentable"`
			PricePerHour float64 `json:"dph_total"`
		} `json:"offers"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return false, err
	}
	
	// Check for RTX 3090 availability
	for _, instance := range result.Instances {
		if instance.GPUName == "RTX 3090" && instance.Available {
			v.available = true
			v.lastCheck = time.Now()
			return true, nil
		}
	}
	
	v.available = false
	v.lastCheck = time.Now()
	return false, nil
}

// RequestCompute requests a GPU instance
func (v *VastAIProvider) RequestCompute(ctx context.Context, req ComputeRequest) (*ComputeSession, error) {
	// Build instance request
	instanceReq := map[string]interface{}{
		"gpu_name":     "RTX 3090",
		"num_gpus":     1,
		"disk_space":   50, // GB
		"image":        "pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
		"onstart":      v.buildStartupScript(req),
	}
	
	reqBody, err := json.Marshal(instanceReq)
	if err != nil {
		return nil, err
	}
	
	// Create instance
	httpReq, err := v.newRequest(ctx, "POST", "/instances", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	
	resp, err := v.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("failed to create instance: %s", body)
	}
	
	var result struct {
		Success    bool   `json:"success"`
		InstanceID string `json:"new_instance"`
		Message    string `json:"msg"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	
	if !result.Success {
		return nil, fmt.Errorf("instance creation failed: %s", result.Message)
	}
	
	// Create session
	session := &ComputeSession{
		ID:          result.InstanceID,
		Provider:    v.Name(),
		StartTime:   time.Now(),
		CostPerHour: v.GetCostPerHour(),
		MemoryMB:    24576, // RTX 3090 has 24GB
		Status:      SessionActive,
	}
	
	return session, nil
}

// ReleaseCompute terminates a GPU instance
func (v *VastAIProvider) ReleaseCompute(ctx context.Context, sessionID string) error {
	// Destroy instance
	httpReq, err := v.newRequest(ctx, "DELETE", fmt.Sprintf("/instances/%s", sessionID), nil)
	if err != nil {
		return err
	}
	
	resp, err := v.httpClient.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("failed to destroy instance: %s", body)
	}
	
	return nil
}

// buildStartupScript creates initialization script for the instance
func (v *VastAIProvider) buildStartupScript(req ComputeRequest) string {
	script := `#!/bin/bash
set -e

# Install dependencies
pip install torch transformers flash-attn

# Download model based on type
`
	
	switch req.ModelType {
	case "HRM":
		script += `
# Download HRM checkpoint
wget https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2/resolve/main/checkpoint.pt
`
	case "ARChitects":
		script += `
# Download ARChitects model
pip install unsloth
wget https://huggingface.co/da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit
`
	default:
		script += "# No specific model requested\n"
	}
	
	script += `
# Signal ready
echo "GPU_READY" > /tmp/ready.flag
`
	
	return script
}

// newRequest creates an authenticated HTTP request
func (v *VastAIProvider) newRequest(ctx context.Context, method, path string, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, v.baseURL+path, body)
	if err != nil {
		return nil, err
	}
	
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+v.apiKey)
	
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	
	return req, nil
}

// VastAIConfig holds Vast.ai configuration
type VastAIConfig struct {
	APIKey          string
	PreferredGPU    string
	MaxPricePerHour float64
	MinGPUMemoryGB  int
}

// ValidateConfig checks if configuration is valid
func (c VastAIConfig) ValidateConfig() error {
	if c.APIKey == "" {
		return fmt.Errorf("API key is required")
	}
	
	if c.MaxPricePerHour <= 0 {
		return fmt.Errorf("max price per hour must be positive")
	}
	
	if c.MinGPUMemoryGB < 8 {
		return fmt.Errorf("minimum GPU memory should be at least 8GB")
	}
	
	return nil
}