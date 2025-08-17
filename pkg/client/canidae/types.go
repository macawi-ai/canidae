package canidae

import (
	"errors"
	"strings"
	"time"
)

// Common errors
var (
	ErrInvalidRequest       = errors.New("invalid request")
	ErrNotConnected         = errors.New("client not connected")
	ErrAuthenticationFailed = errors.New("authentication failed")
	ErrSessionExpired       = errors.New("session expired")
)

// AgentType represents the type of AI agent
type AgentType string

const (
	AgentTypeAnthropic AgentType = "anthropic"
	AgentTypeOpenAI    AgentType = "openai"
	AgentTypeGemini    AgentType = "gemini"
	AgentTypeOllama    AgentType = "ollama"
	AgentTypeDeepSeek  AgentType = "deepseek"
)

// SecurityProfile represents the security profile for execution
type SecurityProfile string

const (
	SecurityProfileEnterprise SecurityProfile = "enterprise"
	SecurityProfileFinance    SecurityProfile = "finance"
	SecurityProfileICS        SecurityProfile = "ics_iot"
	SecurityProfileDebug      SecurityProfile = "debug"
	SecurityProfilePermissive SecurityProfile = "permissive"
)

// ExecuteRequest represents a request to execute an agent
type ExecuteRequest struct {
	Agent           AgentType         `json:"agent"`
	Prompt          string            `json:"prompt"`
	Model           string            `json:"model,omitempty"`
	Temperature     float32           `json:"temperature,omitempty"`
	MaxTokens       int               `json:"max_tokens,omitempty"`
	SecurityProfile SecurityProfile   `json:"security_profile,omitempty"`
	Metadata        map[string]string `json:"metadata,omitempty"`
}

// Validate checks if the request is valid
func (r *ExecuteRequest) Validate() error {
	if r.Agent == "" {
		return errors.New("agent type is required")
	}
	// Trim whitespace and check if prompt is empty
	if strings.TrimSpace(r.Prompt) == "" {
		return errors.New("prompt is required")
	}
	return nil
}

// ExecuteResponse represents the response from an agent execution
type ExecuteResponse struct {
	RequestID  string            `json:"request_id"`
	Response   string            `json:"response"`
	TokensUsed int               `json:"tokens_used"`
	Duration   time.Duration     `json:"duration"`
	Model      string            `json:"model"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// ChainRequest represents a request to chain multiple agents
type ChainRequest struct {
	Steps           []ChainStep       `json:"steps"`
	SecurityProfile SecurityProfile   `json:"security_profile,omitempty"`
	ContinueOnError bool              `json:"continue_on_error,omitempty"`
	Metadata        map[string]string `json:"metadata,omitempty"`
}

// ChainStep represents a single step in an agent chain
type ChainStep struct {
	Agent       AgentType         `json:"agent"`
	Prompt      string            `json:"prompt"`
	Model       string            `json:"model,omitempty"`
	Temperature float32           `json:"temperature,omitempty"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	DependsOn   []string          `json:"depends_on,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// Validate checks if the chain request is valid
func (r *ChainRequest) Validate() error {
	if len(r.Steps) == 0 {
		return errors.New("at least one step is required")
	}
	for i, step := range r.Steps {
		if step.Agent == "" {
			return errors.New("agent type is required for all steps")
		}
		if strings.TrimSpace(step.Prompt) == "" {
			return errors.New("prompt is required for all steps")
		}
		// Validate dependencies exist
		for _, dep := range step.DependsOn {
			found := false
			for j := 0; j < i; j++ {
				if r.Steps[j].Agent == AgentType(dep) {
					found = true
					break
				}
			}
			if !found {
				return errors.New("invalid dependency: " + dep)
			}
		}
	}
	return nil
}

// ChainResponse represents the response from a chain execution
type ChainResponse struct {
	RequestID   string            `json:"request_id"`
	Steps       []StepResponse    `json:"steps"`
	TotalTokens int               `json:"total_tokens"`
	Duration    time.Duration     `json:"duration"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// StepResponse represents the response from a single chain step
type StepResponse struct {
	StepID     string            `json:"step_id"`
	Agent      AgentType         `json:"agent"`
	Response   string            `json:"response"`
	TokensUsed int               `json:"tokens_used"`
	Duration   time.Duration     `json:"duration"`
	Error      string            `json:"error,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// PackRequest represents a request to summon a pack formation
type PackRequest struct {
	Formation       PackFormation     `json:"formation"`
	Objective       string            `json:"objective"`
	SecurityProfile SecurityProfile   `json:"security_profile,omitempty"`
	MaxConcurrency  int               `json:"max_concurrency,omitempty"`
	Timeout         time.Duration     `json:"timeout,omitempty"`
	Metadata        map[string]string `json:"metadata,omitempty"`
}

// PackFormation represents the structure of a pack
type PackFormation struct {
	Alpha    *PackMember  `json:"alpha"`
	Hunters  []PackMember `json:"hunters,omitempty"`
	Scouts   []PackMember `json:"scouts,omitempty"`
	Sentries []PackMember `json:"sentries,omitempty"`
	Elders   []PackMember `json:"elders,omitempty"`
}

// PackMember represents a member of the pack
type PackMember struct {
	Role        string            `json:"role"`
	Agent       AgentType         `json:"agent"`
	Objective   string            `json:"objective"`
	Model       string            `json:"model,omitempty"`
	Temperature float32           `json:"temperature,omitempty"`
	MaxTokens   int               `json:"max_tokens,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// Validate checks if the pack request is valid
func (r *PackRequest) Validate() error {
	if strings.TrimSpace(r.Objective) == "" {
		return errors.New("objective is required")
	}
	if r.Formation.Alpha == nil {
		return errors.New("pack alpha is required")
	}
	// Validate alpha member
	if r.Formation.Alpha.Role == "" {
		return errors.New("role is required for alpha")
	}
	if strings.TrimSpace(r.Formation.Alpha.Objective) == "" {
		return errors.New("objective is required for alpha")
	}
	if r.Formation.Alpha.Agent == "" {
		return errors.New("agent type is required for alpha")
	}
	// Validate other pack members
	for _, member := range r.Formation.Hunters {
		if err := validatePackMember(member, "hunter"); err != nil {
			return err
		}
	}
	for _, member := range r.Formation.Scouts {
		if err := validatePackMember(member, "scout"); err != nil {
			return err
		}
	}
	for _, member := range r.Formation.Sentries {
		if err := validatePackMember(member, "sentry"); err != nil {
			return err
		}
	}
	for _, member := range r.Formation.Elders {
		if err := validatePackMember(member, "elder"); err != nil {
			return err
		}
	}
	return nil
}

// validatePackMember validates a pack member
func validatePackMember(member PackMember, memberType string) error {
	if member.Role == "" {
		return errors.New("role is required for " + memberType)
	}
	if strings.TrimSpace(member.Objective) == "" {
		return errors.New("objective is required for " + memberType)
	}
	if member.Agent == "" {
		return errors.New("agent type is required for " + memberType)
	}
	return nil
}

// PackResponse represents the response from a pack formation
type PackResponse struct {
	RequestID   string            `json:"request_id"`
	PackID      string            `json:"pack_id"`
	Results     []PackResult      `json:"results"`
	TotalTokens int               `json:"total_tokens"`
	Duration    time.Duration     `json:"duration"`
	Metadata    map[string]string `json:"metadata,omitempty"`
}

// PackResult represents the result from a pack member
type PackResult struct {
	MemberID   string            `json:"member_id"`
	Role       string            `json:"role"`
	Agent      AgentType         `json:"agent"`
	Response   string            `json:"response"`
	TokensUsed int               `json:"tokens_used"`
	Duration   time.Duration     `json:"duration"`
	Error      string            `json:"error,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// Status represents the client status
type Status struct {
	Connected      bool      `json:"connected"`
	Authenticated  bool      `json:"authenticated"`
	PackID         string    `json:"pack_id,omitempty"`
	ServerEndpoint string    `json:"server_endpoint"`
	LastActivity   time.Time `json:"last_activity"`
}

// StreamHandler handles streaming responses
type StreamHandler func(event StreamEvent) error

// StreamEvent represents a streaming event
type StreamEvent struct {
	Type     StreamEventType   `json:"type"`
	Data     interface{}       `json:"data"`
	Error    error             `json:"error,omitempty"`
	Metadata map[string]string `json:"metadata,omitempty"`
}
