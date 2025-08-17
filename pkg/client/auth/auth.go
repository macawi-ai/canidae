package auth

import (
	"context"
	"errors"
	"time"

	"github.com/macawi-ai/canidae/pkg/client/config"
)

// Common errors
var (
	ErrAuthenticationFailed = errors.New("authentication failed")
	ErrMFARequired          = errors.New("MFA required")
	ErrInvalidCredentials   = errors.New("invalid credentials")
	ErrTokenExpired         = errors.New("token expired")
)

// Provider represents an authentication provider
type Provider interface {
	// Authenticate performs authentication and returns a token
	Authenticate(ctx context.Context) (*Token, error)

	// Refresh refreshes an existing token
	Refresh(ctx context.Context, token *Token) (*Token, error)

	// Revoke revokes a token
	Revoke(ctx context.Context, token *Token) error

	// GetType returns the provider type
	GetType() string

	// IsInteractive returns true if the provider requires user interaction
	IsInteractive() bool
}

// Token represents an authentication token
type Token struct {
	Type         string            `json:"type"`  // bearer, paseto, etc.
	Value        string            `json:"value"` // The actual token
	ExpiresAt    time.Time         `json:"expires_at"`
	RefreshToken string            `json:"refresh_token,omitempty"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

// IsExpired checks if the token is expired
func (t *Token) IsExpired() bool {
	return time.Now().After(t.ExpiresAt)
}

// ShouldRefresh checks if the token should be refreshed
func (t *Token) ShouldRefresh(before time.Duration) bool {
	return time.Now().Add(before).After(t.ExpiresAt)
}

// NewProvider creates a new authentication provider based on configuration
func NewProvider(cfg config.AuthConfig) (Provider, error) {
	switch cfg.Type {
	case "webauthn":
		if !cfg.WebAuthn.Enabled {
			return nil, errors.New("WebAuthn is not enabled")
		}
		return NewWebAuthnProvider(cfg.WebAuthn)
	case "oauth":
		if !cfg.OAuth.Enabled {
			return nil, errors.New("OAuth is not enabled")
		}
		return NewOAuthProvider(cfg.OAuth)
	case "apikey":
		// API key can be empty for testing, will fail on auth
		return NewAPIKeyProvider(cfg.APIKey)
	default:
		return nil, errors.New("unsupported auth type: " + cfg.Type)
	}
}

// NewWebAuthnProvider creates a new WebAuthn provider (to be implemented)
func NewWebAuthnProvider(cfg config.WebAuthnConfig) (Provider, error) {
	// This will be implemented in webauthn.go
	return nil, errors.New("WebAuthn provider not yet implemented")
}

// NewOAuthProvider creates a new OAuth provider (to be implemented)
func NewOAuthProvider(cfg config.OAuthConfig) (Provider, error) {
	// This will be implemented in oauth.go
	return nil, errors.New("OAuth provider not yet implemented")
}

// NewAPIKeyProvider creates a new API key provider (to be implemented)
func NewAPIKeyProvider(apiKey string) (Provider, error) {
	// This will be implemented in apikey.go
	return &apiKeyProvider{
		apiKey: apiKey,
	}, nil
}

// apiKeyProvider implements a simple API key authentication
type apiKeyProvider struct {
	apiKey string
}

func (p *apiKeyProvider) Authenticate(ctx context.Context) (*Token, error) {
	return &Token{
		Type:      "bearer",
		Value:     p.apiKey,
		ExpiresAt: time.Now().Add(24 * time.Hour), // API keys don't expire
		Metadata: map[string]string{
			"auth_type": "apikey",
		},
	}, nil
}

func (p *apiKeyProvider) Refresh(ctx context.Context, token *Token) (*Token, error) {
	// API keys don't need refresh
	return token, nil
}

func (p *apiKeyProvider) Revoke(ctx context.Context, token *Token) error {
	// API keys can't be revoked client-side
	return nil
}

func (p *apiKeyProvider) GetType() string {
	return "apikey"
}

func (p *apiKeyProvider) IsInteractive() bool {
	return false
}
