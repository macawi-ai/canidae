package config

import (
	"errors"
	"net/url"
	"time"
)

// Config represents the client configuration
type Config struct {
	// Server settings
	ServerEndpoint         string            `json:"server_endpoint" yaml:"server_endpoint"`
	PackID                 string            `json:"pack_id" yaml:"pack_id"`
	DefaultSecurityProfile string            `json:"default_security_profile" yaml:"default_security_profile"`
	
	// Timeouts
	RequestTimeout   time.Duration `json:"request_timeout" yaml:"request_timeout"`
	ConnectionTimeout time.Duration `json:"connection_timeout" yaml:"connection_timeout"`
	
	// Transport configuration
	TransportConfig TransportConfig `json:"transport" yaml:"transport"`
	
	// Auth configuration
	AuthConfig AuthConfig `json:"auth" yaml:"auth"`
	
	// Session configuration
	SessionConfig SessionConfig `json:"session" yaml:"session"`
	
	// Logging
	LogLevel string `json:"log_level" yaml:"log_level"`
	
	// Default metadata
	DefaultMetadata map[string]string `json:"default_metadata" yaml:"default_metadata"`
}

// TransportConfig represents transport layer configuration
type TransportConfig struct {
	Type             string        `json:"type" yaml:"type"` // grpc, grpcweb, http
	TLS              TLSConfig     `json:"tls" yaml:"tls"`
	Retry            RetryConfig   `json:"retry" yaml:"retry"`
	StreamBufferSize int           `json:"stream_buffer_size" yaml:"stream_buffer_size"`
	MaxMessageSize   int           `json:"max_message_size" yaml:"max_message_size"`
}

// TLSConfig represents TLS configuration
type TLSConfig struct {
	Enabled         bool     `json:"enabled" yaml:"enabled"`
	CertFile        string   `json:"cert_file" yaml:"cert_file"`
	KeyFile         string   `json:"key_file" yaml:"key_file"`
	CAFile          string   `json:"ca_file" yaml:"ca_file"`
	ServerName      string   `json:"server_name" yaml:"server_name"`
	InsecureSkipVerify bool  `json:"insecure_skip_verify" yaml:"insecure_skip_verify"`
	CertificatePins []string `json:"certificate_pins" yaml:"certificate_pins"`
}

// RetryConfig represents retry configuration
type RetryConfig struct {
	Enabled     bool          `json:"enabled" yaml:"enabled"`
	MaxAttempts int           `json:"max_attempts" yaml:"max_attempts"`
	Backoff     time.Duration `json:"backoff" yaml:"backoff"`
	MaxBackoff  time.Duration `json:"max_backoff" yaml:"max_backoff"`
}

// AuthConfig represents authentication configuration
type AuthConfig struct {
	Type     string           `json:"type" yaml:"type"` // webauthn, oauth, apikey
	WebAuthn WebAuthnConfig   `json:"webauthn" yaml:"webauthn"`
	OAuth    OAuthConfig      `json:"oauth" yaml:"oauth"`
	APIKey   string           `json:"api_key" yaml:"api_key"`
	MFA      MFAConfig        `json:"mfa" yaml:"mfa"`
}

// WebAuthnConfig represents WebAuthn configuration
type WebAuthnConfig struct {
	Enabled  bool   `json:"enabled" yaml:"enabled"`
	RPID     string `json:"rp_id" yaml:"rp_id"`
	RPOrigin string `json:"rp_origin" yaml:"rp_origin"`
	Timeout  time.Duration `json:"timeout" yaml:"timeout"`
}

// OAuthConfig represents OAuth configuration
type OAuthConfig struct {
	Enabled      bool     `json:"enabled" yaml:"enabled"`
	ClientID     string   `json:"client_id" yaml:"client_id"`
	ClientSecret string   `json:"client_secret" yaml:"client_secret"`
	AuthURL      string   `json:"auth_url" yaml:"auth_url"`
	TokenURL     string   `json:"token_url" yaml:"token_url"`
	RedirectURL  string   `json:"redirect_url" yaml:"redirect_url"`
	Scopes       []string `json:"scopes" yaml:"scopes"`
}

// MFAConfig represents MFA configuration
type MFAConfig struct {
	Required bool   `json:"required" yaml:"required"`
	Type     string `json:"type" yaml:"type"` // totp, sms, email
}

// SessionConfig represents session management configuration
type SessionConfig struct {
	Type           string        `json:"type" yaml:"type"` // paseto, jwt
	TokenDuration  time.Duration `json:"token_duration" yaml:"token_duration"`
	RefreshEnabled bool          `json:"refresh_enabled" yaml:"refresh_enabled"`
	RefreshBefore  time.Duration `json:"refresh_before" yaml:"refresh_before"`
	Storage        string        `json:"storage" yaml:"storage"` // memory, keychain, file
	StoragePath    string        `json:"storage_path" yaml:"storage_path"`
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	return &Config{
		ServerEndpoint:         "192.168.1.38:14001",
		DefaultSecurityProfile: "enterprise",
		RequestTimeout:         30 * time.Second,
		ConnectionTimeout:      10 * time.Second,
		
		TransportConfig: TransportConfig{
			Type: "grpc",
			TLS: TLSConfig{
				Enabled: true,
			},
			Retry: RetryConfig{
				Enabled:     true,
				MaxAttempts: 3,
				Backoff:     1 * time.Second,
				MaxBackoff:  30 * time.Second,
			},
			StreamBufferSize: 1024,
			MaxMessageSize:   4 * 1024 * 1024, // 4MB
		},
		
		AuthConfig: AuthConfig{
			Type: "webauthn",
			WebAuthn: WebAuthnConfig{
				Timeout: 60 * time.Second,
			},
		},
		
		SessionConfig: SessionConfig{
			Type:           "paseto",
			TokenDuration:  15 * time.Minute,
			RefreshEnabled: true,
			RefreshBefore:  5 * time.Minute,
			Storage:        "memory",
		},
		
		LogLevel: "info",
	}
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	// Validate server endpoint
	if c.ServerEndpoint == "" {
		return errors.New("server endpoint is required")
	}
	
	// Parse and validate URL
	if _, err := url.Parse("https://" + c.ServerEndpoint); err != nil {
		return errors.New("invalid server endpoint")
	}
	
	// Validate transport type
	switch c.TransportConfig.Type {
	case "grpc", "grpcweb", "http":
		// Valid types
	default:
		return errors.New("invalid transport type")
	}
	
	// Validate auth type
	switch c.AuthConfig.Type {
	case "webauthn", "oauth", "apikey":
		// Valid types
	default:
		return errors.New("invalid auth type")
	}
	
	// Validate WebAuthn if enabled
	if c.AuthConfig.WebAuthn.Enabled {
		if c.AuthConfig.WebAuthn.RPID == "" {
			return errors.New("WebAuthn RP ID is required")
		}
		if c.AuthConfig.WebAuthn.RPOrigin == "" {
			return errors.New("WebAuthn RP Origin is required")
		}
	}
	
	// Validate OAuth if enabled
	if c.AuthConfig.OAuth.Enabled {
		if c.AuthConfig.OAuth.ClientID == "" {
			return errors.New("OAuth client ID is required")
		}
		if c.AuthConfig.OAuth.AuthURL == "" {
			return errors.New("OAuth auth URL is required")
		}
		if c.AuthConfig.OAuth.TokenURL == "" {
			return errors.New("OAuth token URL is required")
		}
	}
	
	// Validate session type
	switch c.SessionConfig.Type {
	case "paseto", "jwt":
		// Valid types
	default:
		return errors.New("invalid session type")
	}
	
	// Validate TLS if enabled
	if c.TransportConfig.TLS.Enabled {
		if c.TransportConfig.Type == "grpc" {
			// mTLS requires cert and key
			if c.TransportConfig.TLS.CertFile != "" && c.TransportConfig.TLS.KeyFile == "" {
				return errors.New("TLS key file is required when cert file is provided")
			}
			if c.TransportConfig.TLS.KeyFile != "" && c.TransportConfig.TLS.CertFile == "" {
				return errors.New("TLS cert file is required when key file is provided")
			}
		}
	}
	
	return nil
}