package config_test

import (
	"testing"
	"time"
	
	"github.com/stretchr/testify/assert"
	
	"github.com/macawi-ai/canidae/pkg/client/config"
)

func TestDefaultConfig(t *testing.T) {
	cfg := config.DefaultConfig()
	
	assert.NotNil(t, cfg)
	assert.Equal(t, "192.168.1.38:14001", cfg.ServerEndpoint)
	assert.Equal(t, "enterprise", cfg.DefaultSecurityProfile)
	assert.Equal(t, 30*time.Second, cfg.RequestTimeout)
	assert.Equal(t, 10*time.Second, cfg.ConnectionTimeout)
	
	// Transport config
	assert.Equal(t, "grpc", cfg.TransportConfig.Type)
	assert.True(t, cfg.TransportConfig.TLS.Enabled)
	assert.True(t, cfg.TransportConfig.Retry.Enabled)
	assert.Equal(t, 3, cfg.TransportConfig.Retry.MaxAttempts)
	assert.Equal(t, 1024, cfg.TransportConfig.StreamBufferSize)
	assert.Equal(t, 4*1024*1024, cfg.TransportConfig.MaxMessageSize)
	
	// Auth config
	assert.Equal(t, "apikey", cfg.AuthConfig.Type)
	assert.False(t, cfg.AuthConfig.WebAuthn.Enabled)
	assert.Equal(t, 60*time.Second, cfg.AuthConfig.WebAuthn.Timeout)
	
	// Session config
	assert.Equal(t, "memory", cfg.SessionConfig.Type)
	assert.Equal(t, 15*time.Minute, cfg.SessionConfig.TokenDuration)
	assert.True(t, cfg.SessionConfig.RefreshEnabled)
	assert.Equal(t, 5*time.Minute, cfg.SessionConfig.RefreshBefore)
	assert.Equal(t, "memory", cfg.SessionConfig.Storage)
	
	assert.Equal(t, "info", cfg.LogLevel)
}

func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name    string
		modify  func(*config.Config)
		wantErr bool
		errMsg  string
	}{
		{
			name:    "valid default config",
			modify:  func(c *config.Config) {},
			wantErr: false,
		},
		{
			name: "valid with custom endpoint",
			modify: func(c *config.Config) {
				c.ServerEndpoint = "example.com:8080"
			},
			wantErr: false,
		},
		{
			name: "empty server endpoint",
			modify: func(c *config.Config) {
				c.ServerEndpoint = ""
			},
			wantErr: true,
			errMsg:  "server endpoint is required",
		},
		{
			name: "invalid transport type",
			modify: func(c *config.Config) {
				c.TransportConfig.Type = "invalid"
			},
			wantErr: true,
			errMsg:  "invalid transport type",
		},
		{
			name: "valid transport types",
			modify: func(c *config.Config) {
				c.TransportConfig.Type = "grpcweb"
			},
			wantErr: false,
		},
		{
			name: "invalid auth type",
			modify: func(c *config.Config) {
				c.AuthConfig.Type = "invalid"
			},
			wantErr: true,
			errMsg:  "invalid auth type",
		},
		{
			name: "valid auth types",
			modify: func(c *config.Config) {
				c.AuthConfig.Type = "oauth"
			},
			wantErr: false,
		},
		{
			name: "WebAuthn missing RP ID",
			modify: func(c *config.Config) {
				c.AuthConfig.Type = "webauthn"
				c.AuthConfig.WebAuthn.Enabled = true
				c.AuthConfig.WebAuthn.RPID = ""
			},
			wantErr: true,
			errMsg:  "WebAuthn RP ID is required",
		},
		{
			name: "WebAuthn missing RP Origin",
			modify: func(c *config.Config) {
				c.AuthConfig.Type = "webauthn"
				c.AuthConfig.WebAuthn.Enabled = true
				c.AuthConfig.WebAuthn.RPID = "example.com"
				c.AuthConfig.WebAuthn.RPOrigin = ""
			},
			wantErr: true,
			errMsg:  "WebAuthn RP Origin is required",
		},
		{
			name: "valid WebAuthn config",
			modify: func(c *config.Config) {
				c.AuthConfig.Type = "webauthn"
				c.AuthConfig.WebAuthn.Enabled = true
				c.AuthConfig.WebAuthn.RPID = "example.com"
				c.AuthConfig.WebAuthn.RPOrigin = "https://example.com"
			},
			wantErr: false,
		},
		{
			name: "OAuth missing client ID",
			modify: func(c *config.Config) {
				c.AuthConfig.Type = "oauth"
				c.AuthConfig.OAuth.Enabled = true
				c.AuthConfig.OAuth.ClientID = ""
			},
			wantErr: true,
			errMsg:  "OAuth client ID is required",
		},
		{
			name: "OAuth missing auth URL",
			modify: func(c *config.Config) {
				c.AuthConfig.Type = "oauth"
				c.AuthConfig.OAuth.Enabled = true
				c.AuthConfig.OAuth.ClientID = "client-id"
				c.AuthConfig.OAuth.AuthURL = ""
			},
			wantErr: true,
			errMsg:  "OAuth auth URL is required",
		},
		{
			name: "OAuth missing token URL",
			modify: func(c *config.Config) {
				c.AuthConfig.Type = "oauth"
				c.AuthConfig.OAuth.Enabled = true
				c.AuthConfig.OAuth.ClientID = "client-id"
				c.AuthConfig.OAuth.AuthURL = "https://auth.example.com"
				c.AuthConfig.OAuth.TokenURL = ""
			},
			wantErr: true,
			errMsg:  "OAuth token URL is required",
		},
		{
			name: "valid OAuth config",
			modify: func(c *config.Config) {
				c.AuthConfig.Type = "oauth"
				c.AuthConfig.OAuth.Enabled = true
				c.AuthConfig.OAuth.ClientID = "client-id"
				c.AuthConfig.OAuth.AuthURL = "https://auth.example.com"
				c.AuthConfig.OAuth.TokenURL = "https://token.example.com"
			},
			wantErr: false,
		},
		{
			name: "invalid session type",
			modify: func(c *config.Config) {
				c.SessionConfig.Type = "invalid"
			},
			wantErr: true,
			errMsg:  "invalid session type",
		},
		{
			name: "valid session types - jwt",
			modify: func(c *config.Config) {
				c.SessionConfig.Type = "jwt"
			},
			wantErr: false,
		},
		{
			name: "valid session types - paseto",
			modify: func(c *config.Config) {
				c.SessionConfig.Type = "paseto"
			},
			wantErr: false,
		},
		{
			name: "valid session types - memory",
			modify: func(c *config.Config) {
				c.SessionConfig.Type = "memory"
			},
			wantErr: false,
		},
		{
			name: "TLS cert without key",
			modify: func(c *config.Config) {
				c.TransportConfig.Type = "grpc"
				c.TransportConfig.TLS.Enabled = true
				c.TransportConfig.TLS.CertFile = "cert.pem"
				c.TransportConfig.TLS.KeyFile = ""
			},
			wantErr: true,
			errMsg:  "TLS key file is required",
		},
		{
			name: "TLS key without cert",
			modify: func(c *config.Config) {
				c.TransportConfig.Type = "grpc"
				c.TransportConfig.TLS.Enabled = true
				c.TransportConfig.TLS.CertFile = ""
				c.TransportConfig.TLS.KeyFile = "key.pem"
			},
			wantErr: true,
			errMsg:  "TLS cert file is required",
		},
		{
			name: "valid TLS config",
			modify: func(c *config.Config) {
				c.TransportConfig.Type = "grpc"
				c.TransportConfig.TLS.Enabled = true
				c.TransportConfig.TLS.CertFile = "cert.pem"
				c.TransportConfig.TLS.KeyFile = "key.pem"
			},
			wantErr: false,
		},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := config.DefaultConfig()
			tt.modify(cfg)
			
			err := cfg.Validate()
			if tt.wantErr {
				assert.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestTransportConfig(t *testing.T) {
	cfg := config.TransportConfig{
		Type: "grpc",
		TLS: config.TLSConfig{
			Enabled:            true,
			CertFile:           "cert.pem",
			KeyFile:            "key.pem",
			CAFile:             "ca.pem",
			ServerName:         "server.example.com",
			InsecureSkipVerify: false,
			CertificatePins:    []string{"pin1", "pin2"},
		},
		Retry: config.RetryConfig{
			Enabled:     true,
			MaxAttempts: 5,
			Backoff:     2 * time.Second,
			MaxBackoff:  60 * time.Second,
		},
		StreamBufferSize: 2048,
		MaxMessageSize:   8 * 1024 * 1024,
	}
	
	assert.Equal(t, "grpc", cfg.Type)
	assert.True(t, cfg.TLS.Enabled)
	assert.Equal(t, "cert.pem", cfg.TLS.CertFile)
	assert.Equal(t, "key.pem", cfg.TLS.KeyFile)
	assert.Equal(t, "ca.pem", cfg.TLS.CAFile)
	assert.Equal(t, "server.example.com", cfg.TLS.ServerName)
	assert.False(t, cfg.TLS.InsecureSkipVerify)
	assert.Len(t, cfg.TLS.CertificatePins, 2)
	
	assert.True(t, cfg.Retry.Enabled)
	assert.Equal(t, 5, cfg.Retry.MaxAttempts)
	assert.Equal(t, 2*time.Second, cfg.Retry.Backoff)
	assert.Equal(t, 60*time.Second, cfg.Retry.MaxBackoff)
	
	assert.Equal(t, 2048, cfg.StreamBufferSize)
	assert.Equal(t, 8*1024*1024, cfg.MaxMessageSize)
}

func TestAuthConfig(t *testing.T) {
	cfg := config.AuthConfig{
		Type: "webauthn",
		WebAuthn: config.WebAuthnConfig{
			Enabled:  true,
			RPID:     "example.com",
			RPOrigin: "https://example.com",
			Timeout:  120 * time.Second,
		},
		OAuth: config.OAuthConfig{
			Enabled:      true,
			ClientID:     "client-123",
			ClientSecret: "secret-456",
			AuthURL:      "https://auth.example.com/authorize",
			TokenURL:     "https://auth.example.com/token",
			RedirectURL:  "https://example.com/callback",
			Scopes:       []string{"read", "write"},
		},
		APIKey: "api-key-789",
		MFA: config.MFAConfig{
			Required: true,
			Type:     "totp",
		},
	}
	
	assert.Equal(t, "webauthn", cfg.Type)
	
	// WebAuthn config
	assert.True(t, cfg.WebAuthn.Enabled)
	assert.Equal(t, "example.com", cfg.WebAuthn.RPID)
	assert.Equal(t, "https://example.com", cfg.WebAuthn.RPOrigin)
	assert.Equal(t, 120*time.Second, cfg.WebAuthn.Timeout)
	
	// OAuth config
	assert.True(t, cfg.OAuth.Enabled)
	assert.Equal(t, "client-123", cfg.OAuth.ClientID)
	assert.Equal(t, "secret-456", cfg.OAuth.ClientSecret)
	assert.Equal(t, "https://auth.example.com/authorize", cfg.OAuth.AuthURL)
	assert.Equal(t, "https://auth.example.com/token", cfg.OAuth.TokenURL)
	assert.Equal(t, "https://example.com/callback", cfg.OAuth.RedirectURL)
	assert.Len(t, cfg.OAuth.Scopes, 2)
	
	// API Key
	assert.Equal(t, "api-key-789", cfg.APIKey)
	
	// MFA config
	assert.True(t, cfg.MFA.Required)
	assert.Equal(t, "totp", cfg.MFA.Type)
}

func TestSessionConfig(t *testing.T) {
	cfg := config.SessionConfig{
		Type:           "paseto",
		TokenDuration:  30 * time.Minute,
		RefreshEnabled: true,
		RefreshBefore:  10 * time.Minute,
		Storage:        "keychain",
		StoragePath:    "/tmp/sessions",
	}
	
	assert.Equal(t, "paseto", cfg.Type)
	assert.Equal(t, 30*time.Minute, cfg.TokenDuration)
	assert.True(t, cfg.RefreshEnabled)
	assert.Equal(t, 10*time.Minute, cfg.RefreshBefore)
	assert.Equal(t, "keychain", cfg.Storage)
	assert.Equal(t, "/tmp/sessions", cfg.StoragePath)
}

// BenchmarkConfigValidation benchmarks config validation
func BenchmarkConfigValidation(b *testing.B) {
	cfg := config.DefaultConfig()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cfg.Validate()
	}
}

// BenchmarkDefaultConfig benchmarks creating default config
func BenchmarkDefaultConfig(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = config.DefaultConfig()
	}
}