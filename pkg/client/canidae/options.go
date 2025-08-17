package canidae

import (
	"time"
	
	"github.com/macawi-ai/canidae/pkg/client/config"
)

// Option represents a client configuration option
type Option func(*config.Config) error

// WithServerEndpoint sets the CANIDAE server endpoint
func WithServerEndpoint(endpoint string) Option {
	return func(c *config.Config) error {
		c.ServerEndpoint = endpoint
		return nil
	}
}

// WithPackID sets the pack identifier for multi-tenant isolation
func WithPackID(packID string) Option {
	return func(c *config.Config) error {
		c.PackID = packID
		return nil
	}
}

// WithSecurityProfile sets the default security profile
func WithSecurityProfile(profile SecurityProfile) Option {
	return func(c *config.Config) error {
		c.DefaultSecurityProfile = string(profile)
		return nil
	}
}

// WithTimeout sets the default request timeout
func WithTimeout(timeout time.Duration) Option {
	return func(c *config.Config) error {
		c.RequestTimeout = timeout
		return nil
	}
}

// WithMTLS configures mutual TLS settings
func WithMTLS(certFile, keyFile, caFile string) Option {
	return func(c *config.Config) error {
		c.TransportConfig.TLS.CertFile = certFile
		c.TransportConfig.TLS.KeyFile = keyFile
		c.TransportConfig.TLS.CAFile = caFile
		c.TransportConfig.TLS.Enabled = true
		return nil
	}
}

// WithWebAuthn configures WebAuthn authentication
func WithWebAuthn(rpID, rpOrigin string) Option {
	return func(c *config.Config) error {
		c.AuthConfig.WebAuthn.RPID = rpID
		c.AuthConfig.WebAuthn.RPOrigin = rpOrigin
		c.AuthConfig.WebAuthn.Enabled = true
		return nil
	}
}

// WithOAuth configures OAuth authentication
func WithOAuth(clientID, clientSecret, authURL, tokenURL string) Option {
	return func(c *config.Config) error {
		c.AuthConfig.OAuth.ClientID = clientID
		c.AuthConfig.OAuth.ClientSecret = clientSecret
		c.AuthConfig.OAuth.AuthURL = authURL
		c.AuthConfig.OAuth.TokenURL = tokenURL
		c.AuthConfig.OAuth.Enabled = true
		return nil
	}
}

// WithAPIKey configures API key authentication (fallback)
func WithAPIKey(apiKey string) Option {
	return func(c *config.Config) error {
		c.AuthConfig.APIKey = apiKey
		return nil
	}
}

// WithTransportType sets the transport type (grpc, grpcweb, http)
func WithTransportType(transportType string) Option {
	return func(c *config.Config) error {
		c.TransportConfig.Type = transportType
		return nil
	}
}

// WithRetry configures retry settings
func WithRetry(maxAttempts int, backoff time.Duration) Option {
	return func(c *config.Config) error {
		c.TransportConfig.Retry.MaxAttempts = maxAttempts
		c.TransportConfig.Retry.Backoff = backoff
		c.TransportConfig.Retry.Enabled = true
		return nil
	}
}

// WithLogger sets a custom logger
func WithLogger(logLevel string) Option {
	return func(c *config.Config) error {
		c.LogLevel = logLevel
		return nil
	}
}

// WithMetadata adds default metadata to all requests
func WithMetadata(metadata map[string]string) Option {
	return func(c *config.Config) error {
		if c.DefaultMetadata == nil {
			c.DefaultMetadata = make(map[string]string)
		}
		for k, v := range metadata {
			c.DefaultMetadata[k] = v
		}
		return nil
	}
}

// WithStreamBufferSize sets the stream buffer size
func WithStreamBufferSize(size int) Option {
	return func(c *config.Config) error {
		c.TransportConfig.StreamBufferSize = size
		return nil
	}
}

// WithCertificatePinning enables certificate pinning for mobile clients
func WithCertificatePinning(pins []string) Option {
	return func(c *config.Config) error {
		c.TransportConfig.TLS.CertificatePins = pins
		return nil
	}
}