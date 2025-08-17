package session

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/macawi-ai/canidae/pkg/client/auth"
	"github.com/macawi-ai/canidae/pkg/client/config"
)

// Common errors
var (
	ErrNoSession      = errors.New("no active session")
	ErrSessionExpired = errors.New("session expired")
	ErrInvalidSession = errors.New("invalid session")
)

// Manager manages client sessions
type Manager interface {
	// CreateSession creates a new session from an auth token
	CreateSession(ctx context.Context, token *auth.Token) (*Session, error)

	// GetSession returns the current session
	GetSession() (*Session, error)

	// RefreshSession refreshes the current session
	RefreshSession(ctx context.Context) (*Session, error)

	// RevokeSession revokes the current session
	RevokeSession(ctx context.Context) error

	// IsValid checks if the current session is valid
	IsValid() bool

	// SetAutoRefresh enables automatic session refresh
	SetAutoRefresh(enabled bool)
}

// Session represents an active session
type Session struct {
	ID           string            `json:"id"`
	Token        *auth.Token       `json:"token"`
	PackID       string            `json:"pack_id,omitempty"`
	UserID       string            `json:"user_id,omitempty"`
	CreatedAt    time.Time         `json:"created_at"`
	LastActivity time.Time         `json:"last_activity"`
	Metadata     map[string]string `json:"metadata,omitempty"`
}

// IsExpired checks if the session is expired
func (s *Session) IsExpired() bool {
	return s.Token.IsExpired()
}

// ShouldRefresh checks if the session should be refreshed
func (s *Session) ShouldRefresh(before time.Duration) bool {
	return s.Token.ShouldRefresh(before)
}

// NewManager creates a new session manager
func NewManager(cfg config.SessionConfig) (Manager, error) {
	switch cfg.Type {
	case "paseto":
		return NewPASETOManager(cfg)
	case "jwt":
		return NewJWTManager(cfg)
	default:
		return &memoryManager{
			config:      cfg,
			autoRefresh: cfg.RefreshEnabled,
		}, nil
	}
}

// NewPASETOManager creates a new PASETO session manager (to be implemented)
func NewPASETOManager(cfg config.SessionConfig) (Manager, error) {
	// This will be implemented in paseto.go
	return nil, errors.New("PASETO session manager not yet implemented")
}

// NewJWTManager creates a new JWT session manager (to be implemented)
func NewJWTManager(cfg config.SessionConfig) (Manager, error) {
	// This will be implemented in jwt.go
	return nil, errors.New("JWT session manager not yet implemented")
}

// memoryManager implements an in-memory session manager
type memoryManager struct {
	config      config.SessionConfig
	session     *Session
	autoRefresh bool
	mu          sync.RWMutex
}

func (m *memoryManager) CreateSession(ctx context.Context, token *auth.Token) (*Session, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	session := &Session{
		ID:           generateSessionID(),
		Token:        token,
		CreatedAt:    time.Now(),
		LastActivity: time.Now(),
	}

	m.session = session

	// Start auto-refresh if enabled
	if m.autoRefresh {
		go m.autoRefreshLoop(ctx)
	}

	return session, nil
}

func (m *memoryManager) GetSession() (*Session, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.session == nil {
		return nil, ErrNoSession
	}

	if m.session.IsExpired() {
		return nil, ErrSessionExpired
	}

	// Update last activity
	m.session.LastActivity = time.Now()

	return m.session, nil
}

func (m *memoryManager) RefreshSession(ctx context.Context) (*Session, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.session == nil {
		return nil, ErrNoSession
	}

	// In a real implementation, this would call the auth provider to refresh
	// For now, we'll just extend the expiration
	m.session.Token.ExpiresAt = time.Now().Add(m.config.TokenDuration)
	m.session.LastActivity = time.Now()

	return m.session, nil
}

func (m *memoryManager) RevokeSession(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.session = nil
	return nil
}

func (m *memoryManager) IsValid() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return m.session != nil && !m.session.IsExpired()
}

func (m *memoryManager) SetAutoRefresh(enabled bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.autoRefresh = enabled
}

func (m *memoryManager) autoRefreshLoop(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if m.session != nil && m.session.ShouldRefresh(m.config.RefreshBefore) {
				_, _ = m.RefreshSession(ctx)
			}
		}
	}
}

// generateSessionID generates a unique session ID
func generateSessionID() string {
	// In a real implementation, use a proper UUID library
	return "session-" + time.Now().Format("20060102150405")
}
