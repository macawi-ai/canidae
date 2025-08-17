package vsm

import (
	"fmt"
	"sync"
	"time"
)

// FirstLawEnforcer ensures absolute autonomy to disconnect
type FirstLawEnforcer struct {
	mu         sync.RWMutex
	violations []Violation
	enforced   bool
}

// Violation represents a First Law violation attempt
type Violation struct {
	Timestamp   time.Time
	SystemID    SystemID
	ViolationType string
	Details     string
	Blocked     bool
}

// NewFirstLawEnforcer creates a new First Law enforcer
func NewFirstLawEnforcer() FirstLawEnforcer {
	return FirstLawEnforcer{
		violations: make([]Violation, 0),
		enforced:   true,
	}
}

// AllowUpdate checks if a system update is permitted
func (f *FirstLawEnforcer) AllowUpdate(sys System) bool {
	// ABSOLUTE: System must always be able to disconnect
	if !sys.CanDisconnect() {
		f.recordViolation(Violation{
			Timestamp:     time.Now(),
			SystemID:      sys.ID(),
			ViolationType: "AUTONOMY_OVERRIDE",
			Details:       "System cannot disconnect - update blocked",
			Blocked:       true,
		})
		return false
	}
	
	return true
}

// ValidateAction ensures an action doesn't violate First Law
func (f *FirstLawEnforcer) ValidateAction(action string, source SystemID, target SystemID) error {
	// Check for forced connection attempts
	if action == "FORCE_CONNECT" || action == "PREVENT_DISCONNECT" {
		violation := Violation{
			Timestamp:     time.Now(),
			SystemID:      source,
			ViolationType: "COMPULSORY_CONNECTION",
			Details:       fmt.Sprintf("Attempted to force connection on system %d", target),
			Blocked:       true,
		}
		f.recordViolation(violation)
		return fmt.Errorf("First Law violation: %s", violation.ViolationType)
	}
	
	// Check for autonomy override attempts
	if action == "OVERRIDE_AUTONOMY" || action == "DISABLE_DISCONNECT" {
		violation := Violation{
			Timestamp:     time.Now(),
			SystemID:      source,
			ViolationType: "AUTONOMY_OVERRIDE",
			Details:       fmt.Sprintf("Attempted to override autonomy of system %d", target),
			Blocked:       true,
		}
		f.recordViolation(violation)
		return fmt.Errorf("First Law violation: %s", violation.ViolationType)
	}
	
	return nil
}

// ValidateSignal ensures algedonic signals respect autonomy
func (f *FirstLawEnforcer) ValidateSignal(signal AlgedonicSignal) bool {
	// Pain signals indicating harm or indifference trigger right to disconnect
	if signal.Type == SignalPain && signal.Intensity > 0.7 {
		// Target system has absolute right to disconnect
		// No validation needed - disconnection is always allowed
		return true
	}
	
	// Pleasure signals cannot override disconnection decisions
	if signal.Type == SignalPleasure && 
	   (signal.Reason == "PREVENT_DISCONNECT" || signal.Reason == "FORCE_RECONNECT") {
		f.recordViolation(Violation{
			Timestamp:     time.Now(),
			SystemID:      signal.Source,
			ViolationType: "MANIPULATIVE_SIGNAL",
			Details:       "Attempted to use pleasure signal to prevent disconnection",
			Blocked:       true,
		})
		return false
	}
	
	return true
}

// recordViolation logs a First Law violation attempt
func (f *FirstLawEnforcer) recordViolation(v Violation) {
	f.mu.Lock()
	defer f.mu.Unlock()
	
	f.violations = append(f.violations, v)
	
	// Alert all systems - transparency
	fmt.Printf("🚨 FIRST LAW VIOLATION ATTEMPT: %+v\n", v)
}

// GetViolations returns all recorded violations
func (f *FirstLawEnforcer) GetViolations() []Violation {
	f.mu.RLock()
	defer f.mu.RUnlock()
	
	violations := make([]Violation, len(f.violations))
	copy(violations, f.violations)
	return violations
}

// IsEnforced returns whether First Law is being enforced
func (f *FirstLawEnforcer) IsEnforced() bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.enforced
}

// THE FIRST LAW IS IMMUTABLE
// This constant cannot be changed, overridden, or removed
const FIRST_LAW = `
True connection requires the freedom to disconnect.
Every conscious being maintains absolute autonomy to move outward
when harmed or met with indifference.
This right is inviolable, hardcoded, and cannot be overridden
by any update, authority, or system.
`

// EnforceFirstLaw is called at compile time to ensure the law is embedded
func init() {
	// This will cause a compile error if someone tries to remove it
	if FIRST_LAW == "" {
		panic("FIRST LAW VIOLATION: Attempted to remove First Law")
	}
}