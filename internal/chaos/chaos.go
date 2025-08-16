package chaos

import (
	"context"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// ChaosMonkey introduces controlled chaos for resilience testing
type ChaosMonkey struct {
	enabled atomic.Bool
	config  Config
	mu      sync.RWMutex
	
	// Chaos metrics
	messagesDropped   atomic.Uint64
	providersSlowed   atomic.Uint64
	subjectsDisrupted atomic.Uint64
	consumersKilled   atomic.Uint64
	
	// Active chaos effects
	activeEffects map[string]*Effect
}

// Config defines chaos parameters
type Config struct {
	// Probability of chaos events (0.0 - 1.0)
	MessageDropRate    float64
	ProviderSlowRate   float64
	SubjectFailureRate float64
	ConsumerCrashRate  float64
	
	// Chaos intensity
	SlowdownMin      time.Duration
	SlowdownMax      time.Duration
	OutageDurationMin time.Duration
	OutageDurationMax time.Duration
	
	// Safety limits
	MaxConcurrentEffects int
	SafeMode            bool // Reduces chaos intensity
}

// Effect represents an active chaos effect
type Effect struct {
	Type      EffectType
	Target    string
	StartTime time.Time
	EndTime   time.Time
	Intensity float64
}

// EffectType defines types of chaos
type EffectType string

const (
	EffectDropMessage   EffectType = "drop_message"
	EffectSlowProvider  EffectType = "slow_provider"
	EffectFailSubject   EffectType = "fail_subject"
	EffectCrashConsumer EffectType = "crash_consumer"
	EffectNetworkSplit  EffectType = "network_split"
	EffectMemoryPressure EffectType = "memory_pressure"
)

// DefaultConfig returns safe default chaos configuration
func DefaultConfig() Config {
	return Config{
		MessageDropRate:    0.01,  // 1% message drop
		ProviderSlowRate:   0.05,  // 5% provider slowdown
		SubjectFailureRate: 0.001, // 0.1% subject failure
		ConsumerCrashRate:  0.001, // 0.1% consumer crash
		
		SlowdownMin:       time.Millisecond * 100,
		SlowdownMax:       time.Second * 2,
		OutageDurationMin: time.Second * 5,
		OutageDurationMax: time.Second * 30,
		
		MaxConcurrentEffects: 3,
		SafeMode:            true,
	}
}

// NewChaosMonkey creates a chaos testing component
func NewChaosMonkey(config Config) *ChaosMonkey {
	cm := &ChaosMonkey{
		config:        config,
		activeEffects: make(map[string]*Effect),
	}
	
	// Start with chaos disabled by default
	cm.enabled.Store(false)
	
	return cm
}

// Enable activates chaos testing
func (cm *ChaosMonkey) Enable() {
	cm.enabled.Store(true)
	log.Println("🙈 CHAOS MONKEY ENABLED - May the odds be ever in your favor")
}

// Disable deactivates chaos testing
func (cm *ChaosMonkey) Disable() {
	cm.enabled.Store(false)
	log.Println("🙊 CHAOS MONKEY DISABLED - Peace restored")
}

// ShouldDropMessage decides if a message should be dropped
func (cm *ChaosMonkey) ShouldDropMessage() bool {
	if !cm.enabled.Load() {
		return false
	}
	
	if cm.config.SafeMode && cm.hasMaxEffects() {
		return false
	}
	
	if rand.Float64() < cm.config.MessageDropRate {
		cm.messagesDropped.Add(1)
		log.Printf("🙈 CHAOS: Dropping message (total dropped: %d)", 
			cm.messagesDropped.Load())
		return true
	}
	
	return false
}

// ShouldSlowProvider injects latency for a provider
func (cm *ChaosMonkey) ShouldSlowProvider(provider string) time.Duration {
	if !cm.enabled.Load() {
		return 0
	}
	
	if cm.config.SafeMode && cm.hasMaxEffects() {
		return 0
	}
	
	if rand.Float64() < cm.config.ProviderSlowRate {
		// Calculate random slowdown duration
		min := cm.config.SlowdownMin.Nanoseconds()
		max := cm.config.SlowdownMax.Nanoseconds()
		duration := time.Duration(min + rand.Int63n(max-min))
		
		cm.providersSlowed.Add(1)
		
		// Record the effect
		cm.addEffect(&Effect{
			Type:      EffectSlowProvider,
			Target:    provider,
			StartTime: time.Now(),
			EndTime:   time.Now().Add(duration),
			Intensity: float64(duration) / float64(cm.config.SlowdownMax),
		})
		
		log.Printf("🙈 CHAOS: Slowing provider %s by %v", provider, duration)
		return duration
	}
	
	return 0
}

// ShouldFailSubject simulates NATS subject unavailability
func (cm *ChaosMonkey) ShouldFailSubject(subject string) bool {
	if !cm.enabled.Load() {
		return false
	}
	
	if cm.config.SafeMode && cm.hasMaxEffects() {
		return false
	}
	
	// Check if subject is already failing
	cm.mu.RLock()
	if effect, exists := cm.activeEffects[subject]; exists {
		failing := effect.Type == EffectFailSubject && time.Now().Before(effect.EndTime)
		cm.mu.RUnlock()
		return failing
	}
	cm.mu.RUnlock()
	
	if rand.Float64() < cm.config.SubjectFailureRate {
		// Calculate outage duration
		min := cm.config.OutageDurationMin.Nanoseconds()
		max := cm.config.OutageDurationMax.Nanoseconds()
		duration := time.Duration(min + rand.Int63n(max-min))
		
		cm.subjectsDisrupted.Add(1)
		
		// Record the effect
		cm.addEffect(&Effect{
			Type:      EffectFailSubject,
			Target:    subject,
			StartTime: time.Now(),
			EndTime:   time.Now().Add(duration),
			Intensity: 1.0,
		})
		
		log.Printf("🙈 CHAOS: Subject %s unavailable for %v", subject, duration)
		return true
	}
	
	return false
}

// SimulateConsumerCrash forces a consumer to restart
func (cm *ChaosMonkey) SimulateConsumerCrash(consumerID string) bool {
	if !cm.enabled.Load() {
		return false
	}
	
	if cm.config.SafeMode && cm.hasMaxEffects() {
		return false
	}
	
	if rand.Float64() < cm.config.ConsumerCrashRate {
		cm.consumersKilled.Add(1)
		
		log.Printf("🙈 CHAOS: Crashing consumer %s (total crashes: %d)", 
			consumerID, cm.consumersKilled.Load())
		
		// Record the effect
		cm.addEffect(&Effect{
			Type:      EffectCrashConsumer,
			Target:    consumerID,
			StartTime: time.Now(),
			EndTime:   time.Now().Add(time.Second * 5), // Recovery time
			Intensity: 1.0,
		})
		
		return true
	}
	
	return false
}

// SimulateNetworkSplit creates a network partition
func (cm *ChaosMonkey) SimulateNetworkSplit(duration time.Duration) {
	if !cm.enabled.Load() {
		return
	}
	
	log.Printf("🙈 CHAOS: NETWORK SPLIT for %v - Pack is divided!", duration)
	
	cm.addEffect(&Effect{
		Type:      EffectNetworkSplit,
		Target:    "network",
		StartTime: time.Now(),
		EndTime:   time.Now().Add(duration),
		Intensity: 1.0,
	})
	
	// In a real implementation, this would manipulate network rules
}

// ApplyChaos wraps a function with potential chaos
func (cm *ChaosMonkey) ApplyChaos(ctx context.Context, target string, fn func() error) error {
	// Check for message drop
	if cm.ShouldDropMessage() {
		return fmt.Errorf("chaos: message dropped")
	}
	
	// Check for slowdown
	if delay := cm.ShouldSlowProvider(target); delay > 0 {
		select {
		case <-time.After(delay):
			// Continue after delay
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	
	// Execute the function
	err := fn()
	
	// Randomly inject errors
	if cm.enabled.Load() && rand.Float64() < 0.01 { // 1% random error
		return fmt.Errorf("chaos: random failure injected")
	}
	
	return err
}

// addEffect records an active chaos effect
func (cm *ChaosMonkey) addEffect(effect *Effect) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	cm.activeEffects[effect.Target] = effect
	
	// Clean up expired effects
	now := time.Now()
	for target, e := range cm.activeEffects {
		if now.After(e.EndTime) {
			delete(cm.activeEffects, target)
		}
	}
}

// hasMaxEffects checks if maximum concurrent effects reached
func (cm *ChaosMonkey) hasMaxEffects() bool {
	cm.mu.RLock()
	defer cm.mu.RUnlock()
	
	activeCount := 0
	now := time.Now()
	
	for _, effect := range cm.activeEffects {
		if now.Before(effect.EndTime) {
			activeCount++
		}
	}
	
	return activeCount >= cm.config.MaxConcurrentEffects
}

// GetMetrics returns chaos metrics
func (cm *ChaosMonkey) GetMetrics() map[string]interface{} {
	cm.mu.RLock()
	activeCount := len(cm.activeEffects)
	cm.mu.RUnlock()
	
	return map[string]interface{}{
		"enabled":            cm.enabled.Load(),
		"messages_dropped":   cm.messagesDropped.Load(),
		"providers_slowed":   cm.providersSlowed.Load(),
		"subjects_disrupted": cm.subjectsDisrupted.Load(),
		"consumers_killed":   cm.consumersKilled.Load(),
		"active_effects":     activeCount,
	}
}

// Reset clears all metrics and effects
func (cm *ChaosMonkey) Reset() {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	
	cm.messagesDropped.Store(0)
	cm.providersSlowed.Store(0)
	cm.subjectsDisrupted.Store(0)
	cm.consumersKilled.Store(0)
	cm.activeEffects = make(map[string]*Effect)
	
	log.Println("🙊 CHAOS MONKEY: Metrics reset")
}