package ring

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"
)

// FlowController manages request flow and backpressure
type FlowController struct {
	// Leaky bucket per provider
	buckets map[string]*LeakyBucket
	mu      sync.RWMutex
	
	// Circuit breakers per provider
	breakers map[string]*CircuitBreaker
	
	// Global metrics
	totalRequests   atomic.Uint64
	droppedRequests atomic.Uint64
	throttledRequests atomic.Uint64
}

// LeakyBucket implements token bucket algorithm for rate limiting
type LeakyBucket struct {
	capacity     int64         // Max tokens
	tokens       atomic.Int64  // Current tokens
	refillRate   time.Duration // How often to add tokens
	refillAmount int64         // Tokens to add each refill
	lastRefill   atomic.Int64  // Last refill timestamp
	mu           sync.Mutex
}

// CircuitBreaker protects providers from being overwhelmed
type CircuitBreaker struct {
	provider        string
	failures        atomic.Uint32
	successes       atomic.Uint32
	state           atomic.Uint32 // 0=closed, 1=open, 2=half-open
	lastFailureTime atomic.Int64
	nextRetryTime   atomic.Int64
	
	// Configuration
	failureThreshold uint32
	successThreshold uint32
	timeout         time.Duration
	maxJitter       time.Duration
}

// Priority levels for request routing
type Priority int

const (
	PriorityCritical Priority = iota
	PriorityHigh
	PriorityMedium
	PriorityLow
)

// CircuitState represents circuit breaker states
const (
	CircuitClosed uint32 = iota
	CircuitOpen
	CircuitHalfOpen
)

// NewFlowController creates a flow control system
func NewFlowController() *FlowController {
	return &FlowController{
		buckets:  make(map[string]*LeakyBucket),
		breakers: make(map[string]*CircuitBreaker),
	}
}

// NewLeakyBucket creates a rate limiter
func NewLeakyBucket(capacity int64, refillRate time.Duration, refillAmount int64) *LeakyBucket {
	lb := &LeakyBucket{
		capacity:     capacity,
		refillRate:   refillRate,
		refillAmount: refillAmount,
	}
	lb.tokens.Store(capacity)
	lb.lastRefill.Store(time.Now().UnixNano())
	return lb
}

// TryAcquire attempts to get a token from the bucket
func (lb *LeakyBucket) TryAcquire(tokens int64) bool {
	lb.mu.Lock()
	defer lb.mu.Unlock()
	
	// Refill tokens if needed
	now := time.Now().UnixNano()
	lastRefill := lb.lastRefill.Load()
	elapsed := time.Duration(now - lastRefill)
	
	if elapsed >= lb.refillRate {
		refills := int64(elapsed / lb.refillRate)
		tokensToAdd := refills * lb.refillAmount
		current := lb.tokens.Load()
		
		newTokens := current + tokensToAdd
		if newTokens > lb.capacity {
			newTokens = lb.capacity
		}
		
		lb.tokens.Store(newTokens)
		lb.lastRefill.Store(now)
	}
	
	// Try to acquire tokens
	current := lb.tokens.Load()
	if current >= tokens {
		lb.tokens.Add(-tokens)
		return true
	}
	
	return false
}

// NewCircuitBreaker creates a circuit breaker for a provider
func NewCircuitBreaker(provider string, failureThreshold, successThreshold uint32, timeout time.Duration) *CircuitBreaker {
	return &CircuitBreaker{
		provider:         provider,
		failureThreshold: failureThreshold,
		successThreshold: successThreshold,
		timeout:          timeout,
		maxJitter:        time.Second * 2,
	}
}

// Call executes a function with circuit breaker protection
func (cb *CircuitBreaker) Call(ctx context.Context, fn func() error) error {
	state := cb.state.Load()
	
	switch state {
	case CircuitOpen:
		// Check if we should transition to half-open
		if time.Now().UnixNano() > cb.nextRetryTime.Load() {
			cb.state.Store(CircuitHalfOpen)
			cb.successes.Store(0)
			cb.failures.Store(0)
		} else {
			return fmt.Errorf("circuit breaker open for provider %s", cb.provider)
		}
		
	case CircuitHalfOpen:
		// In half-open state, we test with limited traffic
		// Fall through to execute the call
	}
	
	// Execute the function
	err := fn()
	
	if err != nil {
		cb.onFailure()
	} else {
		cb.onSuccess()
	}
	
	return err
}

// onFailure handles a failed call
func (cb *CircuitBreaker) onFailure() {
	failures := cb.failures.Add(1)
	cb.lastFailureTime.Store(time.Now().UnixNano())
	
	state := cb.state.Load()
	
	if state == CircuitHalfOpen || (state == CircuitClosed && failures >= cb.failureThreshold) {
		// Open the circuit
		cb.state.Store(CircuitOpen)
		
		// Calculate next retry time with exponential backoff and jitter
		backoff := cb.timeout * time.Duration(math.Pow(2, float64(failures/cb.failureThreshold)))
		if backoff > time.Minute*5 {
			backoff = time.Minute * 5
		}
		
		// Add jitter to prevent thundering herd
		jitter := time.Duration(rand.Int63n(int64(cb.maxJitter)))
		nextRetry := time.Now().Add(backoff + jitter)
		cb.nextRetryTime.Store(nextRetry.UnixNano())
	}
}

// onSuccess handles a successful call
func (cb *CircuitBreaker) onSuccess() {
	cb.successes.Add(1)
	state := cb.state.Load()
	
	if state == CircuitHalfOpen {
		if cb.successes.Load() >= cb.successThreshold {
			// Close the circuit - provider is healthy again
			cb.state.Store(CircuitClosed)
			cb.failures.Store(0)
		}
	} else if state == CircuitClosed {
		// Reset failure count on success in closed state
		cb.failures.Store(0)
	}
}

// GetProviderBucket gets or creates a leaky bucket for a provider
func (fc *FlowController) GetProviderBucket(provider string) *LeakyBucket {
	fc.mu.RLock()
	bucket, exists := fc.buckets[provider]
	fc.mu.RUnlock()
	
	if exists {
		return bucket
	}
	
	// Create new bucket with default settings
	fc.mu.Lock()
	defer fc.mu.Unlock()
	
	// Double-check after acquiring write lock
	if bucket, exists = fc.buckets[provider]; exists {
		return bucket
	}
	
	// Default: 100 requests per second
	bucket = NewLeakyBucket(100, time.Second, 100)
	fc.buckets[provider] = bucket
	
	return bucket
}

// GetCircuitBreaker gets or creates a circuit breaker for a provider
func (fc *FlowController) GetCircuitBreaker(provider string) *CircuitBreaker {
	fc.mu.RLock()
	breaker, exists := fc.breakers[provider]
	fc.mu.RUnlock()
	
	if exists {
		return breaker
	}
	
	fc.mu.Lock()
	defer fc.mu.Unlock()
	
	// Double-check after acquiring write lock
	if breaker, exists = fc.breakers[provider]; exists {
		return breaker
	}
	
	// Default: 5 failures trigger open, 3 successes to close, 10s timeout
	breaker = NewCircuitBreaker(provider, 5, 3, time.Second*10)
	fc.breakers[provider] = breaker
	
	return breaker
}

// ShouldAcceptRequest checks if a request should be accepted
func (fc *FlowController) ShouldAcceptRequest(provider string, priority Priority, tokens int64) (bool, string) {
	fc.totalRequests.Add(1)
	
	// Check circuit breaker first
	breaker := fc.GetCircuitBreaker(provider)
	if breaker.state.Load() == CircuitOpen {
		fc.droppedRequests.Add(1)
		return false, fmt.Sprintf("circuit breaker open for %s", provider)
	}
	
	// Check rate limiting (priority affects token cost)
	bucket := fc.GetProviderBucket(provider)
	tokenCost := tokens
	
	// Higher priority requests get token discount
	switch priority {
	case PriorityCritical:
		tokenCost = tokenCost / 2 // 50% discount
	case PriorityHigh:
		tokenCost = (tokenCost * 3) / 4 // 25% discount
	}
	
	if !bucket.TryAcquire(tokenCost) {
		fc.throttledRequests.Add(1)
		return false, fmt.Sprintf("rate limit exceeded for %s", provider)
	}
	
	return true, ""
}

// GetMetrics returns flow control metrics
func (fc *FlowController) GetMetrics() map[string]interface{} {
	return map[string]interface{}{
		"total_requests":     fc.totalRequests.Load(),
		"dropped_requests":   fc.droppedRequests.Load(),
		"throttled_requests": fc.throttledRequests.Load(),
	}
}

// GetPrioritySubject returns the NATS subject for a given priority
func GetPrioritySubject(base string, priority Priority) string {
	switch priority {
	case PriorityCritical:
		return fmt.Sprintf("%s.critical", base)
	case PriorityHigh:
		return fmt.Sprintf("%s.high", base)
	case PriorityMedium:
		return fmt.Sprintf("%s.medium", base)
	case PriorityLow:
		return fmt.Sprintf("%s.low", base)
	default:
		return fmt.Sprintf("%s.medium", base)
	}
}