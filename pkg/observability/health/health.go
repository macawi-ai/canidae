// Package health provides health check functionality for CANIDAE components
package health

import (
	"context"
	"encoding/json"
	"net/http"
	"sync"
	"time"
)

// Status represents the health status of a component
type Status string

const (
	StatusHealthy   Status = "healthy"
	StatusDegraded  Status = "degraded"
	StatusUnhealthy Status = "unhealthy"
)

// CheckResult represents the result of a health check
type CheckResult struct {
	Status    Status                 `json:"status"`
	Timestamp time.Time              `json:"timestamp"`
	Duration  time.Duration          `json:"duration_ms"`
	Details   map[string]interface{} `json:"details,omitempty"`
	Error     string                 `json:"error,omitempty"`
}

// Check is a function that performs a health check
type Check func(ctx context.Context) CheckResult

// Checker manages health checks for the system
type Checker struct {
	checks map[string]Check
	mu     sync.RWMutex
}

// NewChecker creates a new health checker
func NewChecker() *Checker {
	return &Checker{
		checks: make(map[string]Check),
	}
}

// Register registers a health check
func (hc *Checker) Register(name string, check Check) {
	hc.mu.Lock()
	defer hc.mu.Unlock()
	hc.checks[name] = check
}

// CheckAll runs all registered health checks
func (hc *Checker) CheckAll(ctx context.Context) map[string]CheckResult {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	results := make(map[string]CheckResult)
	var wg sync.WaitGroup

	for name, check := range hc.checks {
		wg.Add(1)
		go func(n string, c Check) {
			defer wg.Done()
			
			// Create timeout context for individual check
			checkCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
			defer cancel()
			
			start := time.Now()
			result := c(checkCtx)
			result.Duration = time.Since(start)
			result.Timestamp = time.Now().UTC()
			
			results[n] = result
		}(name, check)
	}

	wg.Wait()
	return results
}

// OverallStatus determines the overall system health
func (hc *Checker) OverallStatus(ctx context.Context) Status {
	results := hc.CheckAll(ctx)
	
	hasUnhealthy := false
	hasDegraded := false
	
	for _, result := range results {
		switch result.Status {
		case StatusUnhealthy:
			hasUnhealthy = true
		case StatusDegraded:
			hasDegraded = true
		}
	}
	
	if hasUnhealthy {
		return StatusUnhealthy
	}
	if hasDegraded {
		return StatusDegraded
	}
	return StatusHealthy
}

// HTTPHandler returns an HTTP handler for health checks
func (hc *Checker) HTTPHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		
		// Check if this is a liveness or readiness probe
		probePath := r.URL.Path
		
		switch probePath {
		case "/health/live":
			// Liveness probe - just check if service is running
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]string{
				"status": "alive",
				"timestamp": time.Now().UTC().Format(time.RFC3339),
			})
			
		case "/health/ready":
			// Readiness probe - check all components
			results := hc.CheckAll(ctx)
			overallStatus := hc.OverallStatus(ctx)
			
			response := map[string]interface{}{
				"status":     string(overallStatus),
				"timestamp":  time.Now().UTC().Format(time.RFC3339),
				"components": results,
			}
			
			// Set appropriate HTTP status code
			statusCode := http.StatusOK
			if overallStatus == StatusUnhealthy {
				statusCode = http.StatusServiceUnavailable
			} else if overallStatus == StatusDegraded {
				statusCode = http.StatusOK // Still ready but degraded
			}
			
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(statusCode)
			json.NewEncoder(w).Encode(response)
			
		default:
			// Default health endpoint - basic check
			overallStatus := hc.OverallStatus(ctx)
			
			statusCode := http.StatusOK
			if overallStatus == StatusUnhealthy {
				statusCode = http.StatusServiceUnavailable
			}
			
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(statusCode)
			json.NewEncoder(w).Encode(map[string]string{
				"status":    string(overallStatus),
				"timestamp": time.Now().UTC().Format(time.RFC3339),
			})
		}
	}
}

// Common health checks

// DatabaseCheck creates a health check for database connectivity
func DatabaseCheck(ping func(context.Context) error) Check {
	return func(ctx context.Context) CheckResult {
		err := ping(ctx)
		if err != nil {
			return CheckResult{
				Status: StatusUnhealthy,
				Error:  err.Error(),
			}
		}
		return CheckResult{
			Status: StatusHealthy,
		}
	}
}

// NATSCheck creates a health check for NATS connectivity
func NATSCheck(ping func() error) Check {
	return func(ctx context.Context) CheckResult {
		err := ping()
		if err != nil {
			return CheckResult{
				Status: StatusUnhealthy,
				Error:  err.Error(),
			}
		}
		return CheckResult{
			Status: StatusHealthy,
		}
	}
}

// RedisCheck creates a health check for Redis connectivity
func RedisCheck(ping func(context.Context) error) Check {
	return func(ctx context.Context) CheckResult {
		err := ping(ctx)
		if err != nil {
			return CheckResult{
				Status: StatusUnhealthy,
				Error:  err.Error(),
			}
		}
		return CheckResult{
			Status: StatusHealthy,
		}
	}
}

// DiskSpaceCheck checks available disk space
func DiskSpaceCheck(path string, minBytes uint64) Check {
	return func(ctx context.Context) CheckResult {
		// Implementation would check actual disk space
		// For now, returning healthy
		return CheckResult{
			Status: StatusHealthy,
			Details: map[string]interface{}{
				"path":      path,
				"min_bytes": minBytes,
			},
		}
	}
}

// MemoryCheck checks available memory
func MemoryCheck(maxUsagePercent float64) Check {
	return func(ctx context.Context) CheckResult {
		// Implementation would check actual memory usage
		// For now, returning healthy
		return CheckResult{
			Status: StatusHealthy,
			Details: map[string]interface{}{
				"max_usage_percent": maxUsagePercent,
			},
		}
	}
}