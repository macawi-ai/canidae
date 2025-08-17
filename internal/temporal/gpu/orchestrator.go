package gpu

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Provider represents a GPU compute provider
type Provider interface {
	Name() string
	Available(ctx context.Context) (bool, error)
	RequestCompute(ctx context.Context, req ComputeRequest) (*ComputeSession, error)
	ReleaseCompute(ctx context.Context, sessionID string) error
	GetCostPerHour() float64
	GetLatency() time.Duration
}

// ComputeRequest specifies computational needs
type ComputeRequest struct {
	TaskID      string
	MemoryMB    int64
	TimeLimit   time.Duration
	Priority    int
	ModelType   string // "HRM", "ARChitects", etc.
}

// ComputeSession represents an active GPU session
type ComputeSession struct {
	ID          string
	Provider    string
	StartTime   time.Time
	EndTime     *time.Time
	CostPerHour float64
	MemoryMB    int64
	Status      SessionStatus
}

// SessionStatus represents the state of a compute session
type SessionStatus int

const (
	SessionPending SessionStatus = iota
	SessionActive
	SessionCompleted
	SessionFailed
)

// MetabolicBudget tracks computational resource costs
type MetabolicBudget struct {
	TotalDollars   float64
	SpentDollars   float64
	TimeWindow     time.Duration
	CaloriesPerGPU float64 // Metaphorical "energy" cost
}

// Orchestrator manages GPU resources across providers
type Orchestrator struct {
	mu        sync.RWMutex
	providers map[string]Provider
	sessions  map[string]*ComputeSession
	budget    MetabolicBudget
	
	// Metabolic awareness
	localComputePower float64 // TFLOPS
	
	// Metrics
	totalSessions   int
	totalCost       float64
	successfulTasks int
}

// NewOrchestrator creates a new GPU orchestrator
func NewOrchestrator(budget MetabolicBudget) *Orchestrator {
	return &Orchestrator{
		providers: make(map[string]Provider),
		sessions:  make(map[string]*ComputeSession),
		budget:    budget,
		localComputePower: 2.0, // Assume 2 TFLOPS local
	}
}

// RegisterProvider adds a GPU provider
func (o *Orchestrator) RegisterProvider(provider Provider) {
	o.mu.Lock()
	defer o.mu.Unlock()
	
	o.providers[provider.Name()] = provider
	fmt.Printf("Registered GPU provider: %s ($%.2f/hr, %v latency)\n",
		provider.Name(), provider.GetCostPerHour(), provider.GetLatency())
}

// ShouldUseGPU implements the "dehydrated mouse" logic
func (o *Orchestrator) ShouldUseGPU(task Task) bool {
	localCost := o.EstimateLocalCompute(task)
	cloudCost := o.EstimateCloudCost(task)
	
	// If local is good enough, don't chase distant GPUs
	if localCost.Time < 1*time.Second && localCost.Quality > 0.8 {
		fmt.Printf("Using local compute (dehydrated mouse): time=%v, quality=%.2f\n",
			localCost.Time, localCost.Quality)
		return false
	}
	
	// Check budget constraints
	if o.budget.SpentDollars + cloudCost.Dollars > o.budget.TotalDollars {
		fmt.Printf("Budget constraint: would exceed budget (spent=%.2f, cost=%.2f, total=%.2f)\n",
			o.budget.SpentDollars, cloudCost.Dollars, o.budget.TotalDollars)
		return false
	}
	
	// Only use cloud if significantly better ROI
	roi := cloudCost.ValueRatio() / localCost.ValueRatio()
	if roi > 3.0 {
		fmt.Printf("Using cloud GPU: ROI=%.2f (cloud value=%.2f, local value=%.2f)\n",
			roi, cloudCost.ValueRatio(), localCost.ValueRatio())
		return true
	}
	
	return false
}

// RequestCompute finds best provider and allocates GPU
func (o *Orchestrator) RequestCompute(ctx context.Context, req ComputeRequest) (*ComputeSession, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	
	// Find best available provider
	provider, err := o.selectProvider(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("no suitable provider: %w", err)
	}
	
	// Request compute
	session, err := provider.RequestCompute(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("provider %s failed: %w", provider.Name(), err)
	}
	
	// Track session
	o.sessions[session.ID] = session
	o.totalSessions++
	
	// Update budget
	estimatedCost := provider.GetCostPerHour() * req.TimeLimit.Hours()
	o.budget.SpentDollars += estimatedCost
	o.totalCost += estimatedCost
	
	fmt.Printf("GPU session started: %s on %s ($%.4f estimated)\n",
		session.ID, provider.Name(), estimatedCost)
	
	return session, nil
}

// selectProvider chooses the best provider for a request
func (o *Orchestrator) selectProvider(ctx context.Context, req ComputeRequest) (Provider, error) {
	type candidate struct {
		provider Provider
		score    float64
	}
	
	candidates := make([]candidate, 0)
	
	for _, provider := range o.providers {
		available, err := provider.Available(ctx)
		if err != nil || !available {
			continue
		}
		
		// Score based on cost and latency
		costScore := 1.0 / provider.GetCostPerHour()
		latencyScore := 1.0 / provider.GetLatency().Seconds()
		totalScore := costScore * 0.7 + latencyScore * 0.3 // Weight cost more
		
		candidates = append(candidates, candidate{
			provider: provider,
			score:    totalScore,
		})
	}
	
	if len(candidates) == 0 {
		return nil, fmt.Errorf("no available providers")
	}
	
	// Select highest scoring provider
	best := candidates[0]
	for _, c := range candidates[1:] {
		if c.score > best.score {
			best = c
		}
	}
	
	return best.provider, nil
}

// ReleaseCompute releases a GPU session
func (o *Orchestrator) ReleaseCompute(ctx context.Context, sessionID string) error {
	o.mu.Lock()
	defer o.mu.Unlock()
	
	session, exists := o.sessions[sessionID]
	if !exists {
		return fmt.Errorf("session %s not found", sessionID)
	}
	
	// Find provider
	provider, exists := o.providers[session.Provider]
	if !exists {
		return fmt.Errorf("provider %s not found", session.Provider)
	}
	
	// Release compute
	if err := provider.ReleaseCompute(ctx, sessionID); err != nil {
		return fmt.Errorf("failed to release: %w", err)
	}
	
	// Update session
	now := time.Now()
	session.EndTime = &now
	session.Status = SessionCompleted
	
	// Calculate actual cost
	duration := now.Sub(session.StartTime)
	actualCost := session.CostPerHour * duration.Hours()
	
	fmt.Printf("GPU session ended: %s (duration=%v, cost=$%.4f)\n",
		sessionID, duration, actualCost)
	
	return nil
}

// Task represents a computational task
type Task struct {
	ID          string
	Type        string // "reasoning", "search", "training"
	Complexity  float64
	TimeLimit   time.Duration
	QualityNeed float64 // 0.0 to 1.0
}

// ComputeCost represents the cost of computation
type ComputeCost struct {
	Time    time.Duration
	Dollars float64
	Quality float64 // Expected quality of result
}

// ValueRatio calculates value per dollar
func (c ComputeCost) ValueRatio() float64 {
	if c.Dollars == 0 {
		return c.Quality * 1000 // Free is very valuable
	}
	return c.Quality / c.Dollars
}

// EstimateLocalCompute estimates local computation cost
func (o *Orchestrator) EstimateLocalCompute(task Task) ComputeCost {
	// Estimate based on local compute power
	timeNeeded := time.Duration(task.Complexity / o.localComputePower * float64(time.Second))
	
	// Local compute is "free" in dollars but has opportunity cost
	return ComputeCost{
		Time:    timeNeeded,
		Dollars: 0.0,
		Quality: min(0.8, 1.0/float64(timeNeeded.Seconds())), // Quality degrades with time
	}
}

// EstimateCloudCost estimates cloud GPU cost
func (o *Orchestrator) EstimateCloudCost(task Task) ComputeCost {
	// Find cheapest available provider
	cheapest := 999999.0
	for _, provider := range o.providers {
		if provider.GetCostPerHour() < cheapest {
			cheapest = provider.GetCostPerHour()
		}
	}
	
	// Assume GPU is 50x faster than local
	timeNeeded := time.Duration(task.Complexity / (o.localComputePower * 50) * float64(time.Second))
	cost := cheapest * timeNeeded.Hours()
	
	return ComputeCost{
		Time:    timeNeeded,
		Dollars: cost,
		Quality: 0.95, // GPUs generally give better quality
	}
}

// Metrics returns orchestrator statistics
func (o *Orchestrator) Metrics() map[string]interface{} {
	o.mu.RLock()
	defer o.mu.RUnlock()
	
	activeSessions := 0
	for _, session := range o.sessions {
		if session.Status == SessionActive {
			activeSessions++
		}
	}
	
	return map[string]interface{}{
		"providers":        len(o.providers),
		"total_sessions":   o.totalSessions,
		"active_sessions":  activeSessions,
		"total_cost":       o.totalCost,
		"budget_remaining": o.budget.TotalDollars - o.budget.SpentDollars,
		"success_rate":     float64(o.successfulTasks) / float64(o.totalSessions),
	}
}

// Helper function
func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}