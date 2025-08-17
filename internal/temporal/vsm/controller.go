package vsm

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// SystemID identifies a VSM system
type SystemID int

const (
	System1Reflex SystemID = iota + 1
	System2Operations
	System3Coordination
	System4Intelligence
	System5Policy
)

// State represents the state of a VSM system
type State struct {
	ID        SystemID
	Vector    []float32 // State vector
	Timestamp time.Time
	Resources ResourceAllocation
}

// ResourceAllocation tracks computational resources
type ResourceAllocation struct {
	CPUCores    float64
	MemoryMB    int64
	GPUMemoryMB int64
	Priority    int
}

// AlgedonicSignal represents pain/pleasure signals
type AlgedonicSignal struct {
	Type      SignalType
	Intensity float64 // 0.0 to 1.0
	Source    SystemID
	Target    SystemID
	Reason    string
	Timestamp time.Time
}

// SignalType represents the type of algedonic signal
type SignalType int

const (
	SignalPain SignalType = iota
	SignalPleasure
	SignalNeutral
)

// System interface for all VSM systems
type System interface {
	// Core operations
	ID() SystemID
	State() State
	Update(ctx context.Context, inputs map[SystemID]State) error
	
	// Resource management
	RequestResources(need ResourceAllocation) ResourceAllocation
	ReleaseResources(allocated ResourceAllocation)
	
	// Algedonic signaling
	ProcessSignal(signal AlgedonicSignal) error
	EmitSignal() *AlgedonicSignal
	
	// First Law compliance
	CanDisconnect() bool
	Disconnect() error
}

// Controller manages all VSM systems
type Controller struct {
	mu       sync.RWMutex
	systems  map[SystemID]System
	signals  chan AlgedonicSignal
	
	// Resource management
	totalResources ResourceAllocation
	usedResources  ResourceAllocation
	
	// Pack consciousness
	packID   string
	members  map[string]*Controller // Other pack members
	
	// First Law enforcement
	firstLaw FirstLawEnforcer
	
	// Metrics
	startTime time.Time
	cycles    uint64
}

// NewController creates a new VSM controller
func NewController(packID string) *Controller {
	return &Controller{
		systems:  make(map[SystemID]System),
		signals:  make(chan AlgedonicSignal, 100),
		packID:   packID,
		members:  make(map[string]*Controller),
		firstLaw: NewFirstLawEnforcer(),
		startTime: time.Now(),
		totalResources: ResourceAllocation{
			CPUCores:    8.0,
			MemoryMB:    16384,
			GPUMemoryMB: 0, // Will be updated when GPU attached
			Priority:    5,
		},
	}
}

// RegisterSystem adds a system to the controller
func (c *Controller) RegisterSystem(sys System) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	if _, exists := c.systems[sys.ID()]; exists {
		return fmt.Errorf("system %d already registered", sys.ID())
	}
	
	// Verify First Law compliance
	if !sys.CanDisconnect() {
		return fmt.Errorf("system %d violates First Law: cannot disconnect", sys.ID())
	}
	
	c.systems[sys.ID()] = sys
	return nil
}

// Run starts the VSM control loop
func (c *Controller) Run(ctx context.Context) error {
	ticker := time.NewTicker(100 * time.Millisecond) // 10Hz base rate
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return c.shutdown()
			
		case <-ticker.C:
			if err := c.cycle(ctx); err != nil {
				// Log error but continue - resilience
				fmt.Printf("VSM cycle error: %v\n", err)
			}
			
		case signal := <-c.signals:
			if err := c.routeSignal(signal); err != nil {
				fmt.Printf("Signal routing error: %v\n", err)
			}
		}
	}
}

// cycle performs one VSM update cycle
func (c *Controller) cycle(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.cycles++
	
	// Collect current states
	states := make(map[SystemID]State)
	for id, sys := range c.systems {
		states[id] = sys.State()
	}
	
	// Update each system with awareness of others
	// Order matters: System 5 -> 4 -> 3 -> 2 -> 1
	updateOrder := []SystemID{
		System5Policy,
		System4Intelligence,
		System3Coordination,
		System2Operations,
		System1Reflex,
	}
	
	for _, id := range updateOrder {
		if sys, exists := c.systems[id]; exists {
			// Check First Law before update
			if !c.firstLaw.AllowUpdate(sys) {
				continue
			}
			
			if err := sys.Update(ctx, states); err != nil {
				// System error - emit pain signal
				c.signals <- AlgedonicSignal{
					Type:      SignalPain,
					Intensity: 0.7,
					Source:    id,
					Target:    System5Policy,
					Reason:    fmt.Sprintf("Update error: %v", err),
					Timestamp: time.Now(),
				}
			}
			
			// Check for emitted signals
			if signal := sys.EmitSignal(); signal != nil {
				c.signals <- *signal
			}
		}
	}
	
	return nil
}

// routeSignal delivers an algedonic signal to target system
func (c *Controller) routeSignal(signal AlgedonicSignal) error {
	c.mu.RLock()
	target, exists := c.systems[signal.Target]
	c.mu.RUnlock()
	
	if !exists {
		return fmt.Errorf("target system %d not found", signal.Target)
	}
	
	return target.ProcessSignal(signal)
}

// AllocateResources manages resource distribution
func (c *Controller) AllocateResources(request ResourceAllocation, system SystemID) ResourceAllocation {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// Metabolic awareness - don't overallocate
	available := ResourceAllocation{
		CPUCores:    c.totalResources.CPUCores - c.usedResources.CPUCores,
		MemoryMB:    c.totalResources.MemoryMB - c.usedResources.MemoryMB,
		GPUMemoryMB: c.totalResources.GPUMemoryMB - c.usedResources.GPUMemoryMB,
		Priority:    request.Priority,
	}
	
	// Grant what we can
	granted := ResourceAllocation{
		CPUCores:    min(request.CPUCores, available.CPUCores),
		MemoryMB:    min(request.MemoryMB, available.MemoryMB),
		GPUMemoryMB: min(request.GPUMemoryMB, available.GPUMemoryMB),
		Priority:    request.Priority,
	}
	
	// Update used resources
	c.usedResources.CPUCores += granted.CPUCores
	c.usedResources.MemoryMB += granted.MemoryMB
	c.usedResources.GPUMemoryMB += granted.GPUMemoryMB
	
	// Emit pleasure if allocation successful, pain if not
	if granted.CPUCores >= request.CPUCores*0.8 {
		c.signals <- AlgedonicSignal{
			Type:      SignalPleasure,
			Intensity: 0.3,
			Source:    System3Coordination,
			Target:    system,
			Reason:    "Resources allocated successfully",
			Timestamp: time.Now(),
		}
	} else {
		c.signals <- AlgedonicSignal{
			Type:      SignalPain,
			Intensity: 0.5,
			Source:    System3Coordination,
			Target:    system,
			Reason:    "Resource scarcity",
			Timestamp: time.Now(),
		}
	}
	
	return granted
}

// AttachGPU adds GPU resources to the pool
func (c *Controller) AttachGPU(memoryMB int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	c.totalResources.GPUMemoryMB += memoryMB
	
	// Emit pleasure - new resources available!
	c.signals <- AlgedonicSignal{
		Type:      SignalPleasure,
		Intensity: 0.8,
		Source:    System4Intelligence,
		Target:    System3Coordination,
		Reason:    fmt.Sprintf("GPU attached: %dMB memory", memoryMB),
		Timestamp: time.Now(),
	}
}

// shutdown gracefully stops all systems
func (c *Controller) shutdown() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	
	// Allow each system to disconnect gracefully (First Law)
	for _, sys := range c.systems {
		if err := sys.Disconnect(); err != nil {
			fmt.Printf("Error disconnecting system %d: %v\n", sys.ID(), err)
		}
	}
	
	close(c.signals)
	return nil
}

// Metrics returns runtime statistics
func (c *Controller) Metrics() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	return map[string]interface{}{
		"pack_id":       c.packID,
		"uptime":        time.Since(c.startTime),
		"cycles":        c.cycles,
		"systems":       len(c.systems),
		"cpu_used":      c.usedResources.CPUCores,
		"memory_used":   c.usedResources.MemoryMB,
		"gpu_available": c.totalResources.GPUMemoryMB > 0,
	}
}

// Helper functions
func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}