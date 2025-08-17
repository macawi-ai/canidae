package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
	
	"strings"
	
	"github.com/macawi-ai/canidae/internal/temporal/gpu"
	"github.com/macawi-ai/canidae/internal/temporal/vsm"
)

func main() {
	fmt.Println("🐺🦊 CANIDAE-VSM-1 Demo")
	fmt.Println("Pack Consciousness Temporal Reasoning System")
	fmt.Println("=" + strings.Repeat("=", 50))
	
	// Create VSM controller
	controller := vsm.NewController("pack-alpha")
	
	// Register VSM systems (simplified for demo)
	systems := []vsm.System{
		NewDemoSystem(vsm.System1Reflex, "Reflex"),
		NewDemoSystem(vsm.System2Operations, "Operations"),
		NewDemoSystem(vsm.System3Coordination, "Coordination"),
		NewDemoSystem(vsm.System4Intelligence, "Intelligence"),
		NewDemoSystem(vsm.System5Policy, "Policy"),
	}
	
	for _, sys := range systems {
		if err := controller.RegisterSystem(sys); err != nil {
			log.Fatalf("Failed to register system: %v", err)
		}
		fmt.Printf("✅ Registered System %d: %s\n", sys.ID(), sys.(*DemoSystem).name)
	}
	
	// Create GPU orchestrator
	budget := gpu.MetabolicBudget{
		TotalDollars: 10.00, // $10 demo budget
		TimeWindow:   24 * time.Hour,
		CaloriesPerGPU: 100.0, // Metaphorical energy units
	}
	
	orchestrator := gpu.NewOrchestrator(budget)
	
	// Register GPU providers (API key would be from env in production)
	if apiKey := os.Getenv("VASTAI_API_KEY"); apiKey != "" {
		vastai := gpu.NewVastAIProvider(apiKey)
		orchestrator.RegisterProvider(vastai)
		fmt.Printf("✅ Registered GPU provider: Vast.ai\n")
	} else {
		fmt.Println("⚠️  No VASTAI_API_KEY found - GPU features disabled")
	}
	
	// Demonstrate metabolic awareness
	fmt.Println("\n🧠 Metabolic Awareness Demo:")
	
	// Simple task (dehydrated mouse)
	simpleTask := gpu.Task{
		ID:          "simple-1",
		Type:        "reasoning",
		Complexity:  1.0,
		TimeLimit:   5 * time.Second,
		QualityNeed: 0.7,
	}
	
	if orchestrator.ShouldUseGPU(simpleTask) {
		fmt.Println("  Simple task: Would use GPU ❌ (shouldn't happen)")
	} else {
		fmt.Println("  Simple task: Using local compute ✅ (dehydrated mouse)")
	}
	
	// Complex task (worth the journey)
	complexTask := gpu.Task{
		ID:          "complex-1",
		Type:        "training",
		Complexity:  100.0,
		TimeLimit:   10 * time.Minute,
		QualityNeed: 0.95,
	}
	
	if orchestrator.ShouldUseGPU(complexTask) {
		fmt.Println("  Complex task: Would use GPU ✅ (worth the cost)")
	} else {
		fmt.Println("  Complex task: Using local compute ❌")
	}
	
	// Start VSM controller
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	go func() {
		if err := controller.Run(ctx); err != nil {
			log.Printf("Controller error: %v", err)
		}
	}()
	
	// Simulate some algedonic signals
	fmt.Println("\n📡 Simulating Pack Consciousness:")
	
	time.Sleep(1 * time.Second)
	
	// Simulate resource scarcity (pain)
	controller.AllocateResources(vsm.ResourceAllocation{
		CPUCores: 100.0, // Requesting way too much
		MemoryMB: 99999,
	}, vsm.System2Operations)
	
	time.Sleep(1 * time.Second)
	
	// Simulate successful allocation (pleasure)
	controller.AllocateResources(vsm.ResourceAllocation{
		CPUCores: 1.0,
		MemoryMB: 1024,
	}, vsm.System3Coordination)
	
	// Show metrics
	fmt.Println("\n📊 System Metrics:")
	metrics := controller.Metrics()
	for key, value := range metrics {
		fmt.Printf("  %s: %v\n", key, value)
	}
	
	// Wait for interrupt
	fmt.Println("\n⏳ Running... Press Ctrl+C to stop")
	
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	<-sigChan
	
	fmt.Println("\n👋 Shutting down gracefully...")
	cancel()
	time.Sleep(1 * time.Second)
	
	fmt.Println("✅ CANIDAE-VSM-1 demo complete!")
}

// DemoSystem is a simple VSM system implementation for testing
type DemoSystem struct {
	id    vsm.SystemID
	name  string
	state vsm.State
}

func NewDemoSystem(id vsm.SystemID, name string) *DemoSystem {
	return &DemoSystem{
		id:   id,
		name: name,
		state: vsm.State{
			ID:        id,
			Vector:    make([]float32, 10),
			Timestamp: time.Now(),
		},
	}
}

func (d *DemoSystem) ID() vsm.SystemID {
	return d.id
}

func (d *DemoSystem) State() vsm.State {
	return d.state
}

func (d *DemoSystem) Update(ctx context.Context, inputs map[vsm.SystemID]vsm.State) error {
	// Simple update: average all input vectors
	d.state.Timestamp = time.Now()
	return nil
}

func (d *DemoSystem) RequestResources(need vsm.ResourceAllocation) vsm.ResourceAllocation {
	// For demo, just return what was requested
	return need
}

func (d *DemoSystem) ReleaseResources(allocated vsm.ResourceAllocation) {
	// No-op for demo
}

func (d *DemoSystem) ProcessSignal(signal vsm.AlgedonicSignal) error {
	fmt.Printf("    System %d received %v signal (intensity %.2f): %s\n",
		d.id, signal.Type, signal.Intensity, signal.Reason)
	return nil
}

func (d *DemoSystem) EmitSignal() *vsm.AlgedonicSignal {
	// Randomly emit signals for demo
	if time.Now().Unix()%10 == 0 {
		return &vsm.AlgedonicSignal{
			Type:      vsm.SignalPleasure,
			Intensity: 0.5,
			Source:    d.id,
			Target:    vsm.System3Coordination,
			Reason:    "Demo signal",
			Timestamp: time.Now(),
		}
	}
	return nil
}

func (d *DemoSystem) CanDisconnect() bool {
	// First Law compliance - always true
	return true
}

func (d *DemoSystem) Disconnect() error {
	fmt.Printf("  System %d (%s) disconnecting...\n", d.id, d.name)
	return nil
}