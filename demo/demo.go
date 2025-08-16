package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/canidae/canidae/internal/chaos"
	"github.com/canidae/canidae/internal/providers"
	"github.com/canidae/canidae/internal/ring"
	"github.com/nats-io/nats.go"
)

func main() {
	var (
		natsURL     = flag.String("nats", nats.DefaultURL, "NATS server URL")
		enableChaos = flag.Bool("chaos", false, "Enable chaos monkey")
		provider    = flag.String("provider", "mock", "Provider to use (mock, anthropic, openai)")
	)
	flag.Parse()

	log.Println("🐺 CANIDAE Demo - Pack Coordination in Action")
	log.Println("==============================================")

	// Connect to NATS
	log.Printf("Connecting to NATS at %s...", *natsURL)
	nc, err := nats.Connect(*natsURL,
		nats.Name("canidae-demo"),
		nats.MaxReconnects(5),
	)
	if err != nil {
		log.Fatalf("Failed to connect to NATS: %v", err)
	}
	defer nc.Close()

	// Create provider registry
	registry := providers.NewRegistry()
	
	// Register mock provider for demo
	registry.Register("mock", NewMockProvider)
	
	// Create mock provider
	mockProvider, err := registry.Create("mock", map[string]interface{}{
		"delay": 100 * time.Millisecond,
	})
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}

	// Create flow controller
	flowController := ring.NewFlowController()
	
	// Create chaos monkey (optional)
	var chaosMonkey *chaos.ChaosMonkey
	if *enableChaos {
		chaosMonkey = chaos.NewChaosMonkey(chaos.DefaultConfig())
		chaosMonkey.Enable()
		log.Println("🙈 Chaos Monkey ENABLED - Expect disruptions!")
	}

	// Demo: Flow Control
	log.Println("\n📊 Testing Flow Control...")
	testFlowControl(flowController, mockProvider)

	// Demo: Priority Lanes
	log.Println("\n🚦 Testing Priority Lanes...")
	testPriorityLanes(nc)

	// Demo: Circuit Breaker
	log.Println("\n⚡ Testing Circuit Breaker...")
	testCircuitBreaker(flowController)

	// Demo: Chaos Engineering (if enabled)
	if chaosMonkey != nil {
		log.Println("\n🙈 Testing Chaos Engineering...")
		testChaos(chaosMonkey)
	}

	// Demo: Pack Coordination
	log.Println("\n🐺 Testing Pack Coordination...")
	testPackCoordination(nc, registry)

	log.Println("\n✅ Demo Complete - The pack hunts as one!")
}

// MockProvider for testing
type MockProvider struct {
	providers.BaseProvider
	delay   time.Duration
	failRate float64
}

func NewMockProvider(config map[string]interface{}) (providers.Provider, error) {
	delay := 100 * time.Millisecond
	if d, ok := config["delay"].(time.Duration); ok {
		delay = d
	}
	
	return &MockProvider{
		BaseProvider: providers.BaseProvider{
			Name: "mock",
			Capabilities: providers.Capabilities{
				Models: []providers.ModelInfo{
					{ID: "mock-model", Name: "Mock Model", ContextSize: 4096},
				},
				MaxTokens:      1024,
				SupportsStream: false,
			},
		},
		delay: delay,
	}, nil
}

func (m *MockProvider) Execute(ctx context.Context, req providers.Request) (*providers.Response, error) {
	// Simulate processing delay
	select {
	case <-time.After(m.delay):
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	
	return &providers.Response{
		ID:        fmt.Sprintf("mock-%d", time.Now().UnixNano()),
		RequestID: req.ID,
		Provider:  "mock",
		Model:     req.Model,
		Content:   fmt.Sprintf("Mock response to: %s", req.Messages[0].Content),
		Usage: providers.Usage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
			Cost:             0.0001,
			Currency:         "USD",
		},
		Timestamp: time.Now(),
	}, nil
}

func (m *MockProvider) HealthCheck(ctx context.Context) error {
	return nil
}

func (m *MockProvider) EstimateCost(req providers.Request) providers.Cost {
	return providers.Cost{
		Estimated: 0.0001,
		Minimum:   0.00005,
		Maximum:   0.0002,
		Currency:  "USD",
	}
}

func testFlowControl(fc *ring.FlowController, provider providers.Provider) {
	// Test rate limiting
	accepted := 0
	rejected := 0
	
	for i := 0; i < 150; i++ {
		if ok, reason := fc.ShouldAcceptRequest("mock", ring.PriorityMedium, 1); ok {
			accepted++
		} else {
			rejected++
			if i < 5 { // Only log first few rejections
				log.Printf("  Request %d rejected: %s", i, reason)
			}
		}
	}
	
	log.Printf("  Flow Control: %d accepted, %d rejected", accepted, rejected)
	
	metrics := fc.GetMetrics()
	log.Printf("  Metrics: %+v", metrics)
}

func testPriorityLanes(nc *nats.Conn) {
	subjects := []string{
		ring.GetPrioritySubject("canidae.request", ring.PriorityCritical),
		ring.GetPrioritySubject("canidae.request", ring.PriorityHigh),
		ring.GetPrioritySubject("canidae.request", ring.PriorityMedium),
		ring.GetPrioritySubject("canidae.request", ring.PriorityLow),
	}
	
	for _, subject := range subjects {
		log.Printf("  Created priority lane: %s", subject)
	}
}

func testCircuitBreaker(fc *ring.FlowController) {
	breaker := fc.GetCircuitBreaker("test-provider")
	
	// Simulate failures to open circuit
	failures := 0
	for i := 0; i < 10; i++ {
		err := breaker.Call(context.Background(), func() error {
			if i < 6 { // First 6 calls fail
				return fmt.Errorf("simulated failure")
			}
			return nil
		})
		
		if err != nil {
			failures++
			if i < 3 {
				log.Printf("  Call %d failed: %v", i+1, err)
			}
		}
	}
	
	log.Printf("  Circuit Breaker: %d failures triggered protection", failures)
}

func testChaos(cm *chaos.ChaosMonkey) {
	// Test message dropping
	dropped := 0
	for i := 0; i < 100; i++ {
		if cm.ShouldDropMessage() {
			dropped++
		}
	}
	log.Printf("  Chaos: Dropped %d/100 messages", dropped)
	
	// Test provider slowdown
	if delay := cm.ShouldSlowProvider("test-provider"); delay > 0 {
		log.Printf("  Chaos: Provider slowed by %v", delay)
	}
	
	// Get chaos metrics
	metrics := cm.GetMetrics()
	log.Printf("  Chaos Metrics: %+v", metrics)
}

func testPackCoordination(nc *nats.Conn, registry *providers.Registry) {
	// Subscribe to responses
	responses := make(chan *providers.Response, 10)
	sub, err := nc.Subscribe("canidae.response.demo-pack", func(msg *nats.Msg) {
		var resp providers.Response
		if err := json.Unmarshal(msg.Data, &resp); err == nil {
			responses <- &resp
		}
	})
	if err != nil {
		log.Printf("  Failed to subscribe: %v", err)
		return
	}
	defer sub.Unsubscribe()
	
	// Send test requests as different pack members
	roles := []providers.PackRole{
		providers.PackRoleAlpha,
		providers.PackRoleHunter,
		providers.PackRoleScout,
	}
	
	for i, role := range roles {
		req := providers.Request{
			ID:     fmt.Sprintf("req-%d", i),
			PackID: "demo-pack",
			Role:   role,
			Model:  "mock-model",
			Messages: []providers.Message{
				{Role: "user", Content: fmt.Sprintf("Request from %s", role)},
			},
			Priority:  providers.Priority(i),
			Timestamp: time.Now(),
		}
		
		data, _ := json.Marshal(req)
		nc.Publish("canidae.request.demo-pack", data)
		log.Printf("  %s sent request %s", role, req.ID)
	}
	
	// Wait for responses
	timeout := time.After(2 * time.Second)
	received := 0
	
	for {
		select {
		case resp := <-responses:
			log.Printf("  Received response %s for request %s", resp.ID, resp.RequestID)
			received++
			if received >= len(roles) {
				return
			}
		case <-timeout:
			log.Printf("  Pack coordination: %d/%d responses received", received, len(roles))
			return
		}
	}
}