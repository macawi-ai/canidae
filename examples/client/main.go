package main

import (
	"context"
	"fmt"
	"log"
	"time"
	
	"github.com/macawi-ai/canidae/pkg/client/canidae"
)

func main() {
	// Create client with configuration
	client, err := canidae.NewClient(
		canidae.WithServerEndpoint("192.168.1.38:14001"),
		canidae.WithPackID("demo-pack"),
		canidae.WithSecurityProfile(canidae.SecurityProfileEnterprise),
		canidae.WithAPIKey("demo-api-key"), // Using API key for demo
		canidae.WithTimeout(30*time.Second),
	)
	if err != nil {
		log.Fatal("Failed to create client:", err)
	}
	
	ctx := context.Background()
	
	// Connect to CANIDAE server
	fmt.Println("🐺 Connecting to CANIDAE server...")
	if err := client.Connect(ctx); err != nil {
		log.Fatal("Failed to connect:", err)
	}
	defer client.Disconnect(ctx)
	
	fmt.Println("✅ Connected to CANIDAE server")
	
	// Example 1: Execute a single agent
	fmt.Println("\n📍 Example 1: Execute single agent")
	execResp, err := client.ExecuteAgent(ctx, &canidae.ExecuteRequest{
		Agent:           canidae.AgentTypeAnthropic,
		Prompt:          "What is the CANIDAE project about?",
		Model:           "claude-3-opus",
		Temperature:     0.7,
		MaxTokens:       500,
		SecurityProfile: canidae.SecurityProfileEnterprise,
		Metadata: map[string]string{
			"example": "single-agent",
			"user":    "demo",
		},
	})
	if err != nil {
		log.Printf("Failed to execute agent: %v", err)
	} else {
		fmt.Printf("Response: %s\n", execResp.Response)
		fmt.Printf("Tokens used: %d\n", execResp.TokensUsed)
		fmt.Printf("Duration: %v\n", execResp.Duration)
	}
	
	// Example 2: Chain multiple agents
	fmt.Println("\n📍 Example 2: Chain multiple agents")
	chainResp, err := client.ChainAgents(ctx, &canidae.ChainRequest{
		Steps: []canidae.ChainStep{
			{
				Agent:  canidae.AgentTypeOpenAI,
				Prompt: "Generate a creative story idea about wolves",
				Model:  "gpt-4",
			},
			{
				Agent:     canidae.AgentTypeAnthropic,
				Prompt:    "Expand the previous story idea into a detailed outline",
				Model:     "claude-3-opus",
				DependsOn: []string{"openai"},
			},
			{
				Agent:     canidae.AgentTypeGemini,
				Prompt:    "Create character profiles for the story",
				Model:     "gemini-pro",
				DependsOn: []string{"anthropic"},
			},
		},
		SecurityProfile: canidae.SecurityProfileEnterprise,
		ContinueOnError: true,
	})
	if err != nil {
		log.Printf("Failed to chain agents: %v", err)
	} else {
		fmt.Printf("Chain completed with %d steps\n", len(chainResp.Steps))
		fmt.Printf("Total tokens: %d\n", chainResp.TotalTokens)
		fmt.Printf("Total duration: %v\n", chainResp.Duration)
		
		for i, step := range chainResp.Steps {
			fmt.Printf("\nStep %d (%s):\n", i+1, step.Agent)
			if step.Error != "" {
				fmt.Printf("  Error: %s\n", step.Error)
			} else {
				fmt.Printf("  Response: %.100s...\n", step.Response)
				fmt.Printf("  Tokens: %d\n", step.TokensUsed)
			}
		}
	}
	
	// Example 3: Summon a pack formation
	fmt.Println("\n📍 Example 3: Summon pack formation")
	packResp, err := client.SummonPack(ctx, &canidae.PackRequest{
		Formation: canidae.PackFormation{
			Alpha: &canidae.PackMember{
				Role:      "coordinator",
				Agent:     canidae.AgentTypeAnthropic,
				Objective: "Coordinate the analysis of a complex problem",
				Model:     "claude-3-opus",
			},
			Hunters: []canidae.PackMember{
				{
					Role:      "researcher",
					Agent:     canidae.AgentTypeOpenAI,
					Objective: "Research technical details",
					Model:     "gpt-4",
				},
				{
					Role:      "analyzer",
					Agent:     canidae.AgentTypeGemini,
					Objective: "Analyze patterns and insights",
					Model:     "gemini-pro",
				},
			},
			Scouts: []canidae.PackMember{
				{
					Role:      "explorer",
					Agent:     canidae.AgentTypeOllama,
					Objective: "Explore alternative solutions",
					Model:     "llama2",
				},
			},
		},
		Objective:       "Analyze the best architecture for a distributed AI system",
		SecurityProfile: canidae.SecurityProfileEnterprise,
		MaxConcurrency:  3,
		Timeout:         60 * time.Second,
	})
	if err != nil {
		log.Printf("Failed to summon pack: %v", err)
	} else {
		fmt.Printf("Pack %s completed\n", packResp.PackID)
		fmt.Printf("Total tokens: %d\n", packResp.TotalTokens)
		fmt.Printf("Duration: %v\n", packResp.Duration)
		
		for _, result := range packResp.Results {
			fmt.Printf("\n%s (%s):\n", result.Role, result.Agent)
			if result.Error != "" {
				fmt.Printf("  Error: %s\n", result.Error)
			} else {
				fmt.Printf("  Response: %.100s...\n", result.Response)
				fmt.Printf("  Tokens: %d\n", result.TokensUsed)
			}
		}
	}
	
	// Example 4: Stream real-time responses
	fmt.Println("\n📍 Example 4: Stream real-time responses")
	streamErr := client.Stream(ctx, func(event canidae.StreamEvent) error {
		switch event.Type {
		case canidae.StreamEventTypeConnect:
			fmt.Println("🔌 Stream connected")
		case canidae.StreamEventTypeData:
			fmt.Printf("📦 Data: %v\n", event.Data)
		case canidae.StreamEventTypeProgress:
			fmt.Printf("⏳ Progress: %v\n", event.Data)
		case canidae.StreamEventTypeError:
			fmt.Printf("❌ Error: %v\n", event.Error)
		case canidae.StreamEventTypeComplete:
			fmt.Println("✅ Stream complete")
			return nil // Stop streaming
		}
		return nil
	})
	if streamErr != nil {
		log.Printf("Stream error: %v", streamErr)
	}
	
	// Get client status
	status := client.GetStatus()
	fmt.Printf("\n📊 Client Status:\n")
	fmt.Printf("  Connected: %v\n", status.Connected)
	fmt.Printf("  Authenticated: %v\n", status.Authenticated)
	fmt.Printf("  Pack ID: %s\n", status.PackID)
	fmt.Printf("  Server: %s\n", status.ServerEndpoint)
	fmt.Printf("  Last Activity: %v\n", status.LastActivity)
	
	fmt.Println("\n🐺 CANIDAE SDK demo complete!")
}