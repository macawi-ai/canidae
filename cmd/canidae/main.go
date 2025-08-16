package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
	
	"github.com/canidae/canidae/internal/ring"
	"github.com/canidae/canidae/pkg/types"
)

// Server wraps the Ring orchestrator with HTTP endpoints
type Server struct {
	ring   *ring.Ring
	config ServerConfig
}

// ServerConfig holds server configuration
type ServerConfig struct {
	Port        string
	NatsURL     string
	MetricsPort string
}

// NewServer creates a new HTTP server wrapping the Ring
func NewServer(config ServerConfig) (*Server, error) {
	// Create Ring orchestrator
	ringConfig := ring.Config{
		NatsURL:    config.NatsURL,
		Name:       "canidae-ring",
		MaxRetries: 3,
		RetryDelay: time.Second,
	}
	
	r, err := ring.NewRing(ringConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create ring: %w", err)
	}
	
	return &Server{
		ring:   r,
		config: config,
	}, nil
}

// Start begins serving HTTP requests
func (s *Server) Start(ctx context.Context) error {
	mux := http.NewServeMux()
	
	// Health endpoints
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/ready", s.handleReady)
	mux.HandleFunc("/metrics", s.handleMetrics)
	
	// API endpoints
	mux.HandleFunc("/providers", s.handleProviders)
	mux.HandleFunc("/process", s.handleProcess)
	
	// Root endpoint
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "CANIDAE Ring Orchestrator v0.1.0\nThe pack hunts as one 🐺\n")
	})
	
	// Start Ring message processing
	go func() {
		if err := s.ring.Start(ctx); err != nil {
			log.Printf("Ring processing error: %v", err)
		}
	}()
	
	// Start HTTP server
	server := &http.Server{
		Addr:    ":" + s.config.Port,
		Handler: mux,
	}
	
	// Graceful shutdown
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		server.Shutdown(shutdownCtx)
	}()
	
	log.Printf("Starting CANIDAE Ring on port %s", s.config.Port)
	log.Printf("Health: http://localhost:%s/health", s.config.Port)
	log.Printf("Ready: http://localhost:%s/ready", s.config.Port)
	log.Printf("Metrics: http://localhost:%s/metrics", s.config.Port)
	
	return server.ListenAndServe()
}

// handleHealth returns health status
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	status := "healthy"
	if err := s.ring.HealthCheck(); err != nil {
		status = "unhealthy"
	}
	
	response := map[string]interface{}{
		"status":    status,
		"service":   "canidae-ring",
		"timestamp": time.Now().Format(time.RFC3339),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleReady returns readiness status
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	providers := s.ring.ListProviders()
	
	response := map[string]interface{}{
		"ready":          len(providers) > 0,
		"service":        "canidae-ring",
		"providers":      len(providers),
		"nats_connected": s.ring.HealthCheck() == nil,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleMetrics returns Prometheus metrics
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	fmt.Fprintf(w, "# HELP canidae_ring_up CANIDAE Ring service status\n")
	fmt.Fprintf(w, "# TYPE canidae_ring_up gauge\n")
	fmt.Fprintf(w, "canidae_ring_up 1\n")
	fmt.Fprintf(w, "# HELP canidae_providers_total Total number of providers\n")
	fmt.Fprintf(w, "# TYPE canidae_providers_total gauge\n")
	fmt.Fprintf(w, "canidae_providers_total %d\n", len(s.ring.ListProviders()))
}

// handleProviders returns list of providers
func (s *Server) handleProviders(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	providers := s.ring.ListProviders()
	providerList := make([]map[string]interface{}, 0, len(providers))
	
	for _, p := range providers {
		status := p.GetStatus()
		providerList = append(providerList, map[string]interface{}{
			"id":           p.GetID(),
			"status":       status.Available,
			"health_score": status.HealthScore,
			"latency":      status.Latency,
			"error_rate":   status.ErrorRate,
		})
	}
	
	response := map[string]interface{}{
		"providers": providerList,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleProcess processes an AI request
func (s *Server) handleProcess(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var requestData map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&requestData); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}
	
	// Convert to types.Request
	req := &types.Request{
		ID:         fmt.Sprintf("req-%d", time.Now().UnixNano()),
		PackID:     getStringValue(requestData, "pack_id"),
		ProviderID: getStringValue(requestData, "provider_id"),
		Prompt:     getStringValue(requestData, "prompt"),
		Priority:   types.PriorityMedium,
		Timeout:    30 * time.Second,
		Timestamp:  time.Now(),
	}
	
	if params, ok := requestData["parameters"].(map[string]interface{}); ok {
		req.Parameters = params
	}
	
	// Process request
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	response, err := s.ring.Process(ctx, req)
	if err != nil {
		errorResponse := map[string]interface{}{
			"error":   "processing_failed",
			"message": err.Error(),
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(errorResponse)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// getStringValue safely gets a string value from a map
func getStringValue(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

// Close shuts down the server
func (s *Server) Close() error {
	return s.ring.Close()
}

func main() {
	fmt.Println("🐺 CANIDAE Ring Orchestrator v0.1.0")
	fmt.Println("=====================================")
	
	// Get configuration from environment
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	
	natsURL := os.Getenv("NATS_URL")
	if natsURL == "" {
		natsURL = "nats://localhost:4222"
	}
	
	// Create server
	config := ServerConfig{
		Port:        port,
		NatsURL:     natsURL,
		MetricsPort: "9090",
	}
	
	server, err := NewServer(config)
	if err != nil {
		log.Fatal("Failed to create server:", err)
	}
	defer server.Close()
	
	// Setup signal handling
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	
	go func() {
		<-sigChan
		log.Println("Shutdown signal received")
		cancel()
	}()
	
	// Start server
	if err := server.Start(ctx); err != nil && err != http.ErrServerClosed {
		log.Fatal("Server error:", err)
	}
	
	log.Println("Server shutdown complete")
}