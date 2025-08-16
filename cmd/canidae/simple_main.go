package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"time"
)

func main() {
	fmt.Println("🐺 CANIDAE Ring Orchestrator v0.1.0")
	fmt.Println("=====================================")
	
	// Get port from env or default
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	
	// Health check endpoint
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"healthy","service":"canidae-ring","timestamp":"%s"}`, time.Now().Format(time.RFC3339))
	})
	
	// Ready check endpoint
	http.HandleFunc("/ready", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"ready":true,"service":"canidae-ring"}`)
	})
	
	// Basic info endpoint
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprintf(w, "CANIDAE Ring Orchestrator - The pack hunts as one 🐺\n")
	})
	
	// Metrics endpoint (placeholder)
	http.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprintf(w, "# HELP canidae_ring_up CANIDAE Ring service status\n")
		fmt.Fprintf(w, "# TYPE canidae_ring_up gauge\n")
		fmt.Fprintf(w, "canidae_ring_up 1\n")
	})
	
	log.Printf("Starting CANIDAE Ring on port %s", port)
	log.Printf("Health: http://localhost:%s/health", port)
	log.Printf("Ready: http://localhost:%s/ready", port)
	log.Printf("Metrics: http://localhost:%s/metrics", port)
	
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}