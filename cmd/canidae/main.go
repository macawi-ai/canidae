package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/canidae/canidae/internal/ring"
	"github.com/nats-io/nats.go"
)

var (
	version = "0.1.0-alpha"
	commit  = "unknown"
)

func main() {
	var (
		natsURL     = flag.String("nats", nats.DefaultURL, "NATS server URL")
		showVersion = flag.Bool("version", false, "Show version information")
		configPath  = flag.String("config", "", "Path to configuration file")
	)
	flag.Parse()

	if *showVersion {
		fmt.Printf("CANIDAE v%s (commit: %s)\n", version, commit)
		fmt.Println("Pack-Oriented AI Orchestration Platform")
		os.Exit(0)
	}

	// Initialize logging
	log.SetPrefix("[CANIDAE] ")
	log.Printf("Starting CANIDAE v%s...", version)

	// Create context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle shutdown gracefully
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Println("Shutting down gracefully...")
		cancel()
	}()

	// Connect to NATS
	log.Printf("Connecting to NATS at %s...", *natsURL)
	nc, err := nats.Connect(*natsURL,
		nats.Name("canidae-alpha"),
		nats.MaxReconnects(-1),
		nats.ReconnectWait(nats.DefaultReconnectWait),
		nats.DisconnectErrHandler(func(_ *nats.Conn, err error) {
			if err != nil {
				log.Printf("NATS disconnected: %v", err)
			}
		}),
		nats.ReconnectHandler(func(_ *nats.Conn) {
			log.Println("NATS reconnected")
		}),
	)
	if err != nil {
		log.Fatalf("Failed to connect to NATS: %v", err)
	}
	defer nc.Close()

	// Initialize the Ring (orchestration engine)
	r, err := ring.New(ring.Config{
		NatsConn:   nc,
		ConfigPath: *configPath,
	})
	if err != nil {
		log.Fatalf("Failed to initialize Ring: %v", err)
	}

	// Start the Ring
	log.Println("Starting the Ring orchestration engine...")
	if err := r.Start(ctx); err != nil {
		log.Fatalf("Ring failed: %v", err)
	}

	log.Println("CANIDAE shutdown complete")
}