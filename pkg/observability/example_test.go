package observability_test

import (
	"context"
	"fmt"
	"time"

	"github.com/macawi-ai/canidae/pkg/observability"
	"github.com/macawi-ai/canidae/pkg/observability/health"
)

func ExampleObserver() {
	// Create observability configuration
	cfg := observability.Config{
		ServiceName:    "canidae-ring",
		ServiceVersion: "v1.0.0",
		Environment:    "production",
		JaegerEndpoint: "http://localhost:14268/api/traces",
		PrometheusPort: 9090,
		AuditLogPath:   "/var/log/canidae/audit.log",
		EnableTracing:  true,
		EnableMetrics:  true,
		EnableAudit:    true,
	}

	// Initialize observer
	obs, err := observability.New(cfg)
	if err != nil {
		panic(err)
	}
	defer obs.Close()

	// Register health checks
	healthChecker := health.NewChecker()
	healthChecker.Register("nats", health.NATSCheck(func() error {
		// Check NATS connection
		return nil
	}))
	healthChecker.Register("database", health.DatabaseCheck(func(ctx context.Context) error {
		// Check database connection
		return nil
	}))

	// Example: Start a traced operation
	ctx := context.Background()
	ctx, span := obs.StartSpan(ctx, "process-request")
	defer span.End()

	// Record metrics
	start := time.Now()
	obs.IncrementActiveRequests(ctx)
	
	// Simulate some work
	time.Sleep(100 * time.Millisecond)
	
	// Record completion
	obs.DecrementActiveRequests(ctx)
	obs.RecordRequest(ctx, "POST", time.Since(start), "success")

	// Audit log important events
	obs.AuditLog(ctx, observability.AuditEvent{
		EventType: observability.EventAPIAccess,
		Actor:     "user-123",
		PackID:    "pack-456",
		Resource:  "/api/v1/agents",
		Action:    "CREATE",
		Result:    "SUCCESS",
		Severity:  observability.SeverityInfo,
		Metadata: map[string]interface{}{
			"agent_type": "anthropic",
			"model":      "claude-3",
		},
	})

	fmt.Println("Observability initialized and operational")
}

func ExampleMetricsCollector() {
	// Example of pack-specific observability
	cfg := observability.Config{
		ServiceName:    "canidae-pack-alpha",
		ServiceVersion: "v1.0.0",
		Environment:    "production",
		EnableMetrics:  true,
		PrometheusPort: 9091,
	}

	obs, _ := observability.New(cfg)
	defer obs.Close()

	// Collect pack-specific metrics
	collector := observability.NewMetricsCollector(obs)
	
	ctx := context.Background()
	collector.CollectPackMetrics(ctx, "pack-alpha", observability.PackMetrics{
		MemberCount:  5,
		MessageCount: 1000,
		ErrorCount:   2,
		Latency:      45.3,
	})

	fmt.Println("Pack metrics collected")
}