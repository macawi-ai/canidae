package observability

import (
	"context"
	"fmt"
	"net/http"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/prometheus"
	"go.opentelemetry.io/otel/metric"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// initMetrics initializes OpenTelemetry metrics with Prometheus exporter
func initMetrics(cfg Config) (*sdkmetric.MeterProvider, error) {
	// Create Prometheus exporter
	exporter, err := prometheus.New()
	if err != nil {
		return nil, fmt.Errorf("failed to create Prometheus exporter: %w", err)
	}

	// Create meter provider
	provider := sdkmetric.NewMeterProvider(
		sdkmetric.WithReader(exporter),
	)

	// Start Prometheus HTTP server if port is configured
	if cfg.PrometheusPort > 0 {
		go func() {
			mux := http.NewServeMux()
			mux.Handle("/metrics", promhttp.Handler())
			server := &http.Server{
				Addr:    fmt.Sprintf(":%d", cfg.PrometheusPort),
				Handler: mux,
			}
			if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				// Log error but don't crash the application
				fmt.Printf("Failed to start Prometheus server: %v\n", err)
			}
		}()
	}

	return provider, nil
}

// MetricsCollector provides custom metrics collection
type MetricsCollector struct {
	observer *Observer
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(obs *Observer) *MetricsCollector {
	return &MetricsCollector{observer: obs}
}

// CollectPackMetrics collects pack-specific metrics
func (mc *MetricsCollector) CollectPackMetrics(ctx context.Context, packID string, metrics PackMetrics) {
	if mc.observer.meter == nil {
		return
	}

	// Create counters for pack metrics
	packMessagesCounter, _ := mc.observer.meter.Int64Counter(
		"canidae.pack.messages.total",
	)
	
	packErrorsCounter, _ := mc.observer.meter.Int64Counter(
		"canidae.pack.errors.total",
	)
	
	packLatencyHistogram, _ := mc.observer.meter.Float64Histogram(
		"canidae.pack.latency",
	)

	// Record counter metrics with attributes
	attrs := []attribute.KeyValue{
		attribute.String("pack_id", packID),
	}
	
	packMessagesCounter.Add(ctx, metrics.MessageCount, metric.WithAttributes(attrs...))
	packErrorsCounter.Add(ctx, metrics.ErrorCount, metric.WithAttributes(attrs...))
	packLatencyHistogram.Record(ctx, metrics.Latency, metric.WithAttributes(attrs...))
}

// PackMetrics represents metrics for a pack
type PackMetrics struct {
	MemberCount  int64
	MessageCount int64
	ErrorCount   int64
	Latency      float64
}