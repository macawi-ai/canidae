// Package observability provides comprehensive monitoring, tracing, and auditing
// for the CANIDAE platform. Sister Gemini's wisdom: "Visibility is Power"
package observability

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/trace"
)

// Observer manages all observability components
type Observer struct {
	tracer       trace.Tracer
	meter        metric.Meter
	auditLogger  *AuditLogger
	healthChecks map[string]HealthCheck
	mu           sync.RWMutex
	
	// Metrics
	requestCounter   metric.Int64Counter
	requestDuration  metric.Float64Histogram
	activeRequests   metric.Int64UpDownCounter
	errorCounter     metric.Int64Counter
}

// Config holds observability configuration
type Config struct {
	ServiceName     string
	ServiceVersion  string
	Environment     string
	JaegerEndpoint  string
	PrometheusPort  int
	AuditLogPath    string
	EnableTracing   bool
	EnableMetrics   bool
	EnableAudit     bool
}

// HealthCheck represents a component health check
type HealthCheck func(ctx context.Context) error

// New creates a new Observer with the given configuration
func New(cfg Config) (*Observer, error) {
	obs := &Observer{
		healthChecks: make(map[string]HealthCheck),
	}

	// Initialize tracing
	if cfg.EnableTracing {
		tp, err := initTracer(cfg)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize tracer: %w", err)
		}
		otel.SetTracerProvider(tp)
		obs.tracer = tp.Tracer(cfg.ServiceName)
	}

	// Initialize metrics
	if cfg.EnableMetrics {
		mp, err := initMetrics(cfg)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize metrics: %w", err)
		}
		otel.SetMeterProvider(mp)
		obs.meter = mp.Meter(cfg.ServiceName)
		
		// Create metric instruments
		if err := obs.createMetrics(); err != nil {
			return nil, fmt.Errorf("failed to create metrics: %w", err)
		}
	}

	// Initialize audit logging
	if cfg.EnableAudit {
		al, err := NewAuditLogger(cfg.AuditLogPath)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize audit logger: %w", err)
		}
		obs.auditLogger = al
	}

	return obs, nil
}

// createMetrics initializes all metric instruments
func (o *Observer) createMetrics() error {
	var err error
	
	o.requestCounter, err = o.meter.Int64Counter(
		"canidae.requests.total",
	)
	if err != nil {
		return err
	}

	o.requestDuration, err = o.meter.Float64Histogram(
		"canidae.request.duration",
	)
	if err != nil {
		return err
	}

	o.activeRequests, err = o.meter.Int64UpDownCounter(
		"canidae.requests.active",
	)
	if err != nil {
		return err
	}

	o.errorCounter, err = o.meter.Int64Counter(
		"canidae.errors.total",
	)
	if err != nil {
		return err
	}

	return nil
}

// StartSpan starts a new trace span
func (o *Observer) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	if o.tracer == nil {
		// Return a noop span when tracing is disabled
		return ctx, trace.SpanFromContext(ctx)
	}
	return o.tracer.Start(ctx, name, opts...)
}

// RecordRequest records request metrics
func (o *Observer) RecordRequest(ctx context.Context, method string, duration time.Duration, status string) {
	if o.meter == nil {
		return
	}

	attrs := []attribute.KeyValue{
		attribute.String("method", method),
		attribute.String("status", status),
	}

	o.requestCounter.Add(ctx, 1, metric.WithAttributes(attrs...))
	o.requestDuration.Record(ctx, duration.Seconds()*1000, metric.WithAttributes(attrs...))
}

// RecordError records an error occurrence
func (o *Observer) RecordError(ctx context.Context, errorType string) {
	if o.errorCounter == nil {
		return
	}
	
	o.errorCounter.Add(ctx, 1, metric.WithAttributes(
		attribute.String("error_type", errorType),
	))
}

// IncrementActiveRequests increments the active request counter
func (o *Observer) IncrementActiveRequests(ctx context.Context) {
	if o.activeRequests != nil {
		o.activeRequests.Add(ctx, 1)
	}
}

// DecrementActiveRequests decrements the active request counter
func (o *Observer) DecrementActiveRequests(ctx context.Context) {
	if o.activeRequests != nil {
		o.activeRequests.Add(ctx, -1)
	}
}

// AuditLog writes an audit log entry
func (o *Observer) AuditLog(ctx context.Context, event AuditEvent) {
	if o.auditLogger != nil {
		o.auditLogger.Log(ctx, event)
	}
}

// RegisterHealthCheck registers a health check for a component
func (o *Observer) RegisterHealthCheck(name string, check HealthCheck) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.healthChecks[name] = check
}

// CheckHealth runs all registered health checks
func (o *Observer) CheckHealth(ctx context.Context) map[string]error {
	o.mu.RLock()
	defer o.mu.RUnlock()
	
	results := make(map[string]error)
	for name, check := range o.healthChecks {
		results[name] = check(ctx)
	}
	return results
}

// Close cleanly shuts down all observability components
func (o *Observer) Close() error {
	if o.auditLogger != nil {
		return o.auditLogger.Close()
	}
	return nil
}

