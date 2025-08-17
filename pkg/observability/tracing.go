package observability

import (
	"context"
	"fmt"

	"go.opentelemetry.io/otel/exporters/jaeger"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.21.0"
)

// initTracer initializes OpenTelemetry tracing with Jaeger
func initTracer(cfg Config) (*sdktrace.TracerProvider, error) {
	// Create resource with service information
	res, err := resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceName(cfg.ServiceName),
			semconv.ServiceVersion(cfg.ServiceVersion),
			semconv.DeploymentEnvironment(cfg.Environment),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Create exporter based on configuration
	var exporter sdktrace.SpanExporter
	if cfg.JaegerEndpoint != "" {
		// Use Jaeger exporter for production
		exporter, err = jaeger.New(
			jaeger.WithCollectorEndpoint(
				jaeger.WithEndpoint(cfg.JaegerEndpoint),
			),
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create Jaeger exporter: %w", err)
		}
	} else {
		// Use stdout exporter for development
		exporter, err = stdouttrace.New(
			stdouttrace.WithPrettyPrint(),
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create stdout exporter: %w", err)
		}
	}

	// Create tracer provider with batch span processor
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	)

	return tp, nil
}

// TracingMiddleware provides HTTP middleware for automatic tracing
func TracingMiddleware(obs *Observer, next func(context.Context) error) func(context.Context) error {
	return func(ctx context.Context) error {
		ctx, span := obs.StartSpan(ctx, "http.request")
		defer span.End()
		
		// Increment active requests
		obs.IncrementActiveRequests(ctx)
		defer obs.DecrementActiveRequests(ctx)
		
		return next(ctx)
	}
}