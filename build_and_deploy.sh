#!/bin/bash
# Build and deploy CANIDAE containers

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🐺 CANIDAE Build & Deploy Script${NC}"
echo "=================================="

# Build the Ring orchestrator
echo -e "${BLUE}Building Ring orchestrator...${NC}"
cat > Containerfile.ring << 'EOF'
FROM docker.io/golang:1.23-alpine AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o canidae-ring cmd/canidae/main.go

FROM docker.io/alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /build/canidae-ring .
EXPOSE 8080 9090
CMD ["./canidae-ring", "serve"]
EOF

podman build -f Containerfile.ring -t localhost/canidae-ring:dev .
echo -e "${GREEN}✓ Ring orchestrator built${NC}"

# Build the Provider Gateway
echo -e "${BLUE}Building Provider Gateway...${NC}"
cat > Containerfile.providers << 'EOF'
FROM docker.io/golang:1.23-alpine AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o provider-gateway demo/demo.go

FROM docker.io/alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /build/provider-gateway .
EXPOSE 3000
CMD ["./provider-gateway"]
EOF

podman build -f Containerfile.providers -t localhost/canidae-providers:dev .
echo -e "${GREEN}✓ Provider Gateway built${NC}"

# Tag images for different environments
echo -e "${BLUE}Tagging images...${NC}"
podman tag localhost/canidae-ring:dev localhost/canidae-ring:test
podman tag localhost/canidae-ring:dev localhost/canidae-ring:preprod
podman tag localhost/canidae-providers:dev localhost/canidae-providers:test
podman tag localhost/canidae-providers:dev localhost/canidae-providers:preprod
echo -e "${GREEN}✓ Images tagged${NC}"

# Show built images
echo -e "${BLUE}Built images:${NC}"
podman images | grep -E "canidae|REPOSITORY"

echo ""
echo -e "${GREEN}✅ Build complete!${NC}"
echo ""
echo "To deploy:"
echo "  Dev environment:     ./deployments/podman/scripts/deploy.sh deploy dev"
echo "  Test environment:    ./deployments/podman/scripts/deploy.sh deploy test"
echo "  Preprod environment: ./deployments/podman/scripts/deploy.sh deploy preprod"