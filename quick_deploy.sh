#!/bin/bash
# Quick deployment script for CANIDAE with simplified builds

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🐺 CANIDAE Quick Deploy${NC}"
echo "========================="

# Build simplified Ring orchestrator
echo -e "${BLUE}Building Ring orchestrator (simplified)...${NC}"
cat > Containerfile.ring << 'EOF'
FROM docker.io/golang:1.23-alpine AS builder
WORKDIR /build
COPY cmd/canidae/simple_main.go ./
RUN CGO_ENABLED=0 GOOS=linux go build -o canidae-ring simple_main.go

FROM docker.io/alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /build/canidae-ring .
EXPOSE 8080
CMD ["./canidae-ring"]
EOF

podman build -f Containerfile.ring -t localhost/canidae-ring:dev .
echo -e "${GREEN}✓ Ring orchestrator built${NC}"

# Build NATS container
echo -e "${BLUE}Pulling NATS image...${NC}"
podman pull docker.io/nats:latest
podman tag docker.io/nats:latest localhost/nats:dev
echo -e "${GREEN}✓ NATS ready${NC}"

# Create pod and run containers
echo -e "${BLUE}Creating canidae-dev pod...${NC}"
podman pod rm -f canidae-dev 2>/dev/null || true
podman pod create --name canidae-dev -p 8080:8080 -p 4222:4222 -p 9090:9090

echo -e "${BLUE}Starting NATS...${NC}"
podman run -d --pod canidae-dev --name nats-dev localhost/nats:dev -js

echo -e "${BLUE}Starting Ring orchestrator...${NC}"
podman run -d --pod canidae-dev --name ring-dev \
  -e PORT=8080 \
  localhost/canidae-ring:dev

echo -e "${GREEN}✅ CANIDAE deployed!${NC}"
echo ""
echo "Services:"
echo "  Ring API:    http://$(hostname -I | awk '{print $1}'):8080"
echo "  Ring Health: http://$(hostname -I | awk '{print $1}'):8080/health"
echo "  NATS:        nats://$(hostname -I | awk '{print $1}'):4222"
echo ""
echo "Check status: podman pod ps"
echo "View logs:    podman logs ring-dev"