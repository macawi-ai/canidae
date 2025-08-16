#!/bin/bash
# Deploy monitoring stack for CANIDAE

set -euo pipefail

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 Deploying CANIDAE Monitoring Stack${NC}"
echo "======================================="

# Create monitoring config directory
echo -e "${BLUE}Creating monitoring configs...${NC}"
mkdir -p monitoring/prometheus
mkdir -p monitoring/grafana

# Prometheus configuration
cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'canidae-ring'
    static_configs:
      - targets: ['localhost:8080']
        labels:
          service: 'ring'
          environment: 'dev'
  
  - job_name: 'nats'
    static_configs:
      - targets: ['localhost:8222']
        labels:
          service: 'nats'
          environment: 'dev'
EOF

# Grafana datasource config
cat > monitoring/grafana/datasources.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
EOF

# Create monitoring pod
echo -e "${BLUE}Creating monitoring pod...${NC}"
podman pod rm -f canidae-monitoring 2>/dev/null || true
podman pod create --name canidae-monitoring \
  -p 9091:9090 \
  -p 3001:3000 \
  --network bridge

# Run Prometheus
echo -e "${BLUE}Starting Prometheus...${NC}"
podman run -d --pod canidae-monitoring \
  --name prometheus-dev \
  -v ./monitoring/prometheus:/etc/prometheus:ro \
  docker.io/prom/prometheus:latest \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --web.console.libraries=/usr/share/prometheus/console_libraries \
  --web.console.templates=/usr/share/prometheus/consoles

# Run Grafana
echo -e "${BLUE}Starting Grafana...${NC}"
podman run -d --pod canidae-monitoring \
  --name grafana-dev \
  -v ./monitoring/grafana:/etc/grafana/provisioning/datasources:ro \
  -e GF_SECURITY_ADMIN_USER=admin \
  -e GF_SECURITY_ADMIN_PASSWORD=canidae \
  -e GF_INSTALL_PLUGINS=redis-datasource \
  docker.io/grafana/grafana:latest

echo -e "${GREEN}✅ Monitoring stack deployed!${NC}"
echo ""
echo "Access points:"
echo "  Prometheus: http://$(hostname -I | awk '{print $1}'):9091"
echo "  Grafana:    http://$(hostname -I | awk '{print $1}'):3001"
echo "              User: admin / Password: canidae"
echo ""
echo "Status: podman pod ps"