# 🚀 CANIDAE Deployment Guide

Comprehensive guide for deploying CANIDAE to production and development environments.

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Production Deployment](#production-deployment)
- [Development Setup](#development-setup)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- **OS**: Ubuntu 24.04 LTS (tested) or compatible Linux
- **CPU**: 2+ cores recommended
- **Memory**: 4GB minimum, 8GB recommended
- **Storage**: 20GB available space
- **Network**: Static IP or reliable DHCP

### Software Requirements
```bash
# Check versions
podman --version  # 4.9.0 or higher
go version        # 1.23 or higher
```

### Installing Dependencies

#### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Podman
sudo apt install -y podman

# Install Go (if building from source)
wget https://go.dev/dl/go1.23.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.23.0.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc
```

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/macawi-ai/canidae.git
cd canidae
```

### 2. Run Quick Deploy
```bash
# On target server (e.g., 192.168.1.38)
./quick_deploy.sh
```

This creates:
- CANIDAE Ring orchestrator on port 8080
- NATS JetStream on port 4222
- Health endpoints at `/health` and `/ready`

## Production Deployment

### 1. Server Preparation

#### Create CANIDAE User
```bash
# Run setup script
./setup_canidae_user.sh
```

This creates:
- System user `canidae` for rootless Podman
- Directory structure in `/opt/canidae`
- Proper permissions and systemd lingering

#### Configure Sudo Permissions
```bash
# Add to /etc/sudoers.d/50-canidae
sudo visudo -f /etc/sudoers.d/50-canidae
```

Add the minimal required permissions (see `setup_synth_access.sh`).

### 2. Container Registry Configuration

```bash
# Configure registries for Podman
sudo cp deployment/podman/registries.conf /etc/containers/registries.conf
```

### 3. Build Container Images

```bash
# Build all images
./deployments/podman/scripts/deploy.sh build
```

Creates:
- `localhost/canidae-ring:dev|test|preprod`
- `localhost/canidae-providers:dev|test|preprod`

### 4. Deploy Environments

#### Development Environment
```bash
./deployments/podman/scripts/deploy.sh deploy dev
```

#### Test Environment
```bash
./deployments/podman/scripts/deploy.sh deploy test
```

#### Pre-Production Environment
```bash
./deployments/podman/scripts/deploy.sh deploy preprod
```

### 5. Systemd Service Setup

```bash
# Install systemd service
sudo cp deployments/podman/scripts/canidae-dev.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable canidae-dev
sudo systemctl start canidae-dev
```

## Development Setup

### Local Development
```bash
# Install dependencies
go mod download

# Run tests
go test ./...

# Build locally
go build -o canidae-ring cmd/canidae/main.go

# Run with environment variables
PORT=8080 NATS_URL=nats://localhost:4222 ./canidae-ring serve
```

### Using Podman Compose
```bash
cd deployments/podman/dev
podman-compose up
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Ring API port | 8080 |
| `NATS_URL` | NATS connection string | nats://localhost:4222 |
| `METRICS_PORT` | Prometheus metrics port | 9090 |
| `LOG_LEVEL` | Logging level (debug/info/warn/error) | info |
| `CHAOS_ENABLED` | Enable chaos engineering | false |
| `FLOW_CONTROL_ENABLED` | Enable rate limiting | true |

### Provider Configuration

Create `/opt/canidae/config/providers.yaml`:
```yaml
providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    rate_limit: 100
    timeout: 30s
    
  openai:
    api_key: ${OPENAI_API_KEY}
    rate_limit: 200
    timeout: 60s
```

### NATS Configuration

For production, configure NATS with:
```conf
# /opt/canidae/config/nats.conf
port: 4222
monitor_port: 8222

jetstream {
  store_dir: /opt/canidae/nats-data
  max_memory_store: 1GB
  max_file_store: 10GB
}

authorization {
  default_permissions {
    publish: ["canidae.>"]
    subscribe: ["canidae.>", "_INBOX.>"]
  }
}
```

## Monitoring

### Deploy Monitoring Stack
```bash
./deploy_monitoring.sh
```

Access points:
- **Prometheus**: http://server-ip:9091
- **Grafana**: http://server-ip:3001 (admin/canidae)

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `canidae_ring_up` | Service availability | < 1 |
| `canidae_request_duration_seconds` | Request latency | > 5s |
| `canidae_provider_errors_total` | Provider error rate | > 10/min |
| `canidae_flow_control_rejections` | Rate limit rejections | > 100/min |

### Creating Grafana Dashboard

1. Login to Grafana
2. Import dashboard from `/deployments/monitoring/dashboards/canidae.json`
3. Configure alerts as needed

## Troubleshooting

### Common Issues

#### Pod Won't Start
```bash
# Check pod status
podman pod ps -a

# View logs
podman pod logs canidae-dev

# Remove and recreate
podman pod rm -f canidae-dev
./quick_deploy.sh
```

#### NATS Connection Issues
```bash
# Test NATS connectivity
nats server ping -s nats://localhost:4222

# Check NATS logs
podman logs nats-dev
```

#### Port Conflicts
```bash
# Find process using port
sudo lsof -i :8080

# Use alternative ports
PORT=8081 ./canidae-ring serve
```

### Health Checks

```bash
# Check Ring health
curl http://localhost:8080/health

# Check readiness
curl http://localhost:8080/ready

# Check metrics
curl http://localhost:9090/metrics
```

### Debugging

#### Enable Debug Logging
```bash
LOG_LEVEL=debug ./canidae-ring serve
```

#### Container Shell Access
```bash
# Access running container
podman exec -it ring-dev /bin/sh

# Check container filesystem
podman run --rm -it localhost/canidae-ring:dev /bin/sh
```

### Recovery Procedures

#### Full System Reset
```bash
# Stop all services
./deployments/podman/scripts/deploy.sh stop all

# Clean all data
rm -rf /opt/canidae/*/nats-data/*

# Rebuild and redeploy
./deployments/podman/scripts/deploy.sh build
./deployments/podman/scripts/deploy.sh deploy all
```

## Security Considerations

### Network Security
- Use firewall rules to restrict access
- Enable TLS for production deployments
- Rotate API keys regularly

### Container Security
```bash
# Run security scan
podman scan localhost/canidae-ring:dev

# Check for vulnerabilities
trivy image localhost/canidae-ring:dev
```

## Backup and Recovery

### Backup NATS Data
```bash
# Create backup
tar -czf nats-backup-$(date +%Y%m%d).tar.gz /opt/canidae/*/nats-data
```

### Restore from Backup
```bash
# Stop services
./deployments/podman/scripts/deploy.sh stop all

# Restore data
tar -xzf nats-backup-20250816.tar.gz -C /

# Restart services
./deployments/podman/scripts/deploy.sh deploy all
```

## Support

- 📚 Documentation: https://github.com/macawi-ai/canidae/docs
- 🐛 Issues: https://github.com/macawi-ai/canidae/issues
- 💬 Discussions: https://github.com/macawi-ai/canidae/discussions

---

*The pack hunts as one* 🐺