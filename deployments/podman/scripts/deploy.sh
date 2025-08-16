#!/bin/bash
# CANIDAE Deployment Script for canidae server (192.168.1.38)
# Uses rootless Podman with environment isolation

set -euo pipefail

# Configuration
CANIDAE_SERVER="192.168.1.38"
CANIDAE_USER="${CANIDAE_USER:-canidae}"
CANIDAE_BASE="/opt/canidae"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check if running on canidae server
check_server() {
    local current_ip=$(hostname -I | awk '{print $1}')
    if [[ "$current_ip" != "$CANIDAE_SERVER" ]]; then
        log_warn "Not running on canidae server ($CANIDAE_SERVER)"
        log_info "Current server: $current_ip"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Setup user for rootless Podman
setup_user() {
    log_info "Setting up user for rootless Podman..."
    
    if ! id "$CANIDAE_USER" &>/dev/null; then
        log_info "Creating user $CANIDAE_USER..."
        sudo useradd -r -m -s /bin/bash "$CANIDAE_USER"
    fi
    
    # Configure subuid/subgid for rootless containers
    if ! grep -q "^${CANIDAE_USER}:" /etc/subuid; then
        echo "${CANIDAE_USER}:100000:65536" | sudo tee -a /etc/subuid
    fi
    
    if ! grep -q "^${CANIDAE_USER}:" /etc/subgid; then
        echo "${CANIDAE_USER}:100000:65536" | sudo tee -a /etc/subgid
    fi
    
    # Enable lingering for systemd user services
    sudo loginctl enable-linger "$CANIDAE_USER"
    
    log_success "User setup complete"
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    local environments=("dev" "test" "preprod")
    local dirs=("nats-data" "config" "logs" "audit" "security" "results")
    
    for env in "${environments[@]}"; do
        for dir in "${dirs[@]}"; do
            sudo mkdir -p "${CANIDAE_BASE}/${env}/${dir}"
        done
    done
    
    # Create shared directories
    sudo mkdir -p "${CANIDAE_BASE}/secrets"
    sudo mkdir -p "${CANIDAE_BASE}/secrets-encrypted"
    sudo mkdir -p "${CANIDAE_BASE}/tls"
    sudo mkdir -p "${CANIDAE_BASE}/backups"
    
    # Set ownership
    sudo chown -R "${CANIDAE_USER}:${CANIDAE_USER}" "${CANIDAE_BASE}"
    
    log_success "Directories created"
}

# Generate TLS certificates for pre-production
generate_tls() {
    log_info "Generating TLS certificates..."
    
    local tls_dir="${CANIDAE_BASE}/tls"
    
    if [[ ! -f "${tls_dir}/ca.pem" ]]; then
        # Generate CA
        openssl req -new -x509 -days 365 -nodes \
            -keyout "${tls_dir}/ca-key.pem" \
            -out "${tls_dir}/ca.pem" \
            -subj "/C=US/ST=State/L=City/O=CANIDAE/CN=CANIDAE-CA"
        
        # Generate server cert
        openssl req -new -nodes \
            -keyout "${tls_dir}/key.pem" \
            -out "${tls_dir}/cert.csr" \
            -subj "/C=US/ST=State/L=City/O=CANIDAE/CN=canidae.local"
        
        openssl x509 -req -days 365 \
            -in "${tls_dir}/cert.csr" \
            -CA "${tls_dir}/ca.pem" \
            -CAkey "${tls_dir}/ca-key.pem" \
            -CAcreateserial \
            -out "${tls_dir}/cert.pem"
        
        # Set permissions
        chmod 600 "${tls_dir}"/*.pem
        sudo chown -R "${CANIDAE_USER}:${CANIDAE_USER}" "${tls_dir}"
        
        log_success "TLS certificates generated"
    else
        log_info "TLS certificates already exist"
    fi
}

# Build container images
build_images() {
    log_info "Building container images..."
    
    cd "${PROJECT_ROOT}"
    
    # Build Ring image
    log_info "Building Ring orchestrator image..."
    cat > Containerfile.ring << EOF
FROM golang:1.23-alpine AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o canidae-ring cmd/canidae/main.go

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /build/canidae-ring .
EXPOSE 8080 9090
CMD ["./canidae-ring", "serve"]
EOF
    
    podman build -f Containerfile.ring -t localhost/canidae-ring:dev .
    podman tag localhost/canidae-ring:dev localhost/canidae-ring:test
    podman tag localhost/canidae-ring:dev localhost/canidae-ring:preprod
    
    # Build Provider Gateway image (placeholder for now)
    log_info "Building Provider Gateway image..."
    cat > Containerfile.providers << EOF
FROM golang:1.23-alpine AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o provider-gateway demo/demo.go

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /build/provider-gateway .
EXPOSE 3000
CMD ["./provider-gateway"]
EOF
    
    podman build -f Containerfile.providers -t localhost/canidae-providers:dev .
    podman tag localhost/canidae-providers:dev localhost/canidae-providers:test
    podman tag localhost/canidae-providers:dev localhost/canidae-providers:preprod
    
    log_success "Images built successfully"
}

# Deploy environment
deploy_environment() {
    local env=$1
    log_info "Deploying $env environment..."
    
    local pod_file="${SCRIPT_DIR}/../${env}/canidae-${env}.yaml"
    
    if [[ ! -f "$pod_file" ]]; then
        log_error "Pod file not found: $pod_file"
    fi
    
    # Stop existing pod if running
    if podman pod exists "canidae-${env}" 2>/dev/null; then
        log_info "Stopping existing ${env} pod..."
        podman pod stop "canidae-${env}" || true
        podman pod rm "canidae-${env}" || true
    fi
    
    # Deploy the pod
    log_info "Starting ${env} pod..."
    sudo -u "$CANIDAE_USER" podman play kube "$pod_file"
    
    # Wait for pod to be ready
    sleep 5
    
    # Check pod status
    if sudo -u "$CANIDAE_USER" podman pod ps | grep -q "canidae-${env}"; then
        log_success "${env} environment deployed successfully"
    else
        log_error "Failed to deploy ${env} environment"
    fi
}

# Main deployment flow
main() {
    log_info "🐺 CANIDAE Deployment Script"
    log_info "=============================="
    
    # Parse arguments
    local action="${1:-help}"
    local env="${2:-all}"
    
    case "$action" in
        setup)
            check_server
            setup_user
            create_directories
            generate_tls
            log_success "Setup complete!"
            ;;
        build)
            build_images
            ;;
        deploy)
            if [[ "$env" == "all" ]]; then
                deploy_environment "dev"
                deploy_environment "test"
                deploy_environment "preprod"
            else
                deploy_environment "$env"
            fi
            ;;
        status)
            log_info "Pod status:"
            sudo -u "$CANIDAE_USER" podman pod ps
            log_info "Container status:"
            sudo -u "$CANIDAE_USER" podman ps
            ;;
        logs)
            if [[ "$env" == "all" ]]; then
                log_error "Please specify an environment (dev/test/preprod)"
            fi
            sudo -u "$CANIDAE_USER" podman pod logs "canidae-${env}"
            ;;
        stop)
            if [[ "$env" == "all" ]]; then
                for e in dev test preprod; do
                    sudo -u "$CANIDAE_USER" podman pod stop "canidae-${e}" || true
                done
            else
                sudo -u "$CANIDAE_USER" podman pod stop "canidae-${env}"
            fi
            ;;
        clean)
            log_warn "This will remove all pods and images. Continue? (y/N)"
            read -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                sudo -u "$CANIDAE_USER" podman pod rm -f -a
                sudo -u "$CANIDAE_USER" podman rmi -f -a
                log_success "Cleanup complete"
            fi
            ;;
        help|*)
            cat << EOF
Usage: $0 <action> [environment]

Actions:
  setup       - Initial server setup (users, directories, TLS)
  build       - Build container images
  deploy      - Deploy environment(s)
  status      - Show pod and container status
  logs        - Show logs for environment
  stop        - Stop environment(s)
  clean       - Remove all pods and images

Environments:
  dev         - Development environment
  test        - Testing environment
  preprod     - Pre-production environment
  all         - All environments (default)

Examples:
  $0 setup                    # Initial setup
  $0 build                    # Build images
  $0 deploy dev              # Deploy dev environment
  $0 deploy all              # Deploy all environments
  $0 logs test               # Show test environment logs
  $0 stop preprod            # Stop pre-production
EOF
            ;;
    esac
}

# Run main function
main "$@"