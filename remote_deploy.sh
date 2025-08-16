#!/bin/bash
# Remote deployment helper for CANIDAE
# This script helps coordinate deployment to the canidae server

set -euo pipefail

CANIDAE_SERVER="192.168.1.38"
CANIDAE_USER="cy"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🐺 CANIDAE Remote Deployment Helper${NC}"
echo "======================================"

# Step 1: Ensure project is synced
echo -e "${BLUE}[1/5]${NC} Syncing project to canidae server..."
rsync -avzP --exclude '.git' --exclude 'node_modules' . ${CANIDAE_USER}@${CANIDAE_SERVER}:~/canidae/ | grep -E "sent|total size" || true

# Step 2: Run setup (requires manual interaction)
echo -e "${BLUE}[2/5]${NC} Running setup on canidae server..."
echo -e "${YELLOW}Note: You'll need to enter your sudo password on the remote server${NC}"
echo ""
echo "Please run these commands on the canidae server:"
echo -e "${GREEN}ssh ${CANIDAE_USER}@${CANIDAE_SERVER}${NC}"
echo -e "${GREEN}cd ~/canidae${NC}"
echo -e "${GREEN}chmod +x deployments/podman/scripts/deploy.sh${NC}"
echo -e "${GREEN}./deployments/podman/scripts/deploy.sh setup${NC}"
echo ""
echo "Press ENTER when setup is complete..."
read -r

# Step 3: Build images
echo -e "${BLUE}[3/5]${NC} Building container images..."
ssh ${CANIDAE_USER}@${CANIDAE_SERVER} "cd ~/canidae && ./deployments/podman/scripts/deploy.sh build"

# Step 4: Deploy dev environment
echo -e "${BLUE}[4/5]${NC} Deploying development environment..."
ssh ${CANIDAE_USER}@${CANIDAE_SERVER} "cd ~/canidae && ./deployments/podman/scripts/deploy.sh deploy dev"

# Step 5: Check status
echo -e "${BLUE}[5/5]${NC} Checking deployment status..."
ssh ${CANIDAE_USER}@${CANIDAE_SERVER} "cd ~/canidae && ./deployments/podman/scripts/deploy.sh status"

echo ""
echo -e "${GREEN}✅ Deployment process complete!${NC}"
echo ""
echo "To view logs:"
echo -e "  ssh ${CANIDAE_USER}@${CANIDAE_SERVER} 'cd ~/canidae && ./deployments/podman/scripts/deploy.sh logs dev'"
echo ""
echo "To access the services:"
echo -e "  Ring API: http://${CANIDAE_SERVER}:8080"
echo -e "  NATS: nats://${CANIDAE_SERVER}:4222"
echo -e "  Metrics: http://${CANIDAE_SERVER}:9090/metrics"