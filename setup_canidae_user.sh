#!/bin/bash
# Setup CANIDAE user with our specific sudo permissions

set -euo pipefail

BLUE='\033[0;34m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${BLUE}🐺 Setting up CANIDAE user and directories${NC}"
echo "==========================================="

# Create canidae user
echo -e "${BLUE}Creating canidae user...${NC}"
if ! sudo /usr/bin/id canidae &>/dev/null; then
    sudo /usr/sbin/useradd -r -m -s /bin/bash canidae
    echo -e "${GREEN}✓ User created${NC}"
else
    echo -e "${GREEN}✓ User already exists${NC}"
fi

# Configure subuid/subgid for rootless containers
echo -e "${BLUE}Configuring subuid/subgid...${NC}"
if ! sudo /usr/bin/grep "^canidae:" /etc/subuid &>/dev/null; then
    echo "canidae:100000:65536" | sudo /usr/bin/tee -a /etc/subuid
    echo -e "${GREEN}✓ subuid configured${NC}"
else
    echo -e "${GREEN}✓ subuid already configured${NC}"
fi

if ! sudo /usr/bin/grep "^canidae:" /etc/subgid &>/dev/null; then
    echo "canidae:100000:65536" | sudo /usr/bin/tee -a /etc/subgid
    echo -e "${GREEN}✓ subgid configured${NC}"
else
    echo -e "${GREEN}✓ subgid already configured${NC}"
fi

# Enable lingering for systemd user services
echo -e "${BLUE}Enabling systemd lingering...${NC}"
sudo /usr/bin/loginctl enable-linger canidae
echo -e "${GREEN}✓ Lingering enabled${NC}"

# Create directory structure
echo -e "${BLUE}Creating /opt/canidae directory structure...${NC}"
for env in dev test preprod; do
    for dir in nats-data config logs audit security results; do
        sudo /usr/bin/mkdir -p /opt/canidae/${env}/${dir}
    done
done

# Create shared directories
sudo /usr/bin/mkdir -p /opt/canidae/secrets
sudo /usr/bin/mkdir -p /opt/canidae/secrets-encrypted
sudo /usr/bin/mkdir -p /opt/canidae/tls
sudo /usr/bin/mkdir -p /opt/canidae/backups

# Set ownership
echo -e "${BLUE}Setting ownership...${NC}"
sudo /usr/bin/chown -R canidae:canidae /opt/canidae
echo -e "${GREEN}✓ Ownership set${NC}"

echo ""
echo -e "${GREEN}✅ CANIDAE user and directories setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Generate TLS certificates"
echo "  2. Build container images"
echo "  3. Deploy environments"