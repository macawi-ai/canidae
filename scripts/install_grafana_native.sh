#!/bin/bash
# Install Grafana natively on CANIDAE server
# Since containers keep getting terminated

set -e

echo "Installing Grafana natively on CANIDAE..."

# Download Grafana binary
cd /tmp
wget -q https://dl.grafana.com/oss/release/grafana-12.2.0.linux-amd64.tar.gz
tar -zxf grafana-12.2.0.linux-amd64.tar.gz

# Move to home directory
mv grafana-v12.2.0 ~/grafana

# Create data directory
mkdir -p ~/grafana-data

# Create minimal config
cat > ~/grafana/conf/custom.ini << EOF
[server]
http_port = 3000
http_addr = 0.0.0.0

[database]
path = /home/cy/grafana-data/grafana.db

[paths]
data = /home/cy/grafana-data
logs = /home/cy/grafana-data/logs
plugins = /home/cy/grafana-data/plugins
provisioning = /home/cy/grafana/conf/provisioning

[analytics]
enabled = false
check_for_updates = false

[security]
admin_user = admin
admin_password = admin
EOF

echo "Grafana installed to ~/grafana"
echo "To start: cd ~/grafana && ./bin/grafana-server -config conf/custom.ini"