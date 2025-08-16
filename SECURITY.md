# 🔒 Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security seriously at CANIDAE. If you discover a security vulnerability, please follow these steps:

### 1. Do NOT Create a Public Issue
Security vulnerabilities should not be reported via public GitHub issues.

### 2. Send a Private Report
Email security details to: security@macawi-ai.com

Include:
- Type of vulnerability
- Full paths of affected source files
- Location of affected code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce
- Proof-of-concept or exploit code (if possible)
- Impact assessment

### 3. Response Timeline
- **Initial Response**: Within 48 hours
- **Status Update**: Within 5 business days
- **Resolution Target**: 30 days for critical, 90 days for low severity

## Security Best Practices

### API Keys and Secrets
- Never commit API keys or secrets to the repository
- Use environment variables for sensitive configuration
- Rotate credentials regularly
- Use secret management tools in production

### Container Security
```bash
# Scan images before deployment
podman scan localhost/canidae-ring:latest

# Run containers with minimal privileges
podman run --security-opt=no-new-privileges:true ...

# Use read-only filesystems where possible
podman run --read-only ...
```

### Network Security
- Use TLS for all production deployments
- Implement rate limiting on all endpoints
- Validate all input data
- Use network policies to restrict pod communication

### Authentication & Authorization
- Implement strong authentication for all admin endpoints
- Use JWT tokens with short expiration times
- Implement RBAC for multi-tenant deployments
- Log all authentication attempts

## Security Features

### Built-in Protections
- **Rate Limiting**: Configurable per-provider limits
- **Circuit Breakers**: Automatic failure recovery
- **Input Validation**: Strict validation of all requests
- **Audit Logging**: Comprehensive activity logging
- **Chaos Engineering**: Resilience testing capabilities

### Recommended Configurations

#### Production Settings
```yaml
security:
  tls:
    enabled: true
    cert_file: /opt/canidae/tls/cert.pem
    key_file: /opt/canidae/tls/key.pem
  
  rate_limiting:
    enabled: true
    global_limit: 1000
    per_ip_limit: 100
  
  authentication:
    type: jwt
    secret: ${JWT_SECRET}
    expiry: 3600
  
  audit:
    enabled: true
    log_file: /opt/canidae/audit/audit.log
    include_payload: false
```

## Vulnerability Disclosure

After a security issue is resolved:
1. Security advisory will be published
2. CVE will be requested if applicable
3. All users will be notified via GitHub Security Advisories

## Security Checklist

### For Contributors
- [ ] No hardcoded credentials
- [ ] Input validation implemented
- [ ] Error messages don't leak sensitive info
- [ ] Dependencies are up to date
- [ ] Security tests included

### For Deployments
- [ ] TLS enabled for production
- [ ] Firewall rules configured
- [ ] Secrets stored securely
- [ ] Monitoring and alerting active
- [ ] Backup strategy implemented
- [ ] Incident response plan ready

## Contact

**Security Team Email**: security@macawi-ai.com
**PGP Key**: Available at https://macawi-ai.com/pgp-key.asc

---

*Protecting the pack* 🐺🔒