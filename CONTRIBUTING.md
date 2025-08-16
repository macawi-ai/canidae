# 🐺 Contributing to CANIDAE

Welcome to the pack! We're excited you're interested in contributing to CANIDAE.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Requirements](#testing-requirements)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

### Our Pack Values
- **Respect**: Every pack member deserves respect, regardless of experience level
- **Collaboration**: We hunt together, not alone
- **Growth**: Help others learn and be open to learning yourself
- **Quality**: Take pride in your contributions

### Unacceptable Behavior
- Harassment, discrimination, or hostile communication
- Trolling, insulting comments, or personal attacks
- Publishing others' private information
- Any conduct inappropriate for a professional setting

## Getting Started

### Prerequisites
- Go 1.23 or higher
- Podman 4.9+ (for containerization)
- NATS CLI tools (optional but recommended)
- Git

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/macawi-ai/canidae.git
cd canidae

# Install dependencies
go mod download

# Run tests
go test ./...

# Build the project
go build -o canidae-ring cmd/canidae/main.go
```

## Development Workflow

### Branch Strategy
We use a simplified Git Flow:
- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - Feature branches
- `fix/*` - Bug fix branches
- `docs/*` - Documentation updates

### Creating a Feature Branch
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

## Coding Standards

### Go Code Style
- Follow standard Go formatting (`gofmt`)
- Use `golangci-lint` for linting
- Maximum line length: 120 characters
- Use meaningful variable names

### Code Organization
```
/cmd           - Main applications
/internal      - Private application code
/pkg           - Public libraries
/api           - API definitions and protobuf
/deployments   - Deployment configurations
/test          - Integration tests
/docs          - Documentation
```

### Error Handling
```go
// Good
if err != nil {
    return fmt.Errorf("failed to connect to NATS: %w", err)
}

// Bad
if err != nil {
    return err
}
```

## Commit Guidelines

We follow Conventional Commits specification:

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Examples
```bash
feat(ring): implement circuit breaker for provider calls
fix(nats): resolve connection timeout issue
docs(api): add OpenAPI specification for HOWL protocol
```

## Pull Request Process

### Before Submitting
1. **Test your changes**: Run `go test ./...`
2. **Lint your code**: Run `golangci-lint run`
3. **Update documentation**: If you changed APIs or behavior
4. **Write/update tests**: Maintain or improve coverage

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### Review Process
1. Submit PR against `develop` branch
2. Ensure CI checks pass
3. Request review from maintainers
4. Address feedback
5. Maintainer merges upon approval

## Testing Requirements

### Unit Tests
- Minimum 80% code coverage for new code
- Test edge cases and error conditions
- Use table-driven tests where appropriate

### Integration Tests
```go
// Test files should be named *_test.go
func TestRingOrchestration(t *testing.T) {
    // Setup
    ring := setupTestRing(t)
    defer ring.Close()
    
    // Test
    result, err := ring.Process(testRequest)
    
    // Assert
    require.NoError(t, err)
    assert.Equal(t, expected, result)
}
```

### Running Tests
```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run specific package tests
go test ./internal/ring/...

# Run with race detector
go test -race ./...
```

## Issue Reporting

### Before Creating an Issue
1. Check existing issues (including closed ones)
2. Verify you're using the latest version
3. Gather relevant information (logs, configs, environment)

### Issue Template
Use the appropriate template:
- 🐛 Bug Report
- ✨ Feature Request
- 📚 Documentation
- ❓ Question

### Good Issue Example
```markdown
**Bug Description**
Ring orchestrator fails to reconnect after NATS restart

**Steps to Reproduce**
1. Start CANIDAE with `./canidae-ring serve`
2. Restart NATS server
3. Observe connection errors without recovery

**Expected Behavior**
Automatic reconnection with exponential backoff

**Environment**
- CANIDAE version: 0.1.0
- Go version: 1.23
- OS: Ubuntu 24.04
```

## Questions?

- 💬 GitHub Discussions for questions
- 🐛 GitHub Issues for bugs
- 📧 Contact maintainers: canidae@macawi-ai.com

---

Thank you for contributing to CANIDAE! Together, the pack grows stronger. 🐺