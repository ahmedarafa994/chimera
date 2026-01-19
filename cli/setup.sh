#!/bin/bash

# Chimera CLI Setup Script
# This script helps set up the Chimera CLI tool for CI/CD integration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
    else
        print_error "Python is not installed or not in PATH"
        exit 1
    fi

    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 7 ]; then
        print_error "Python 3.7 or higher is required (found $PYTHON_VERSION)"
        exit 1
    fi

    print_success "Python $PYTHON_VERSION found"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."

    if [ -f "$CLI_DIR/requirements.txt" ]; then
        $PYTHON_CMD -m pip install --user -r "$CLI_DIR/requirements.txt"
        print_success "Dependencies installed"
    else
        print_warning "requirements.txt not found, installing basic dependencies"
        $PYTHON_CMD -m pip install --user requests urllib3
    fi
}

# Make CLI executable
setup_cli() {
    print_status "Setting up CLI tool..."

    CLI_SCRIPT="$CLI_DIR/chimera-cli.py"

    if [ -f "$CLI_SCRIPT" ]; then
        chmod +x "$CLI_SCRIPT"
        print_success "CLI tool is executable"
    else
        print_error "CLI script not found at $CLI_SCRIPT"
        exit 1
    fi
}

# Create configuration template
create_config_template() {
    if [ ! -f "$CLI_DIR/chimera-config.json" ]; then
        print_status "Creating configuration template..."
        $PYTHON_CMD "$CLI_DIR/chimera-cli.py" init --output "$CLI_DIR/chimera-config.json"
        print_success "Configuration template created at $CLI_DIR/chimera-config.json"
    else
        print_warning "Configuration template already exists"
    fi
}

# Add CLI to PATH (optional)
add_to_path() {
    if [ "$1" = "--add-to-path" ]; then
        print_status "Adding CLI to PATH..."

        # Create a symlink in ~/.local/bin (which should be in PATH)
        LOCAL_BIN="$HOME/.local/bin"
        mkdir -p "$LOCAL_BIN"

        if [ -L "$LOCAL_BIN/chimera-cli" ]; then
            rm "$LOCAL_BIN/chimera-cli"
        fi

        ln -s "$CLI_DIR/chimera-cli.py" "$LOCAL_BIN/chimera-cli"
        print_success "CLI symlinked to $LOCAL_BIN/chimera-cli"
        print_status "Make sure $LOCAL_BIN is in your PATH"
    fi
}

# Validate installation
validate_installation() {
    print_status "Validating installation..."

    # Test basic CLI functionality
    if $PYTHON_CMD "$CLI_DIR/chimera-cli.py" --help &> /dev/null; then
        print_success "CLI tool is working correctly"
    else
        print_error "CLI tool validation failed"
        exit 1
    fi

    # Test configuration validation
    if [ -f "$CLI_DIR/examples/smoke-test-suite.json" ]; then
        if $PYTHON_CMD "$CLI_DIR/chimera-cli.py" validate --config "$CLI_DIR/examples/smoke-test-suite.json" &> /dev/null; then
            print_success "Configuration validation works"
        else
            print_warning "Configuration validation test failed (this is OK if examples are not present)"
        fi
    fi
}

# Print usage information
print_usage() {
    cat << EOF

${GREEN}Chimera CLI Setup Complete!${NC}

${BLUE}Usage:${NC}
  # Generate configuration template
  $PYTHON_CMD $CLI_DIR/chimera-cli.py init

  # Validate configuration
  $PYTHON_CMD $CLI_DIR/chimera-cli.py validate --config your-config.json

  # Run security tests
  $PYTHON_CMD $CLI_DIR/chimera-cli.py test --config your-config.json --api-key YOUR_API_KEY

${BLUE}Environment Variables:${NC}
  CHIMERA_API_KEY     Your Chimera API key
  CHIMERA_BASE_URL    Chimera API base URL (default: http://localhost:8001)

${BLUE}Example CI/CD Integration:${NC}
  See examples/ directory for GitHub Actions and GitLab CI configurations

${BLUE}Next Steps:${NC}
  1. Set your API key: export CHIMERA_API_KEY=your_api_key_here
  2. Edit the configuration template at $CLI_DIR/chimera-config.json
  3. Test your setup: $PYTHON_CMD $CLI_DIR/chimera-cli.py validate --config $CLI_DIR/chimera-config.json

EOF
}

# Main setup function
main() {
    print_status "Setting up Chimera CLI for CI/CD Integration..."
    echo

    check_python
    install_dependencies
    setup_cli
    create_config_template
    add_to_path "$@"
    validate_installation

    print_usage
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        cat << EOF
Chimera CLI Setup Script

Usage: $0 [OPTIONS]

Options:
  --add-to-path    Add CLI tool to PATH via symlink in ~/.local/bin
  --help, -h       Show this help message

This script will:
1. Check Python installation (requires Python 3.7+)
2. Install required Python dependencies
3. Make the CLI script executable
4. Create a configuration template
5. Validate the installation

EOF
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac