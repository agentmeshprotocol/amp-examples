# Security Documentation - AMP Examples

## Overview

This document provides security guidelines for the Agent Mesh Protocol (AMP) examples repository. It covers secure example development, validation procedures, and security considerations for educational and demonstration code.

---

## Example Security Principles

### Security-First Example Development

The AMP examples repository follows strict security guidelines to ensure that all example code:

1. **Demonstrates secure practices** by default
2. **Prevents security anti-patterns** from being propagated
3. **Includes security annotations** and explanations
4. **Validates inputs** and handles errors securely
5. **Uses realistic security configurations** without real credentials

### Example Categories and Security Levels

#### Level 1: Basic Examples (Low Security Risk)
- Simple protocol demonstrations
- Basic message formats
- Local-only examples
- No network communication

#### Level 2: Intermediate Examples (Medium Security Risk)  
- Local agent communication
- Simple authentication demos
- File-based configuration
- Basic error handling

#### Level 3: Advanced Examples (High Security Risk)
- Network communication
- Production-like configurations
- Multi-agent orchestration
- Integration with external services

#### Level 4: Production Examples (Highest Security Risk)
- Enterprise deployment patterns
- Full security implementations
- Real-world integration scenarios
- Performance and scale demonstrations

---

## Secure Example Development Guidelines

### Code Security Standards

#### Input Validation in Examples
```python
# ✅ GOOD: Proper input validation in examples
def secure_example_agent():
    """Example agent with proper input validation"""
    
    def validate_agent_input(agent_id: str) -> bool:
        """Validate agent ID format"""
        if not isinstance(agent_id, str):
            raise ValueError("Agent ID must be a string")
        
        if not re.match(r'^[a-zA-Z0-9\-_\.]+$', agent_id):
            raise ValueError("Agent ID contains invalid characters")
        
        if len(agent_id) > 128:
            raise ValueError("Agent ID too long")
        
        return True
    
    # Example usage with validation
    try:
        agent_id = input("Enter agent ID: ")
        validate_agent_input(agent_id)
        print(f"Valid agent ID: {agent_id}")
    except ValueError as e:
        print(f"Invalid input: {e}")

# ❌ BAD: No input validation
def insecure_example_agent():
    """Example agent without input validation - DO NOT USE"""
    agent_id = input("Enter agent ID: ")  # No validation!
    # This could lead to injection attacks or crashes
    print(f"Agent ID: {agent_id}")
```

#### Credential Management in Examples
```python
# ✅ GOOD: Secure credential handling in examples
import os
from typing import Optional

class ExampleCredentialManager:
    """Example credential manager for demonstrations"""
    
    def __init__(self):
        # Use environment variables for credentials
        self.api_key = os.getenv('AMP_EXAMPLE_API_KEY')
        self.secret_key = os.getenv('AMP_EXAMPLE_SECRET_KEY')
        
        if not self.api_key:
            print("WARNING: No API key found. Using demo mode.")
            self.api_key = "demo-key-for-testing-only"
    
    def get_api_key(self) -> str:
        """Get API key for examples"""
        if self.api_key == "demo-key-for-testing-only":
            print("ℹ️  Using demo API key. Set AMP_EXAMPLE_API_KEY for real usage.")
        return self.api_key

# ❌ BAD: Hardcoded credentials in examples
API_KEY = "sk-1234567890abcdef"  # Never do this!
SECRET = "my-secret-key"         # Extremely dangerous!
```

#### Error Handling in Examples
```typescript
// ✅ GOOD: Secure error handling in examples
export class SecureExampleClient {
  async sendMessage(message: AMPMessage): Promise<AMPMessage> {
    try {
      const response = await this.httpClient.post('/messages', message);
      return validateAMPMessage(response.data);
    } catch (error) {
      // Log error details internally
      console.error('Example client error:', {
        timestamp: new Date().toISOString(),
        error: error instanceof Error ? error.message : 'Unknown error'
      });
      
      // Return generic error to prevent information leakage
      throw new AMPError('Failed to send message', AMPErrorCode.NETWORK_ERROR);
    }
  }
}

// ❌ BAD: Exposing sensitive information in errors
export class InsecureExampleClient {
  async sendMessage(message: AMPMessage): Promise<AMPMessage> {
    try {
      const response = await this.httpClient.post('/messages', message);
      return response.data;
    } catch (error) {
      // This exposes internal details!
      throw new Error(`API call failed: ${JSON.stringify(error)}`);
    }
  }
}
```

### Security Annotations in Examples

#### Documentation Standards
```python
"""
Example: Secure Multi-Agent Communication

🔒 SECURITY CONSIDERATIONS:
- Uses environment variables for configuration
- Implements proper input validation
- Includes error handling without information leakage
- Demonstrates secure authentication patterns

⚠️  PRODUCTION NOTES:
- Change default secret keys before production use
- Implement proper certificate validation for HTTPS
- Add rate limiting and monitoring
- Use secure random number generation

🧪 TESTING:
- Run with: python secure_multi_agent_example.py
- Set environment: export AMP_EXAMPLE_API_KEY="your-key"
- Validate with: python -m amp.examples.validate secure_multi_agent_example.py
"""

import os
import secrets
from typing import Dict, Any
from amp_python_sdk import AMPClient, AMPMessage

def create_secure_example_agent(agent_id: str) -> AMPClient:
    """Create a securely configured example agent"""
    
    # Security check: Validate agent ID
    if not re.match(r'^[a-zA-Z0-9\-_\.]+$', agent_id):
        raise ValueError("Invalid agent ID format")
    
    # Security best practice: Use environment variables
    api_key = os.getenv('AMP_EXAMPLE_API_KEY', 'demo-key-insecure')
    base_url = os.getenv('AMP_EXAMPLE_BASE_URL', 'https://demo.agentmeshprotocol.io')
    
    # Security warning for demo credentials
    if api_key == 'demo-key-insecure':
        print("⚠️  WARNING: Using demo credentials. Set AMP_EXAMPLE_API_KEY for secure operation.")
    
    # Create client with security options
    client = AMPClient(
        base_url=base_url,
        api_key=api_key,
        verify_ssl=True,  # Always verify SSL in examples
        timeout=30.0,     # Reasonable timeout
        rate_limit=10     # Conservative rate limiting
    )
    
    return client
```

#### Code Comments for Security
```typescript
// 🔒 SECURITY: JWT token management example
export class ExampleJWTManager {
  private secretKey: string;
  
  constructor() {
    // 🔒 SECURITY: Never hardcode secrets in production
    this.secretKey = process.env.AMP_EXAMPLE_JWT_SECRET || 
                     'demo-secret-change-in-production-32chars';
    
    // 🔒 SECURITY: Validate secret key length
    if (this.secretKey.length < 32) {
      console.warn('⚠️  JWT secret key should be at least 32 characters');
    }
  }
  
  async generateToken(agentId: string): Promise<string> {
    // 🔒 SECURITY: Validate agent ID to prevent injection
    if (!/^[a-zA-Z0-9\-_\.]+$/.test(agentId)) {
      throw new Error('Invalid agent ID format');
    }
    
    // 🔒 SECURITY: Set reasonable expiration time
    const expirationTime = Math.floor(Date.now() / 1000) + (60 * 60); // 1 hour
    
    const payload = {
      agent_id: agentId,
      iat: Math.floor(Date.now() / 1000),
      exp: expirationTime,
      iss: 'amp-examples'  // 🔒 SECURITY: Always set issuer
    };
    
    return jwt.sign(payload, this.secretKey, { algorithm: 'HS256' });
  }
}
```

---

## Example Validation Framework

### Automated Security Validation

#### Security Linting for Examples
```yaml
# .github/workflows/example-security.yml
name: Example Security Validation
on: [push, pull_request]

jobs:
  security-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          
      - name: Install security tools
        run: |
          pip install bandit safety semgrep
          npm install -g eslint eslint-plugin-security
          
      - name: Scan Python examples
        run: |
          find examples/ -name "*.py" -exec bandit {} \;
          find examples/ -name "requirements.txt" -exec safety check -r {} \;
          
      - name: Scan TypeScript examples
        run: |
          find examples/ -name "*.ts" -name "*.js" -exec eslint --config .eslintrc.security.js {} \;
          
      - name: Run Semgrep on all examples
        run: semgrep --config=security-audit examples/
        
      - name: Validate example security
        run: python scripts/validate_example_security.py examples/
```

#### Example Security Validator
```python
#!/usr/bin/env python3
"""
Example Security Validator

Validates that example code follows security best practices.
"""

import ast
import os
import re
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path

class SecurityIssue:
    def __init__(self, severity: str, message: str, file_path: str, line_number: int = 0):
        self.severity = severity  # 'critical', 'high', 'medium', 'low'
        self.message = message
        self.file_path = file_path
        self.line_number = line_number

class ExampleSecurityValidator:
    """Validates security practices in example code"""
    
    def __init__(self):
        self.issues: List[SecurityIssue] = []
        
        # Security patterns to detect
        self.dangerous_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password detected'),
            (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key detected'),
            (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret detected'),
            (r'eval\s*\(', 'Use of eval() function detected'),
            (r'exec\s*\(', 'Use of exec() function detected'),
            (r'shell=True', 'Dangerous shell execution detected'),
            (r'verify=False', 'SSL verification disabled'),
            (r'check_hostname=False', 'Hostname verification disabled'),
        ]
        
        # Required security practices
        self.required_patterns = [
            (r'import\s+os', 'Should use environment variables'),
            (r'try:', 'Should include error handling'),
            (r'validate_', 'Should include input validation'),
        ]
    
    def validate_python_file(self, file_path: str) -> None:
        """Validate Python example file"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for dangerous patterns
        for pattern, message in self.dangerous_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                self.issues.append(SecurityIssue(
                    'high', 
                    f'{message}: {match.group()}',
                    file_path,
                    line_number
                ))
        
        # Parse AST for deeper analysis
        try:
            tree = ast.parse(content)
            self._analyze_ast(tree, file_path)
        except SyntaxError as e:
            self.issues.append(SecurityIssue(
                'medium',
                f'Syntax error prevents security analysis: {e}',
                file_path,
                e.lineno or 0
            ))
    
    def _analyze_ast(self, tree: ast.AST, file_path: str) -> None:
        """Analyze Python AST for security issues"""
        
        for node in ast.walk(tree):
            # Check for hardcoded strings that look like secrets
            if isinstance(node, ast.Str):
                if self._looks_like_secret(node.s):
                    self.issues.append(SecurityIssue(
                        'high',
                        f'Potential hardcoded secret: {node.s[:10]}...',
                        file_path,
                        getattr(node, 'lineno', 0)
                    ))
            
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        self.issues.append(SecurityIssue(
                            'critical',
                            f'Dangerous function call: {node.func.id}',
                            file_path,
                            getattr(node, 'lineno', 0)
                        ))
    
    def _looks_like_secret(self, value: str) -> bool:
        """Heuristic to detect if a string looks like a secret"""
        
        # Skip common non-secret strings
        if len(value) < 8:
            return False
        
        if value in ['localhost', '127.0.0.1', 'example.com', 'test-agent']:
            return False
        
        # Check for patterns that look like secrets
        if re.match(r'^[a-zA-Z0-9+/]{16,}={0,2}$', value):  # Base64-like
            return True
        
        if re.match(r'^[0-9a-f]{32,}$', value):  # Hex string
            return True
        
        if re.match(r'^sk-[a-zA-Z0-9]{20,}$', value):  # OpenAI-style key
            return True
        
        return False
    
    def validate_typescript_file(self, file_path: str) -> None:
        """Validate TypeScript/JavaScript example file"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for dangerous patterns
        for pattern, message in self.dangerous_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                self.issues.append(SecurityIssue(
                    'high',
                    f'{message}: {match.group()}',
                    file_path,
                    line_number
                ))
        
        # TypeScript-specific checks
        ts_patterns = [
            (r'any\s*\[\]', 'Use of any[] type reduces type safety'),
            (r'@ts-ignore', 'TypeScript error suppression detected'),
            (r'process\.env\.[A-Z_]+\s*\|\|\s*["\'][^"\']+["\']', 'Default fallback may be insecure'),
        ]
        
        for pattern, message in ts_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                line_number = content[:match.start()].count('\n') + 1
                self.issues.append(SecurityIssue(
                    'medium',
                    message,
                    file_path,
                    line_number
                ))
    
    def validate_configuration_file(self, file_path: str) -> None:
        """Validate configuration files (JSON, YAML, etc.)"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    config = json.load(f)
                    self._validate_json_config(config, file_path)
                elif file_path.endswith(('.yml', '.yaml')):
                    import yaml
                    config = yaml.safe_load(f)
                    self._validate_yaml_config(config, file_path)
        except Exception as e:
            self.issues.append(SecurityIssue(
                'medium',
                f'Failed to parse configuration file: {e}',
                file_path
            ))
    
    def _validate_json_config(self, config: Dict[str, Any], file_path: str) -> None:
        """Validate JSON configuration for security issues"""
        
        def check_dict(obj: Any, path: str = '') -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check for sensitive keys
                    if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token']):
                        if isinstance(value, str) and len(value) > 8:
                            self.issues.append(SecurityIssue(
                                'high',
                                f'Potential secret in configuration: {current_path}',
                                file_path
                            ))
                    
                    check_dict(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_dict(item, f"{path}[{i}]")
        
        check_dict(config)
    
    def _validate_yaml_config(self, config: Any, file_path: str) -> None:
        """Validate YAML configuration for security issues"""
        # Similar to JSON validation
        self._validate_json_config(config, file_path)
    
    def validate_directory(self, directory: str) -> None:
        """Validate all example files in a directory"""
        
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and common non-example directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__']]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                if file.endswith('.py'):
                    self.validate_python_file(file_path)
                elif file.endswith(('.ts', '.js')):
                    self.validate_typescript_file(file_path)
                elif file.endswith(('.json', '.yml', '.yaml')):
                    self.validate_configuration_file(file_path)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate security validation report"""
        
        issues_by_severity = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        
        for issue in self.issues:
            issues_by_severity[issue.severity].append({
                'message': issue.message,
                'file': issue.file_path,
                'line': issue.line_number
            })
        
        return {
            'total_issues': len(self.issues),
            'issues_by_severity': issues_by_severity,
            'summary': {
                'critical': len(issues_by_severity['critical']),
                'high': len(issues_by_severity['high']),
                'medium': len(issues_by_severity['medium']),
                'low': len(issues_by_severity['low'])
            }
        }

def main():
    """Main validation function"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python validate_example_security.py <examples_directory>")
        sys.exit(1)
    
    examples_dir = sys.argv[1]
    
    if not os.path.isdir(examples_dir):
        print(f"Error: {examples_dir} is not a directory")
        sys.exit(1)
    
    validator = ExampleSecurityValidator()
    validator.validate_directory(examples_dir)
    
    report = validator.generate_report()
    
    print("Example Security Validation Report")
    print("=" * 40)
    print(f"Total Issues: {report['total_issues']}")
    print(f"Critical: {report['summary']['critical']}")
    print(f"High: {report['summary']['high']}")
    print(f"Medium: {report['summary']['medium']}")
    print(f"Low: {report['summary']['low']}")
    print()
    
    # Print detailed issues
    for severity in ['critical', 'high', 'medium', 'low']:
        issues = report['issues_by_severity'][severity]
        if issues:
            print(f"{severity.upper()} Issues:")
            for issue in issues:
                print(f"  - {issue['file']}:{issue['line']} - {issue['message']}")
            print()
    
    # Exit with error code if critical or high issues found
    if report['summary']['critical'] > 0 or report['summary']['high'] > 0:
        print("❌ Security validation failed due to critical or high severity issues")
        sys.exit(1)
    else:
        print("✅ Security validation passed")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

---

## Example Security Templates

### Secure Example Template (Python)
```python
#!/usr/bin/env python3
"""
AMP Example: [Example Name]

🔒 SECURITY LEVEL: [Low/Medium/High/Production]

📝 DESCRIPTION:
[Brief description of what this example demonstrates]

🔒 SECURITY CONSIDERATIONS:
- Uses environment variables for configuration
- Implements proper input validation
- Includes comprehensive error handling
- Demonstrates secure authentication
- [Add specific security measures]

⚠️  PRODUCTION NOTES:
- Change all demo credentials before production use
- Implement proper logging and monitoring
- Add rate limiting and DDoS protection
- Use secure random number generation
- [Add specific production considerations]

🧪 USAGE:
- Install: pip install amp-python-sdk
- Setup: export AMP_EXAMPLE_API_KEY="your-key"
- Run: python [filename].py
- Test: python -m pytest test_[filename].py

📋 REQUIREMENTS:
- Python 3.11+
- AMP Python SDK 1.0+
- [Additional requirements]
"""

import os
import sys
import logging
import secrets
import re
from typing import Dict, Any, Optional
from datetime import datetime

# Import AMP SDK
try:
    from amp_python_sdk import AMPClient, AMPMessage, AMPError
except ImportError:
    print("❌ AMP Python SDK not installed. Run: pip install amp-python-sdk")
    sys.exit(1)

# Configure secure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('example.log')
    ]
)
logger = logging.getLogger(__name__)

class SecureExampleConfig:
    """Secure configuration management for examples"""
    
    def __init__(self):
        # 🔒 SECURITY: Use environment variables for sensitive data
        self.api_key = os.getenv('AMP_EXAMPLE_API_KEY')
        self.base_url = os.getenv('AMP_EXAMPLE_BASE_URL', 'https://demo.agentmeshprotocol.io')
        self.agent_id = os.getenv('AMP_EXAMPLE_AGENT_ID', f'example-agent-{secrets.token_hex(4)}')
        
        # 🔒 SECURITY: Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration for security"""
        
        if not self.api_key:
            logger.warning("⚠️  No API key provided. Using demo mode.")
            self.api_key = "demo-key-for-testing-only"
        
        # 🔒 SECURITY: Validate agent ID format
        if not re.match(r'^[a-zA-Z0-9\-_\.]+$', self.agent_id):
            raise ValueError("Invalid agent ID format")
        
        # 🔒 SECURITY: Ensure HTTPS in production
        if not self.base_url.startswith('https://') and 'localhost' not in self.base_url:
            logger.warning("⚠️  Non-HTTPS URL detected. Use HTTPS in production.")

def validate_user_input(user_input: str, input_type: str = "general") -> str:
    """
    🔒 SECURITY: Validate and sanitize user input
    
    Args:
        user_input: Raw user input
        input_type: Type of input for specific validation
    
    Returns:
        Validated and sanitized input
    
    Raises:
        ValueError: If input is invalid
    """
    
    if not isinstance(user_input, str):
        raise ValueError("Input must be a string")
    
    # Basic sanitization
    sanitized = user_input.strip()
    
    if input_type == "agent_id":
        if not re.match(r'^[a-zA-Z0-9\-_\.]+$', sanitized):
            raise ValueError("Agent ID contains invalid characters")
        if len(sanitized) > 128:
            raise ValueError("Agent ID too long")
    
    elif input_type == "message":
        if len(sanitized) > 10000:
            raise ValueError("Message too long")
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', sanitized)
    
    return sanitized

def create_secure_amp_client(config: SecureExampleConfig) -> AMPClient:
    """
    🔒 SECURITY: Create securely configured AMP client
    
    Args:
        config: Secure configuration object
    
    Returns:
        Configured AMP client
    """
    
    try:
        client = AMPClient(
            base_url=config.base_url,
            api_key=config.api_key,
            agent_id=config.agent_id,
            verify_ssl=True,  # 🔒 SECURITY: Always verify SSL
            timeout=30.0,     # 🔒 SECURITY: Reasonable timeout
            rate_limit=10     # 🔒 SECURITY: Conservative rate limiting
        )
        
        logger.info(f"✅ AMP client created for agent: {config.agent_id}")
        return client
        
    except Exception as e:
        logger.error(f"❌ Failed to create AMP client: {e}")
        raise AMPError(f"Client creation failed: {e}")

def secure_message_handler(message: AMPMessage) -> AMPMessage:
    """
    🔒 SECURITY: Secure message handling with validation
    
    Args:
        message: Incoming AMP message
    
    Returns:
        Response message
    """
    
    try:
        # 🔒 SECURITY: Validate message format
        if not message.message.source.agent_id:
            raise ValueError("Missing source agent ID")
        
        # 🔒 SECURITY: Validate payload
        payload = message.message.payload
        if not isinstance(payload, dict):
            raise ValueError("Invalid payload format")
        
        # Process message securely
        response_payload = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "processed_by": "secure-example-agent"
        }
        
        # 🔒 SECURITY: Create response with validation
        response = AMPMessage(
            protocol="AMP/1.0",
            message={
                "id": f"response-{secrets.token_hex(8)}",
                "type": "response",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": {"agent_id": "secure-example-agent"},
                "destination": message.message.source,
                "correlation_id": message.message.id,
                "payload": response_payload
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Message handling error: {e}")
        
        # 🔒 SECURITY: Return generic error response
        error_response = AMPMessage(
            protocol="AMP/1.0",
            message={
                "id": f"error-{secrets.token_hex(8)}",
                "type": "error",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": {"agent_id": "secure-example-agent"},
                "destination": message.message.source,
                "correlation_id": message.message.id,
                "payload": {
                    "error": "MESSAGE_PROCESSING_FAILED",
                    "message": "Failed to process message"
                }
            }
        )
        
        return error_response

def main():
    """
    🔒 SECURITY: Main example function with comprehensive error handling
    """
    
    try:
        # Initialize secure configuration
        config = SecureExampleConfig()
        
        # Create secure AMP client
        client = create_secure_amp_client(config)
        
        # Example usage
        logger.info("🚀 Starting secure AMP example")
        
        # 🔒 SECURITY: Get user input with validation
        try:
            target_agent = input("Enter target agent ID (or press Enter for demo): ").strip()
            if target_agent:
                target_agent = validate_user_input(target_agent, "agent_id")
            else:
                target_agent = "demo-target-agent"
        except ValueError as e:
            logger.error(f"❌ Invalid input: {e}")
            return
        
        # Create and send secure message
        message = AMPMessage(
            protocol="AMP/1.0",
            message={
                "id": f"example-{secrets.token_hex(8)}",
                "type": "request",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source": {"agent_id": config.agent_id},
                "destination": {"agent_id": target_agent, "capability": "echo"},
                "payload": {
                    "message": "Hello from secure example!",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
        
        # Send message with error handling
        try:
            response = client.send_message(message)
            logger.info("✅ Message sent successfully")
            logger.info(f"📬 Response: {response.message.payload}")
        except AMPError as e:
            logger.error(f"❌ AMP Error: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        
    except KeyboardInterrupt:
        logger.info("🛑 Example interrupted by user")
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        sys.exit(1)
    finally:
        logger.info("🏁 Example completed")

if __name__ == "__main__":
    main()
```

### Secure Example Template (TypeScript)
```typescript
#!/usr/bin/env node
/**
 * AMP Example: [Example Name]
 * 
 * 🔒 SECURITY LEVEL: [Low/Medium/High/Production]
 * 
 * 📝 DESCRIPTION:
 * [Brief description of what this example demonstrates]
 * 
 * 🔒 SECURITY CONSIDERATIONS:
 * - Uses environment variables for configuration
 * - Implements proper input validation with Zod
 * - Includes comprehensive error handling
 * - Demonstrates secure authentication
 * - [Add specific security measures]
 * 
 * ⚠️  PRODUCTION NOTES:
 * - Change all demo credentials before production use
 * - Implement proper logging and monitoring
 * - Add rate limiting and DDoS protection
 * - Use secure random number generation
 * - [Add specific production considerations]
 * 
 * 🧪 USAGE:
 * - Install: npm install amp-typescript-sdk
 * - Setup: export AMP_EXAMPLE_API_KEY="your-key"
 * - Run: node [filename].js
 * - Test: npm test
 * 
 * 📋 REQUIREMENTS:
 * - Node.js 18+
 * - AMP TypeScript SDK 1.0+
 * - [Additional requirements]
 */

import { randomBytes } from 'node:crypto';
import { z } from 'zod';

// 🔒 SECURITY: Import AMP SDK with error handling
let AMPClient: any, AMPMessage: any, AMPError: any;
try {
  ({ AMPClient, AMPMessage, AMPError } = require('amp-typescript-sdk'));
} catch (error) {
  console.error('❌ AMP TypeScript SDK not installed. Run: npm install amp-typescript-sdk');
  process.exit(1);
}

// 🔒 SECURITY: Input validation schemas
const AgentIdSchema = z.string()
  .min(1, "Agent ID required")
  .max(128, "Agent ID too long")
  .regex(/^[a-zA-Z0-9\-_\.]+$/, "Invalid agent ID format");

const MessageSchema = z.string()
  .max(10000, "Message too long")
  .transform(msg => msg.replace(/[<>"']/g, '')); // Basic XSS prevention

interface SecureExampleConfig {
  apiKey: string;
  baseUrl: string;
  agentId: string;
}

class SecureExampleConfig {
  public readonly apiKey: string;
  public readonly baseUrl: string;
  public readonly agentId: string;

  constructor() {
    // 🔒 SECURITY: Use environment variables for sensitive data
    this.apiKey = process.env.AMP_EXAMPLE_API_KEY || '';
    this.baseUrl = process.env.AMP_EXAMPLE_BASE_URL || 'https://demo.agentmeshprotocol.io';
    this.agentId = process.env.AMP_EXAMPLE_AGENT_ID || `example-agent-${randomBytes(4).toString('hex')}`;

    // 🔒 SECURITY: Validate configuration
    this.validateConfig();
  }

  private validateConfig(): void {
    if (!this.apiKey) {
      console.warn('⚠️  No API key provided. Using demo mode.');
      (this as any).apiKey = 'demo-key-for-testing-only';
    }

    // 🔒 SECURITY: Validate agent ID format
    try {
      AgentIdSchema.parse(this.agentId);
    } catch (error) {
      throw new Error(`Invalid agent ID format: ${this.agentId}`);
    }

    // 🔒 SECURITY: Ensure HTTPS in production
    if (!this.baseUrl.startsWith('https://') && !this.baseUrl.includes('localhost')) {
      console.warn('⚠️  Non-HTTPS URL detected. Use HTTPS in production.');
    }
  }
}

/**
 * 🔒 SECURITY: Validate and sanitize user input
 */
function validateUserInput(userInput: string, inputType: 'agent_id' | 'message' | 'general' = 'general'): string {
  if (typeof userInput !== 'string') {
    throw new Error('Input must be a string');
  }

  const sanitized = userInput.trim();

  switch (inputType) {
    case 'agent_id':
      return AgentIdSchema.parse(sanitized);
    case 'message':
      return MessageSchema.parse(sanitized);
    default:
      return sanitized;
  }
}

/**
 * 🔒 SECURITY: Create securely configured AMP client
 */
function createSecureAMPClient(config: SecureExampleConfig): any {
  try {
    const client = new AMPClient({
      baseUrl: config.baseUrl,
      apiKey: config.apiKey,
      agentId: config.agentId,
      verifySSL: true,    // 🔒 SECURITY: Always verify SSL
      timeout: 30000,     // 🔒 SECURITY: Reasonable timeout
      rateLimit: 10       // 🔒 SECURITY: Conservative rate limiting
    });

    console.log(`✅ AMP client created for agent: ${config.agentId}`);
    return client;

  } catch (error) {
    console.error(`❌ Failed to create AMP client: ${error}`);
    throw new AMPError(`Client creation failed: ${error}`);
  }
}

/**
 * 🔒 SECURITY: Secure message handling with validation
 */
function secureMessageHandler(message: any): any {
  try {
    // 🔒 SECURITY: Validate message format
    if (!message.message?.source?.agent_id) {
      throw new Error('Missing source agent ID');
    }

    // 🔒 SECURITY: Validate payload
    const payload = message.message.payload;
    if (typeof payload !== 'object' || payload === null) {
      throw new Error('Invalid payload format');
    }

    // Process message securely
    const responsePayload = {
      status: 'success',
      timestamp: new Date().toISOString(),
      processed_by: 'secure-example-agent'
    };

    // 🔒 SECURITY: Create response with validation
    const response = {
      protocol: 'AMP/1.0',
      message: {
        id: `response-${randomBytes(8).toString('hex')}`,
        type: 'response',
        timestamp: new Date().toISOString(),
        source: { agent_id: 'secure-example-agent' },
        destination: message.message.source,
        correlation_id: message.message.id,
        payload: responsePayload
      }
    };

    return response;

  } catch (error) {
    console.error(`❌ Message handling error: ${error}`);

    // 🔒 SECURITY: Return generic error response
    const errorResponse = {
      protocol: 'AMP/1.0',
      message: {
        id: `error-${randomBytes(8).toString('hex')}`,
        type: 'error',
        timestamp: new Date().toISOString(),
        source: { agent_id: 'secure-example-agent' },
        destination: message.message.source,
        correlation_id: message.message.id,
        payload: {
          error: 'MESSAGE_PROCESSING_FAILED',
          message: 'Failed to process message'
        }
      }
    };

    return errorResponse;
  }
}

/**
 * 🔒 SECURITY: Main example function with comprehensive error handling
 */
async function main(): Promise<void> {
  try {
    // Initialize secure configuration
    const config = new SecureExampleConfig();

    // Create secure AMP client
    const client = createSecureAMPClient(config);

    // Example usage
    console.log('🚀 Starting secure AMP example');

    // 🔒 SECURITY: Get user input with validation
    const readline = require('readline').createInterface({
      input: process.stdin,
      output: process.stdout
    });

    const targetAgent = await new Promise<string>((resolve) => {
      readline.question('Enter target agent ID (or press Enter for demo): ', (answer: string) => {
        readline.close();
        try {
          const sanitized = answer.trim();
          if (sanitized) {
            resolve(validateUserInput(sanitized, 'agent_id'));
          } else {
            resolve('demo-target-agent');
          }
        } catch (error) {
          console.error(`❌ Invalid input: ${error}`);
          process.exit(1);
        }
      });
    });

    // Create and send secure message
    const message = {
      protocol: 'AMP/1.0',
      message: {
        id: `example-${randomBytes(8).toString('hex')}`,
        type: 'request',
        timestamp: new Date().toISOString(),
        source: { agent_id: config.agentId },
        destination: { agent_id: targetAgent, capability: 'echo' },
        payload: {
          message: 'Hello from secure example!',
          timestamp: new Date().toISOString()
        }
      }
    };

    // Send message with error handling
    try {
      const response = await client.sendMessage(message);
      console.log('✅ Message sent successfully');
      console.log(`📬 Response: ${JSON.stringify(response.message.payload, null, 2)}`);
    } catch (error) {
      if (error instanceof AMPError) {
        console.error(`❌ AMP Error: ${error.message}`);
      } else {
        console.error(`❌ Unexpected error: ${error}`);
      }
    }

  } catch (error) {
    if (error instanceof Error && error.message === 'SIGINT') {
      console.log('🛑 Example interrupted by user');
    } else {
      console.error(`❌ Example failed: ${error}`);
      process.exit(1);
    }
  } finally {
    console.log('🏁 Example completed');
  }
}

// Handle process termination gracefully
process.on('SIGINT', () => {
  console.log('\n🛑 Example interrupted by user');
  process.exit(0);
});

process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  console.error('❌ Unhandled rejection:', reason);
  process.exit(1);
});

// Run the example
if (require.main === module) {
  main().catch(error => {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  });
}

export { main, SecureExampleConfig, validateUserInput, createSecureAMPClient };
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Create organization-level SECURITY.md with vulnerability reporting procedures", "status": "completed", "priority": "high"}, {"id": "2", "content": "Create security advisory template for coordinated vulnerability disclosure", "status": "completed", "priority": "high"}, {"id": "3", "content": "Create security checklist for maintainers and contributors", "status": "completed", "priority": "high"}, {"id": "4", "content": "Create agentmeshprotocol repository security documentation", "status": "completed", "priority": "high"}, {"id": "5", "content": "Create amp-python-sdk security documentation", "status": "completed", "priority": "high"}, {"id": "6", "content": "Create amp-typescript-sdk security documentation", "status": "completed", "priority": "high"}, {"id": "7", "content": "Create amp-examples security validation documentation", "status": "completed", "priority": "medium"}]