"""
Security management for AMP.
"""

import hashlib
import hmac
import json
import jwt
import time
from typing import Dict, Any, Optional
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key


class SecurityManager:
    """Manages security for AMP messages."""
    
    def __init__(self, api_key: Optional[str] = None, 
                 private_key: Optional[bytes] = None,
                 public_key: Optional[bytes] = None):
        self.api_key = api_key
        self.private_key = private_key
        self.public_key = public_key
        self.jwt_secret = api_key or "default-secret"
        
    def sign_message(self, message: Dict[str, Any]) -> str:
        """Sign message for integrity verification."""
        # Use HMAC-SHA256 for message signing
        secret_key = (self.api_key or "default-key").encode()
        message_bytes = json.dumps(message["message"], sort_keys=True).encode()
        
        signature = hmac.new(secret_key, message_bytes, hashlib.sha256).hexdigest()
        
        # Add signature to message headers
        if "headers" not in message["message"]:
            message["message"]["headers"] = {}
        
        message["message"]["headers"]["signature"] = {
            "algorithm": "HMAC-SHA256",
            "value": signature
        }
        
        return signature
    
    def verify_message(self, message: Dict[str, Any]) -> bool:
        """Verify message signature."""
        try:
            headers = message.get("message", {}).get("headers", {})
            signature_info = headers.get("signature", {})
            
            if not signature_info:
                return True  # No signature to verify
            
            algorithm = signature_info.get("algorithm")
            provided_signature = signature_info.get("value")
            
            if algorithm != "HMAC-SHA256":
                return False
            
            # Remove signature for verification
            message_copy = json.loads(json.dumps(message))
            del message_copy["message"]["headers"]["signature"]
            
            # Calculate expected signature
            secret_key = (self.api_key or "default-key").encode()
            message_bytes = json.dumps(message_copy["message"], sort_keys=True).encode()
            expected_signature = hmac.new(secret_key, message_bytes, hashlib.sha256).hexdigest()
            
            return hmac.compare_digest(expected_signature, provided_signature)
            
        except Exception:
            return False
    
    def create_jwt_token(self, agent_id: str, capabilities: list = None,
                        expires_in: int = 3600) -> str:
        """Create JWT token for agent authentication."""
        payload = {
            "agent_id": agent_id,
            "capabilities": capabilities or [],
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.InvalidTokenError:
            return None
    
    def encrypt_payload(self, payload: Dict[str, Any]) -> bytes:
        """Encrypt payload using RSA public key."""
        if not self.public_key:
            raise ValueError("Public key not configured")
        
        public_key = load_pem_public_key(self.public_key)
        payload_bytes = json.dumps(payload).encode()
        
        encrypted = public_key.encrypt(
            payload_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted
    
    def decrypt_payload(self, encrypted_payload: bytes) -> Dict[str, Any]:
        """Decrypt payload using RSA private key."""
        if not self.private_key:
            raise ValueError("Private key not configured")
        
        private_key = load_pem_private_key(self.private_key, password=None)
        
        decrypted = private_key.decrypt(
            encrypted_payload,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return json.loads(decrypted.decode())


class RateLimiter:
    """Rate limiting for API calls."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier."""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ]
        else:
            self.requests[identifier] = []
        
        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        
        return False
    
    def get_remaining(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        now = time.time()
        window_start = now - self.window_seconds
        
        if identifier in self.requests:
            current_count = len([
                req_time for req_time in self.requests[identifier]
                if req_time > window_start
            ])
            return max(0, self.max_requests - current_count)
        
        return self.max_requests


class AuthenticationManager:
    """Manages authentication for AMP agents."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.tokens: Dict[str, Dict[str, Any]] = {}
    
    def register_api_key(self, key: str, agent_id: str, 
                        capabilities: list = None, expires: Optional[float] = None):
        """Register an API key."""
        self.api_keys[key] = {
            "agent_id": agent_id,
            "capabilities": capabilities or [],
            "expires": expires,
            "created": time.time()
        }
    
    def validate_api_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return agent info."""
        if key not in self.api_keys:
            return None
        
        key_info = self.api_keys[key]
        
        # Check expiration
        if key_info.get("expires") and time.time() > key_info["expires"]:
            del self.api_keys[key]
            return None
        
        return key_info
    
    def create_session_token(self, agent_id: str, duration: int = 3600) -> str:
        """Create a session token."""
        token = hashlib.sha256(f"{agent_id}{time.time()}".encode()).hexdigest()
        
        self.tokens[token] = {
            "agent_id": agent_id,
            "expires": time.time() + duration,
            "created": time.time()
        }
        
        return token
    
    def validate_session_token(self, token: str) -> Optional[str]:
        """Validate session token and return agent ID."""
        if token not in self.tokens:
            return None
        
        token_info = self.tokens[token]
        
        # Check expiration
        if time.time() > token_info["expires"]:
            del self.tokens[token]
            return None
        
        return token_info["agent_id"]
    
    def revoke_token(self, token: str):
        """Revoke a session token."""
        self.tokens.pop(token, None)
    
    def cleanup_expired(self):
        """Clean up expired tokens and keys."""
        now = time.time()
        
        # Clean expired tokens
        expired_tokens = [
            token for token, info in self.tokens.items()
            if now > info["expires"]
        ]
        for token in expired_tokens:
            del self.tokens[token]
        
        # Clean expired API keys
        expired_keys = [
            key for key, info in self.api_keys.items()
            if info.get("expires") and now > info["expires"]
        ]
        for key in expired_keys:
            del self.api_keys[key]
