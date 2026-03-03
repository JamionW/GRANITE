"""
GRANITE Error Classes

Provides specific exception types for actionable error messages.
"""


class GRANITEError(Exception):
 """Base exception for GRANITE errors."""
 pass


class DataNotFoundError(GRANITEError):
 """Raised when required data files are missing."""
 
 def __init__(self, file_path: str, data_type: str, download_url: str = None):
 self.file_path = file_path
 self.data_type = data_type
 self.download_url = download_url
 
 message = f"{data_type} not found at {file_path}."
 if download_url:
 message += f" Download from: {download_url}"
 
 super().__init__(message)


class ServerConnectionError(GRANITEError):
 """Raised when required servers are not accessible."""
 
 def __init__(self, server_name: str, url: str, start_command: str = None):
 self.server_name = server_name
 self.url = url
 self.start_command = start_command
 
 message = f"Cannot connect to {server_name} at {url}."
 if start_command:
 message += f" Start with: {start_command}"
 
 super().__init__(message)


class ConfigurationError(GRANITEError):
 """Raised when configuration is invalid or missing."""
 pass


class FeatureComputationError(GRANITEError):
 """Raised when feature computation fails."""
 pass


class ValidationError(GRANITEError):
 """Raised when data validation fails."""
 pass