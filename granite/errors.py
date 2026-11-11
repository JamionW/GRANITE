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

class GraphRandomizationError(GRANITEError):
    """Raised when a degree-preserving swap leaves a tract's edge list unchanged.

    records: list of per-tract swap records (dicts) whose randomized flag is False
    and whose tract is not allowlisted.
    """

    def __init__(self, records):
        self.records = list(records)
        parts = [f"{r.get('tract_fips')} (seed={r.get('graph_draw_seed')}, "
                 f"outcome={r.get('swap_outcome')})" for r in self.records]
        message = (
            f"randomized graph identical to input for {len(self.records)} tract(s): "
            + ", ".join(parts)
            + ". pass --allow-unswapped (config key allow_unswapped) to proceed on the "
              "unswapped graph with randomized=False"
        )
        super().__init__(message)
