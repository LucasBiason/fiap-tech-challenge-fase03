"""
Simple Configuration for Data Processing
"""


class Config:
    """Simple configuration class for data processing."""

    def __init__(self, **kwargs):
        """
        Initialize configuration with custom parameters.

        Args:
            **kwargs: Any configuration parameters
        """
        # Set defaults
        self.required_fields = ['title', 'content']
        self.min_title_length = 5
        self.min_content_length = 10
        self.chunk_size = 300
        self.generate_synthetic_content = True

        # Override with any provided parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        """Convert to dictionary for easy passing to other classes."""
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_')}

    def update(self, **kwargs):
        """Update configuration with new parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        """String representation of configuration."""
        items = [f"{k}={v}" for k, v in self.to_dict().items()]
        return f"Config({', '.join(items)})"
