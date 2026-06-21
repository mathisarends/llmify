class LLMifyError(Exception):
    """Base exception for all llmify errors."""


class RetryableError(LLMifyError):
    """Raised when a request failed but may succeed if retried (e.g. 500/503, timeouts)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(RetryableError):
    """Raised when the provider returns HTTP 429 Too Many Requests."""

    def __init__(
        self, message: str = "Rate limit exceeded", retry_after: float | None = None
    ):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class OutOfCreditsError(LLMifyError):
    """Raised when the account has insufficient credits / quota to complete the request."""

    def __init__(self, message: str = "Out of credits"):
        super().__init__(message)


class ContextLengthExceededError(LLMifyError):
    """Raised when the input exceeds the model's maximum context length."""

    def __init__(self, message: str = "Context length exceeded"):
        super().__init__(message)


class AuthenticationError(LLMifyError):
    """Raised when the API key or credentials are invalid or missing."""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message)
