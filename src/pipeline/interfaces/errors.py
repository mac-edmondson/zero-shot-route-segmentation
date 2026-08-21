class PipelineError(RuntimeError):
    """Base class for automatic hold-detection failures."""


class InvalidImageError(PipelineError):
    """Raised when a batch has an invalid image."""


class BatchAlignmentError(PipelineError):
    """Raised when image and detection batches differ in length."""
