from dataclasses import dataclass, field
from typing import Union
from multiprocessing import cpu_count

@dataclass
class ExecutionConfig:
    """Configuration for simulation execution."""
    debug_mode: bool = field(default=False)
    parallel_enabled: bool = field(default=True)
    max_cores: Union[str, int] = field(default="auto")
    chunk_size: int = field(default=100)
    memory_limit_mb: int = field(default=1024)

    def __post_init__(self):
        if isinstance(self.max_cores, str) and self.max_cores.lower() == "auto":
            self.max_cores = max(1, cpu_count() - 1)
        else:
            self.max_cores = int(self.max_cores)
        self._validate()
    
    def _validate(self):
        if self.max_cores <= 0:
            raise ValueError("max_cores must be greater than 0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be greater than 0")