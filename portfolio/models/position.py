from dataclasses import dataclass
from decimal import Decimal

@dataclass
class ETFPosition:
    """Represents an ETF position in the portfolio."""
    shares: Decimal
    price: Decimal
    avg_price: Decimal

    def __post_init__(self):
        self.shares = Decimal(str(self.shares))
        self.price = Decimal(str(self.price))
        self.avg_price = Decimal(str(self.avg_price))
        self._validate()

    def _validate(self):
        if any(v < 0 for v in [self.shares, self.price, self.avg_price]):
            raise ValueError("ETF position values must not be negative")

    @property
    def value(self) -> Decimal:
        """Calculate the current value of the position."""
        return self.shares * self.price