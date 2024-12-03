from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict
from .execution_config import ExecutionConfig

@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    accumulation_years: int
    maintenance_years: int
    withdrawal_years: int
    investment_amount: Decimal
    maintenance_withdrawal: Decimal
    withdrawal_amount: Decimal
    mandatory_pension: Decimal
    complementary_pension: Decimal
    mean_return: Decimal
    std_dev_return: Decimal
    inflation_rate: Decimal
    black_swan_probability: Decimal
    black_swan_impact: Decimal
    batch_size: int = field(default=1000)
    risk_free_rate: Decimal = field(default=Decimal('0.02'))
    confidence_level: Decimal = field(default=Decimal('0.95'))
    drawdown_threshold: Decimal = field(default=Decimal('0.10'))
    high_inflation_scenario: Decimal = field(default=Decimal('8.0'))
    market_crash_impact: Decimal = field(default=Decimal('-40.0'))
    bear_market_years: int = field(default=5)
    combined_stress_impact: Decimal = field(default=Decimal('-50.0'))
    plot_height: int = field(default=800)
    plot_width: int = field(default=1200)
    distribution_bins: int = field(default=50)
    execution_settings: ExecutionConfig = field(default_factory=ExecutionConfig)

    def __post_init__(self):
        self._convert_to_decimal()
        self._validate()

    def _convert_to_decimal(self):
        decimal_fields = [
            'investment_amount', 'maintenance_withdrawal', 'withdrawal_amount',
            'mandatory_pension', 'complementary_pension', 'mean_return',
            'std_dev_return', 'inflation_rate', 'black_swan_probability',
            'black_swan_impact', 'risk_free_rate', 'confidence_level',
            'drawdown_threshold', 'high_inflation_scenario', 'market_crash_impact',
            'combined_stress_impact'
        ]
        
        for field in decimal_fields:
            current_value = getattr(self, field)
            if not isinstance(current_value, Decimal):
                setattr(self, field, Decimal(str(current_value)))

    def _validate(self):
        self._validate_years()
        self._validate_monetary_amounts()
        self._validate_rates()
        self._validate_probabilities()
        self._validate_plot_params()

    def _validate_years(self):
        if not all(isinstance(x, int) for x in [self.accumulation_years, 
                                            self.maintenance_years, 
                                            self.withdrawal_years]):
            raise ValueError("Years must be integers")
        if any(x < 0 for x in [self.accumulation_years, 
                            self.maintenance_years, 
                            self.withdrawal_years]):
            raise ValueError("Years cannot be negative")
        if self.accumulation_years == 0 and self.withdrawal_years == 0:
            raise ValueError("At least one phase (accumulation or withdrawal) must have duration > 0")

    def _validate_monetary_amounts(self):
        monetary_values = [
            self.investment_amount,
            self.withdrawal_amount,
            self.mandatory_pension,
            self.complementary_pension
        ]
        if any(v < 0 for v in monetary_values):
            raise ValueError("Monetary amounts cannot be negative")

    def _validate_rates(self):
        if self.std_dev_return <= 0:
            raise ValueError("Standard deviation must be positive")
        if self.inflation_rate < 0:
            raise ValueError("Inflation rate cannot be negative")
        if self.risk_free_rate < 0:
            raise ValueError("Risk-free rate cannot be negative")

    def _validate_probabilities(self):
        if not 0 <= float(self.black_swan_probability) <= 100:
            raise ValueError("Black swan probability must be between 0 and 100")
        if not 0 < float(self.confidence_level) < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if not 0 < float(self.drawdown_threshold) < 1:
            raise ValueError("Drawdown threshold must be between 0 and 1")

    def _validate_plot_params(self):
        plot_params = [self.plot_height, self.plot_width, self.distribution_bins]
        if any(not isinstance(param, int) for param in plot_params):
            raise ValueError("Plot parameters must be integers")
        if any(param <= 0 for param in plot_params):
            raise ValueError("Plot parameters must be positive")
        
    def get_optimal_chunk_size(self, total_simulations: int) -> int:
        """Calcola la dimensione ottimale del chunk basata sul numero totale di simulazioni."""
        if not self.execution_settings.parallel_enabled:
            return total_simulations
            
        base_chunk_size = self.execution_settings.chunk_size
        num_cores = self.execution_settings.max_cores
        
        # Assicurati che ci siano almeno chunk_size/2 simulazioni per core
        min_chunk_size = max(1, base_chunk_size // 2)
        
        # Calcola il numero ottimale di chunk
        optimal_chunks = min(num_cores * 4, total_simulations // min_chunk_size)
        if optimal_chunks <= 0:
            return total_simulations
            
        return max(min_chunk_size, total_simulations // optimal_chunks)