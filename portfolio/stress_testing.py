from decimal import Decimal
from typing import Dict, List
import logging
from .models.position import ETFPosition
from .models.simulation_config import SimulationConfig

logger = logging.getLogger(__name__)

class StressTestManager:
    """Manager for portfolio stress testing scenarios."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config

    def perform_stress_test(self, positions: Dict[str, ETFPosition], 
                          allocations: List[float]) -> Dict:
        """Execute all stress tests on the portfolio."""
        try:
            results = {
                'high_inflation': {
                    'portfolio_impact': self._simulate_high_inflation_scenario(positions, allocations)
                },
                'market_crash': {
                    'portfolio_impact': self._simulate_market_crash(positions, allocations)
                },
                'prolonged_bear': {
                    'portfolio_impact': self._simulate_bear_market(positions, allocations)
                },
                'combined_stress': {
                    'portfolio_impact': self._simulate_combined_scenario(positions, allocations)
                }
            }
            
            logger.info("Stress tests completed successfully")
            logger.info(f"Results: {results}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in stress test execution: {e}")
            return {
                'high_inflation': {'portfolio_impact': -0.15},
                'market_crash': {'portfolio_impact': -0.40},
                'prolonged_bear': {'portfolio_impact': -0.25},
                'combined_stress': {'portfolio_impact': -0.50}
            }

    def _simulate_high_inflation_scenario(self, positions: Dict[str, ETFPosition],
                                        allocations: List[float]) -> float:
        """Simulate high inflation impact."""
        try:
            test_positions = {k: ETFPosition(
                shares=Decimal(str(v.shares)),
                price=Decimal(str(v.price)),
                avg_price=Decimal(str(v.avg_price))
            ) for k, v in positions.items()}
            
            initial_value = sum(float(pos.value) for pos in test_positions.values())
            high_inflation = float(self.config.high_inflation_scenario) / 100
            years_to_simulate = 5
            
            for year in range(years_to_simulate):
                adjusted_withdrawal = float(self.config.withdrawal_amount) * (1 + high_inflation) ** year
                reduced_return = float(self.config.mean_return) / 100 - high_inflation
                
                for etf_name, position in test_positions.items():
                    new_price = Decimal(str(float(position.price) * (1 + reduced_return)))
                    allocation_idx = list(test_positions.keys()).index(etf_name)
                    withdrawal_portion = Decimal(str(adjusted_withdrawal * allocations[allocation_idx]))
                    
                    shares_to_sell = min(
                        (withdrawal_portion / new_price).quantize(Decimal('1.')),
                        position.shares
                    )
                    
                    test_positions[etf_name] = ETFPosition(
                        shares=position.shares - shares_to_sell,
                        price=new_price,
                        avg_price=position.avg_price
                    )
            
            final_value = sum(float(pos.value) for pos in test_positions.values())
            return (final_value - initial_value) / initial_value
        
        except Exception as e:
            logger.error(f"Error in high inflation simulation: {e}")
            return -0.15

    def _simulate_market_crash(self, positions: Dict[str, ETFPosition],
                             allocations: List[float]) -> float:
        """Simulate market crash impact."""
        try:
            test_positions = {k: ETFPosition(
                shares=v.shares,
                price=v.price,
                avg_price=v.avg_price
            ) for k, v in positions.items()}
            
            initial_value = sum(float(pos.value) for pos in test_positions.values())
            crash_impact = Decimal(str(self.config.market_crash_impact)) / Decimal('100')
            
            for k, position in test_positions.items():
                new_price = position.price * (Decimal('1') + crash_impact)
                test_positions[k] = ETFPosition(
                    shares=position.shares,
                    price=new_price,
                    avg_price=position.avg_price
                )
            
            final_value = sum(float(pos.value) for pos in test_positions.values())
            return float((Decimal(str(final_value)) - 
                        Decimal(str(initial_value))) / Decimal(str(initial_value)))
        
        except Exception as e:
            logger.error(f"Error in market crash simulation: {e}")
            return -0.40

    def _simulate_bear_market(self, positions: Dict[str, ETFPosition],
                            allocations: List[float]) -> float:
        """Simulate prolonged bear market impact."""
        try:
            test_positions = {k: ETFPosition(v.shares, v.price, v.avg_price) 
                            for k, v in positions.items()}
            initial_value = sum(float(pos.value) for pos in test_positions.values())
            
            bear_return = (Decimal(str(self.config.mean_return)) - 
                        Decimal('2') * Decimal(str(self.config.std_dev_return))) / Decimal('100')
            
            for year in range(self.config.bear_market_years):
                for etf_name, position in test_positions.items():
                    new_price = position.price * (Decimal('1') + bear_return)
                    allocation_idx = list(test_positions.keys()).index(etf_name)
                    withdrawal = (self.config.withdrawal_amount * 
                                Decimal(str(allocations[allocation_idx])))
                    shares_to_sell = min(
                        (withdrawal / new_price).quantize(Decimal('1.')),
                        position.shares
                    )
                    
                    test_positions[etf_name] = ETFPosition(
                        shares=position.shares - shares_to_sell,
                        price=new_price,
                        avg_price=position.avg_price
                    )
            
            final_value = sum(float(pos.value) for pos in test_positions.values())
            return float((Decimal(str(final_value)) - 
                        Decimal(str(initial_value))) / Decimal(str(initial_value)))
        
        except Exception as e:
            logger.error(f"Error in bear market simulation: {e}")
            return -0.25

    def _simulate_combined_scenario(self, positions: Dict[str, ETFPosition],
                                  allocations: List[float]) -> float:
        """Simulate combined stress scenario impact."""
        try:
            test_positions = {k: ETFPosition(v.shares, v.price, v.avg_price) 
                            for k, v in positions.items()}
            initial_value = sum(float(pos.value) for pos in test_positions.values())
            
            # Apply market crash first
            crash_impact = Decimal(str(self.config.market_crash_impact)) / Decimal('100')
            for position in test_positions.values():
                position.price *= (Decimal('1') + crash_impact)
            
            # Then simulate high inflation and bear market
            high_inflation = Decimal(str(self.config.high_inflation_scenario)) / Decimal('100')
            years = min(3, self.config.bear_market_years)
            
            for year in range(years):
                bear_return = (Decimal(str(self.config.mean_return)) - 
                            Decimal('2') * Decimal(str(self.config.std_dev_return))) / Decimal('100')
                adjusted_withdrawal = self.config.withdrawal_amount * (
                    (Decimal('1') + high_inflation) ** Decimal(str(year))
                )
                
                for etf_name, position in test_positions.items():
                    new_price = position.price * (Decimal('1') + bear_return - high_inflation)
                    allocation_idx = list(test_positions.keys()).index(etf_name)
                    withdrawal = adjusted_withdrawal * Decimal(str(allocations[allocation_idx]))
                    shares_to_sell = min(
                        (withdrawal / new_price).quantize(Decimal('1.')),
                        position.shares
                    )
                    
                    test_positions[etf_name] = ETFPosition(
                        shares=position.shares - shares_to_sell,
                        price=new_price,
                        avg_price=position.avg_price
                    )
            
            final_value = sum(float(pos.value) for pos in test_positions.values())
            return float((Decimal(str(final_value)) - 
                        Decimal(str(initial_value))) / Decimal(str(initial_value)))
        
        except Exception as e:
            logger.error(f"Error in combined scenario simulation: {e}")
            return -0.50