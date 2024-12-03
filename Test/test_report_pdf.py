import unittest
from decimal import Decimal
from sim_mc_bs_8 import MonteCarloSimulator, SimulationConfig, ETFPosition
from multiprocessing import cpu_count
import numpy as np

class TestParallelExecution(unittest.TestCase):
    def setUp(self):
        """Setup con configurazione base per test di parallelizzazione."""
        self.config = SimulationConfig(
            accumulation_years=10,
            maintenance_years=1,
            withdrawal_years=20,
            investment_amount=Decimal('20000'),
            maintenance_withdrawal=Decimal('25000'),
            withdrawal_amount=Decimal('30000'),
            mandatory_pension=Decimal('15000'),
            complementary_pension=Decimal('5000'),
            mean_return=Decimal('5.86'),
            std_dev_return=Decimal('16.91'),
            inflation_rate=Decimal('3'),
            black_swan_probability=Decimal('2.5'),
            black_swan_impact=Decimal('-45'),
            execution_settings={
                'debug_mode': False,
                'parallel_enabled': True,
                'max_cores': "auto",
                'chunk_size': 100,
                'memory_limit_mb': 1024
            }
        )
        self.simulator = MonteCarloSimulator(self.config)
        self.test_positions = {
            "ETF1": ETFPosition(
                shares=Decimal('100'),
                price=Decimal('100'),
                avg_price=Decimal('90')
            )
        }
        self.test_allocations = [1.0]

    def test_parallel_execution_enabled(self):
        """Test che la parallelizzazione sia effettivamente abilitata."""
        self.assertTrue(self.simulator.config.execution_settings.parallel_enabled)
        self.assertEqual(self.simulator.config.execution_settings.max_cores, cpu_count() - 1)

    def test_batch_simulation_output(self):
        """Test che le simulazioni batch producano il numero corretto di risultati."""
        num_simulations = 100
        results = self.simulator.run_batch_simulations(
            self.test_positions, 
            self.test_allocations, 
            num_simulations
        )
        self.assertEqual(len(results), num_simulations)

    def test_simulation_consistency(self):
        """Test che i risultati delle simulazioni siano consistenti."""
        num_simulations = 50
        results = self.simulator.run_batch_simulations(
            self.test_positions, 
            self.test_allocations, 
            num_simulations
        )
        
        # Verifica che tutti i risultati abbiano la stessa lunghezza
        expected_length = (self.config.accumulation_years + 
                         self.config.maintenance_years + 
                         self.config.withdrawal_years)
        
        for sim_result in results:
            self.assertEqual(
                len(sim_result), 
                expected_length,
                "Tutte le simulazioni dovrebbero avere la stessa lunghezza"
            )

    def test_parallel_performance(self):
        """Test delle performance della parallelizzazione."""
        import time
        
        # Test con parallelizzazione
        self.simulator.config.execution_settings.parallel_enabled = True
        start_time = time.time()
        parallel_results = self.simulator.run_batch_simulations(
            self.test_positions,
            self.test_allocations,
            100
        )
        parallel_time = time.time() - start_time
        
        # Test senza parallelizzazione
        self.simulator.config.execution_settings.parallel_enabled = False
        start_time = time.time()
        sequential_results = self.simulator.run_batch_simulations(
            self.test_positions,
            self.test_allocations,
            100
        )
        sequential_time = time.time() - start_time
        
        # La versione parallela dovrebbe essere più veloce con un numero sufficiente di simulazioni
        print(f"\nTempo parallelo: {parallel_time:.2f}s")
        print(f"Tempo sequenziale: {sequential_time:.2f}s")
        print(f"Speedup: {sequential_time/parallel_time:.2f}x")

if __name__ == '__main__':
    unittest.main(verbosity=2)