import unittest
import numpy as np
from decimal import Decimal
from portfolio.models.simulation_config import SimulationConfig
from portfolio.analyzers import PortfolioAnalyzer

class TestPortfolioAnalyzer(unittest.TestCase):
    def setUp(self):
        """Setup base per tutti i test con configurazione standard."""
        self.config = SimulationConfig(
            accumulation_years=0,
            maintenance_years=1,
            withdrawal_years=20,
            investment_amount=Decimal('100000'),
            maintenance_withdrawal=Decimal('25000'),
            withdrawal_amount=Decimal('25000'),
            mandatory_pension=Decimal('0'),
            complementary_pension=Decimal('0'),
            mean_return=Decimal('4'),
            std_dev_return=Decimal('15'),
            inflation_rate=Decimal('2'),
            black_swan_probability=Decimal('5'),
            black_swan_impact=Decimal('-40'),
            drawdown_threshold=Decimal('0.1')
        )
        self.analyzer = PortfolioAnalyzer(self.config)

    def test_basic_scenarios(self):
        """Test scenari base: costante, crescita, crash."""
        # Caso 1: Valore costante
        constant_sims = np.full((100, 21), 100000.0)
        const_metrics = self.analyzer.calculate_risk_metrics(constant_sims)
        self.assertEqual(const_metrics.max_drawdown, 0.0)
        self.assertEqual(const_metrics.success_rate, 1.0)
        
        # Caso 2: Crescita lineare 10% annuo
        growth_base = np.linspace(100000, 100000*1.1**20, 21)
        growth_sims = np.tile(growth_base, (100, 1))
        growth_metrics = self.analyzer.calculate_risk_metrics(growth_sims)
        self.assertEqual(growth_metrics.max_drawdown, 0.0)
        self.assertEqual(growth_metrics.success_rate, 1.0)
        
        # Caso 3: Crash singolo del 40%
        crash_base = np.full(21, 100000.0)
        crash_base[5:] *= 0.6
        crash_sims = np.tile(crash_base, (100, 1))
        crash_metrics = self.analyzer.calculate_risk_metrics(crash_sims)
        self.assertAlmostEqual(crash_metrics.max_drawdown, -0.4, places=2)

    def test_stability_vs_drawdown(self):
        """Test coerenza tra stabilità e drawdown."""
        # Portafoglio stabile
        t = np.linspace(0, 20, 21)
        stable_base = 100000 * (1 + 0.01 * np.sin(t))
        stable_sims = np.tile(stable_base, (100, 1))
        stable_metrics = self.analyzer.calculate_risk_metrics(stable_sims)
        
        # Portafoglio instabile
        unstable_base = 100000 * (1 + 0.5 * np.sin(t))
        unstable_sims = np.tile(unstable_base, (100, 1))
        unstable_metrics = self.analyzer.calculate_risk_metrics(unstable_sims)
        
        # Verifiche
        self.assertGreater(stable_metrics.return_stability_index,
                          unstable_metrics.return_stability_index)
        self.assertLess(abs(stable_metrics.max_drawdown),
                       abs(unstable_metrics.max_drawdown))

    def test_success_vs_failure_rates(self):
        """Test coerenza tra success rate e probabilità di fallimento."""
        # Crea set di dati con 80% successi e 20% fallimenti
        successes = np.full((80, 21), 100000.0)
        failures = np.full((20, 21), 50000.0)
        mixed_sims = np.vstack([successes, failures])
        
        metrics = self.analyzer.calculate_risk_metrics(mixed_sims)
        withdrawal = self.analyzer.analyze_withdrawal_sustainability(mixed_sims)
        
        # La somma di successi e fallimenti dovrebbe essere circa 1
        total_prob = (metrics.success_rate + 
                     max(withdrawal['failure_probability_by_year']))
        self.assertAlmostEqual(total_prob, 1.0, places=1)

    def test_maintenance_phase(self):
        """Test correttezza della fase di mantenimento."""
        # Simulazioni complete per la fase di mantenimento
        maint_years = self.config.maintenance_years
        maint_sims = np.full((100, maint_years), 100000.0)
        
        withdrawal = self.analyzer.analyze_withdrawal_sustainability(maint_sims)
        maint_analysis = withdrawal['maintenance_phase_analysis']
        
        # La vita mediana dovrebbe essere uguale agli anni di mantenimento
        self.assertEqual(maint_analysis['median_portfolio_life'], maint_years)
        
        # Con valore costante, non dovrebbe esserci fallimento
        self.assertEqual(maint_analysis['total_failure_probability'], 0.0)

    def test_stress_scenarios(self):
        """Test realismo degli scenari di stress."""
        from portfolio.stress_testing import StressTestManager
        from portfolio.models.position import ETFPosition
        from decimal import Decimal
        
        # Creiamo un'istanza di StressTestManager con la stessa config
        stress_manager = StressTestManager(self.config)
        
        # Crea le posizioni di test
        positions = {
            'ETF1': ETFPosition(
                shares=Decimal('100'),
                price=Decimal('1000'),
                avg_price=Decimal('1000')
            )
        }
        allocations = [1.0]
        
        # Esegui lo stress test
        stress_results = stress_manager.perform_stress_test(positions, allocations)
        
        # Verifiche di realismo con assertGreaterEqual
        self.assertGreaterEqual(stress_results['prolonged_bear']['portfolio_impact'], -1.0)
        self.assertGreaterEqual(stress_results['combined_stress']['portfolio_impact'], -1.0)
        self.assertGreater(
            stress_results['market_crash']['portfolio_impact'],
            stress_results['combined_stress']['portfolio_impact']
        )

    def test_recovery_patterns(self):
        """
        Test del tempo di recupero dopo un drawdown.
        Scenario:
        - Drawdown del 40% dal 5° al 10° anno
        - Recupero lineare fino al valore originale nei successivi 11 anni
        
        Il recovery time dovrebbe essere circa 7 anni considerando:
        - Inizio drawdown: anno 5
        - Recupero all'85% del peak: circa anno 12
        """
        base = np.full(21, 100000.0)
        base[5:10] *= 0.6  # -40% dal 5° al 10° anno
        base[10:] = np.linspace(base[9], 100000, len(base[10:]))  # Recupero lineare
        recovery_sims = np.tile(base, (100, 1))
        
        metrics = self.analyzer.calculate_risk_metrics(recovery_sims)
        
        # Verifiche
        self.assertAlmostEqual(metrics.max_drawdown, -0.4, places=2)  # Verifica drawdown del 40%
        self.assertGreater(metrics.recovery_time, 5)  # Minimo tempo per recupero 85%
        self.assertLess(metrics.recovery_time, 8)     # Massimo tempo per recupero 85%

    def test_underwater_periods(self):
        """Test identificazione periodi underwater."""
        # Crea scenario con due periodi underwater distinti
        base = np.full(21, 100000.0)
        base[5:7] *= 0.85  # -15% per 2 anni
        base[7:10] = base[4]  # Recupero
        base[12:15] *= 0.8  # -20% per 3 anni
        base[15:] = base[11]  # Recupero
        underwater_sims = np.tile(base, (100, 1))
        
        metrics = self.analyzer.calculate_risk_metrics(underwater_sims)
        
        # Dovremmo vedere esattamente 2 periodi underwater
        self.assertEqual(metrics.underwater_periods, 2)

    def test_risk_metrics_coherence(self):
        """Test coerenza generale delle metriche di rischio."""
        # Scenario base con moderata volatilità
        t = np.linspace(0, 20, 21)
        base = 100000 * (1 + 0.05*t + 0.1*np.sin(t))
        test_sims = np.tile(base, (100, 1))
        
        metrics = self.analyzer.calculate_risk_metrics(test_sims)
        
        # Verifiche di coerenza
        self.assertGreaterEqual(metrics.risk_score, 0)
        self.assertLessEqual(metrics.risk_score, 100)
        self.assertGreaterEqual(metrics.success_rate, 0)
        self.assertLessEqual(metrics.success_rate, 1)
        self.assertGreaterEqual(metrics.stress_resilience_score, 0)
        self.assertLessEqual(metrics.stress_resilience_score, 100)

if __name__ == '__main__':
    unittest.main(verbosity=2)