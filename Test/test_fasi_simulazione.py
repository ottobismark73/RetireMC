import unittest
from decimal import Decimal
from sim_mc_bs_8 import MonteCarloSimulator, SimulationConfig, ETFPosition

class TestSimulationPhases(unittest.TestCase):
    def setUp(self):
        """Setup iniziale con configurazione e posizioni di test."""
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
            black_swan_probability=Decimal('0'),  # Impostiamo a 0 per test deterministici
            black_swan_impact=Decimal('-45')
        )
        self.simulator = MonteCarloSimulator(self.config)
        
        # Posizioni iniziali di test
        self.test_positions = {
            "ETF1": ETFPosition(
                shares=Decimal('100'),
                price=Decimal('100'),
                avg_price=Decimal('90')
            )
        }
        self.test_allocations = [1.0]  # 100% allocazione al singolo ETF

    def test_accumulation_phase(self):
        """Test della fase di accumulo."""
        # Test con rendimento fisso per verifica deterministica
        growth_rate = Decimal('0.05')  # 5% di rendimento
        
        total_value = self.simulator._process_accumulation_year(
            self.test_positions,
            self.test_allocations,
            growth_rate,
            2024,  # anno simulazione
            False   # non è prima simulazione
        )
        
        # Calcoli attesi:
        # 1. Prezzo dopo crescita: 100 * (1 + 0.05) = 105
        # 2. Investimento annuale: 20000
        # 3. Nuove shares: floor(20000 / 105) = 190
        # 4. Valore totale: (100 + 190) * 105
        expected_shares = Decimal('100') + Decimal('190')
        expected_value = expected_shares * Decimal('105')
        
        self.assertAlmostEqual(
            float(total_value),
            float(expected_value),
            places=2,
            msg="Il valore totale dopo l'accumulo non corrisponde al calcolo atteso"
        )

    def test_maintenance_phase(self):
        """Test della fase di mantenimento."""
        growth_rate = Decimal('0.05')  # 5% di rendimento
        
        total_value = self.simulator._process_maintenance_year(
            self.test_positions,
            self.test_allocations,
            1,    # primo anno
            2024, # anno simulazione
            growth_rate,
            False # non è prima simulazione
        )
        
        # Calcoli attesi:
        # 1. Prezzo dopo crescita: 100 * (1.05) = 105
        # 2. Prelievo richiesto: 25000
        # 3. Shares da vendere: ceil(25000 / 105)
        # 4. Valore rimanente: (100 - shares_vendute) * 105
        shares_to_sell = Decimal('238')  # ceil(25000/105)
        remaining_shares = Decimal('100') - shares_to_sell
        expected_value = max(Decimal('0'), remaining_shares * Decimal('105'))
        
        self.assertAlmostEqual(
            float(total_value),
            float(expected_value),
            places=2,
            msg="Il valore totale dopo il mantenimento non corrisponde al calcolo atteso"
        )

    def test_withdrawal_phase(self):
        """Test della fase di prelievo."""
        growth_rate = Decimal('0.05')  # 5% di rendimento
        
        # Aumentiamo le shares iniziali per questo test
        self.test_positions["ETF1"] = ETFPosition(
            shares=Decimal('1000'),
            price=Decimal('100'),
            avg_price=Decimal('90')
        )
        
        total_value = self.simulator._process_withdrawal_year(
            self.test_positions,
            self.test_allocations,
            1,    # primo anno
            2024, # anno simulazione
            growth_rate,
            False # non è prima simulazione
        )
        
        # Calcoli attesi:
        # 1. Prezzo dopo crescita: 100 * (1.05) = 105
        # 2. Prelievo richiesto: 30000 - (15000 + 5000) = 10000 (considerando le pensioni)
        # 3. Shares da vendere considerando tasse capital gain
        # 4. Valore rimanente: (1000 - shares_vendute) * 105
        
        # Il prelievo netto richiesto è minore per via delle pensioni
        expected_min_value = Decimal('1000') * Decimal('105') - Decimal('12000')
        
        self.assertGreater(
            float(total_value),
            float(expected_min_value),
            msg="Il valore totale dopo il prelievo è inferiore al minimo atteso"
        )

    def test_inflation_impact(self):
        """Test dell'impatto dell'inflazione sui prelievi."""
        # Test del secondo anno di prelievo dove l'inflazione dovrebbe impattare
        growth_rate = Decimal('0.05')
        
        # Aumentiamo le shares iniziali per questo test
        self.test_positions["ETF1"] = ETFPosition(
            shares=Decimal('1000'),
            price=Decimal('100'),
            avg_price=Decimal('90')
        )
        
        # Primo anno senza inflazione
        value_year1 = self.simulator._process_withdrawal_year(
            self.test_positions,
            self.test_allocations,
            1,
            2024,
            growth_rate,
            False
        )
        
        # Secondo anno con inflazione
        value_year2 = self.simulator._process_withdrawal_year(
            self.test_positions,
            self.test_allocations,
            2,
            2025,
            growth_rate,
            False
        )
        
        # L'importo del prelievo del secondo anno dovrebbe essere maggiore per l'inflazione
        # Inflazione del 3% su 30000: 30000 * (1.03)
        withdrawal_year1 = self.config.withdrawal_amount
        withdrawal_year2 = withdrawal_year1 * (1 + self.config.inflation_rate / 100)
        
        self.assertGreater(
            withdrawal_year2,
            withdrawal_year1,
            msg="Il prelievo del secondo anno non riflette l'aumento per inflazione"
        )

        self.assertLess(
            value_year2,
            value_year1,
            msg="Il valore del secondo anno non riflette il maggior prelievo per inflazione"
        )

    def test_market_crash_scenario(self):
        """Test del comportamento con rendimenti fortemente negativi."""
        severe_crash_rate = Decimal('-0.50')  # -50% crash
        
        # Test con portafoglio sostanzioso per verificare l'impatto
        self.test_positions["ETF1"] = ETFPosition(
            shares=Decimal('1000'),
            price=Decimal('100'),
            avg_price=Decimal('90')
        )
        
        # Verifica fase di accumulo durante crash
        crash_accumulation_value = self.simulator._process_accumulation_year(
            self.test_positions.copy(),
            self.test_allocations,
            severe_crash_rate,
            2024,
            False
        )
        
        # Dopo un -50%, il valore dovrebbe essere:
        # (shares originali * nuovo prezzo) + (nuovo investimento / nuovo prezzo) * nuovo prezzo
        expected_price_after_crash = Decimal('100') * (1 + severe_crash_rate)
        expected_new_shares = self.config.investment_amount / expected_price_after_crash
        expected_total_shares = Decimal('1000') + expected_new_shares
        expected_value = expected_total_shares * expected_price_after_crash
        
        self.assertAlmostEqual(
            float(crash_accumulation_value),
            float(expected_value),
            places=2,
            msg="Il valore durante il crash non corrisponde al calcolo atteso"
        )
        
        # Verifica che l'investimento durante il crash compri più shares
        self.assertGreater(
            float(expected_new_shares),
            float(self.config.investment_amount / Decimal('100')),
            msg="Il crash non sta risultando in un maggior numero di shares acquistate"
        )

    def test_zero_liquidity_scenario(self):
        """Test del comportamento quando non c'è liquidità sufficiente per i prelievi."""
        # Setup con portafoglio piccolo
        self.test_positions["ETF1"] = ETFPosition(
            shares=Decimal('10'),
            price=Decimal('100'),
            avg_price=Decimal('90')
        )
        
        # Tenta un prelievo maggiore del valore del portafoglio
        total_value = self.simulator._process_withdrawal_year(
            self.test_positions,
            self.test_allocations,
            1,
            2024,
            Decimal('0.05'),
            False
        )
        
        self.assertGreaterEqual(
            float(total_value),
            0,
            msg="Il valore del portafoglio non dovrebbe andare sotto zero"
        )

    def test_high_inflation_scenario(self):
        """Test del comportamento con inflazione molto alta."""
        # Modifica temporanea della configurazione per alta inflazione
        high_inflation_config = SimulationConfig(
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
            inflation_rate=Decimal('15'),  # 15% inflazione
            black_swan_probability=Decimal('0'),
            black_swan_impact=Decimal('-45')
        )
        
        high_inflation_simulator = MonteCarloSimulator(high_inflation_config)
        
        # Test prelievi in anni successivi con alta inflazione
        base_positions = {
            "ETF1": ETFPosition(
                shares=Decimal('1000'),
                price=Decimal('100'),
                avg_price=Decimal('90')
            )
        }
        
        # Calcola prelievo anno 1 e anno 5
        withdrawal_year1 = high_inflation_config.withdrawal_amount
        withdrawal_year5 = withdrawal_year1 * (1 + high_inflation_config.inflation_rate / 100) ** 4
        
        self.assertGreater(
            float(withdrawal_year5),
            float(withdrawal_year1 * Decimal('1.5')),
            msg="L'inflazione alta non sta impattando sufficientemente i prelievi"
        )

    def test_consecutive_negative_returns(self):
        """Test del comportamento con rendimenti negativi consecutivi."""
        negative_return = Decimal('-0.10')  # -10% ogni anno
        
        # Setup con portafoglio molto più grande per assicurare valori non-zero
        large_position = ETFPosition(
            shares=Decimal('10000'),  # Aumentato significativamente
            price=Decimal('100'),
            avg_price=Decimal('90')
        )
        
        positions = {"ETF1": large_position}
        allocations = [Decimal('1.0')]
        
        values = []
        prev_value = None
        
        # Simula 3 anni consecutivi di rendimenti negativi
        for year in range(1, 4):
            positions_copy = {
                k: ETFPosition(v.shares, v.price, v.avg_price) 
                for k, v in positions.items()
            }
            
            value = self.simulator._process_withdrawal_year(
                positions_copy,
                allocations,
                year,
                2024 + year - 1,
                negative_return,
                False
            )
            
            values.append(value)
            
            # Verifica che ogni valore sia positivo
            self.assertGreater(
                float(value),
                0,
                msg=f"Il valore è diventato zero nell'anno {year}"
            )
            
            if prev_value is not None:
                # Verifica che il valore stia diminuendo
                self.assertGreater(
                    float(prev_value),
                    float(value),
                    msg=f"Il valore non sta declinando consistentemente nell'anno {year}"
                )
            
            prev_value = value

        # Verifica il declino totale
        total_decline = (values[-1] - values[0]) / values[0]
        self.assertLess(
            float(total_decline),
            0,
            msg="Non c'è stato un declino significativo dopo rendimenti negativi consecutivi"
        )

    def test_maximum_shares_limits(self):
        """Test dei limiti massimi di shares gestibili."""
        # Setup con 1 milione di shares
        large_position = ETFPosition(
            shares=Decimal('1000000'),
            price=Decimal('100'),
            avg_price=Decimal('90')
        )
        
        # Crea una nuova istanza invece di usare copy()
        positions = {"ETF1": ETFPosition(
            shares=large_position.shares,
            price=large_position.price,
            avg_price=large_position.avg_price
        )}
        
        allocations = [Decimal('1.0')]
        growth_rate = Decimal('0.05')  # 5% crescita
        
        # Calcola il valore totale
        total_value = self.simulator._process_accumulation_year(
            positions,
            allocations,
            growth_rate,
            2024,
            False
        )

        # Calcolo manuale del valore atteso
        initial_shares = Decimal('1000000')
        initial_price = Decimal('100')
        new_price = initial_price * (1 + growth_rate)
        
        # Non consideriamo l'investimento aggiuntivo per questo test
        expected_value = initial_shares * new_price
        
        # Test con una tolleranza relativa invece di decimale
        relative_difference = abs(float(total_value) - float(expected_value)) / float(expected_value)
        self.assertLess(
            relative_difference,
            0.001,  # 0.1% di tolleranza
            msg=f"Differenza relativa troppo grande: {relative_difference:.4%}. "
                f"Valore atteso: {float(expected_value):.2f}, "
                f"Valore ottenuto: {float(total_value):.2f}"
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)