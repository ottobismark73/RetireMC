import unittest
from decimal import Decimal
from sim_mc_bs_8 import MonteCarloSimulator, SimulationConfig, ETFPosition

class TestFinancialCalculations(unittest.TestCase):
    def setUp(self):
        """Setup iniziale per i test con una configurazione base."""
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
            black_swan_impact=Decimal('-45')
        )
        self.simulator = MonteCarloSimulator(self.config)

    def test_capital_gains_tax_calculation(self):
        """Test del calcolo delle tasse sui capital gain."""
        # Caso 1: Guadagno in capitale
        avg_price = Decimal('100')
        current_price = Decimal('150')
        target_net = Decimal('1000')
        
        gross_amount, tax = self.simulator.calculate_capital_gains_tax(
            avg_price, current_price, target_net
        )
        
        # Verifica che il netto dopo le tasse sia uguale al target
        self.assertAlmostEqual(
            float(gross_amount - tax), 
            float(target_net),
            places=2,
            msg="Il netto dopo le tasse dovrebbe essere uguale al target"
        )
        
        # Verifica che la tassa sia il 26% del capital gain
        expected_tax = (gross_amount - (gross_amount * avg_price / current_price)) * Decimal('0.26')
        self.assertAlmostEqual(
            float(tax),
            float(expected_tax),
            places=2,
            msg="La tassa dovrebbe essere il 26% del capital gain"
        )

        # Caso 2: Perdita in capitale (no tasse)
        avg_price = Decimal('150')
        current_price = Decimal('100')
        target_net = Decimal('1000')
        
        gross_amount, tax = self.simulator.calculate_capital_gains_tax(
            avg_price, current_price, target_net
        )
        
        self.assertEqual(
            gross_amount,
            target_net,
            "Con una perdita, l'importo lordo dovrebbe essere uguale al netto"
        )
        self.assertEqual(
            tax,
            Decimal('0'),
            "Con una perdita, non dovrebbero esserci tasse"
        )

    def test_capital_gains_tax_calculation_pmc(self):
        """Test approfondito del calcolo delle tasse sui capital gain usando il metodo PMC."""
        
        # Scenario 1: Acquisti multipli con PMC
        position = ETFPosition(
            shares=Decimal('100'),
            price=Decimal('150'),
            avg_price=Decimal('100')  # PMC dopo acquisti multipli
        )
        
        # Vendita con guadagno usando PMC
        target_net = Decimal('1000')
        gross_amount, tax = self.simulator.calculate_capital_gains_tax(
            position.avg_price,
            position.price,
            target_net
        )
        
        # Verifica che il capital gain sia calcolato sulla differenza tra prezzo attuale e PMC
        expected_gain_per_share = position.price - position.avg_price
        expected_tax_rate = Decimal('0.26')
        
        # Calcola quante azioni servono per ottenere il target netto
        shares_needed = target_net / (position.price - (expected_gain_per_share * expected_tax_rate))
        expected_tax = shares_needed * expected_gain_per_share * expected_tax_rate
        
        self.assertAlmostEqual(
            float(tax),
            float(expected_tax),
            places=2,
            msg="La tassa dovrebbe essere calcolata correttamente usando il PMC"
        )
        
        # Scenario 2: Vendita parziale con PMC differente
        initial_pmc = Decimal('80')
        current_price = Decimal('120')
        target_net = Decimal('2000')
        
        gross_amount, tax = self.simulator.calculate_capital_gains_tax(
            initial_pmc,
            current_price,
            target_net
        )
        
        # Verifica che il capital gain consideri il PMC corretto
        capital_gain_per_share = current_price - initial_pmc
        shares_to_sell = target_net / (current_price - (capital_gain_per_share * expected_tax_rate))
        expected_tax = shares_to_sell * capital_gain_per_share * expected_tax_rate
        
        self.assertAlmostEqual(
            float(tax),
            float(expected_tax),
            places=2,
            msg="La tassa su vendita parziale dovrebbe usare il PMC corretto"
        )
        
        # Scenario 3: Vendita con PMC più alto del prezzo corrente (perdita)
        high_pmc = Decimal('150')
        current_price = Decimal('120')
        target_net = Decimal('1000')
        
        gross_amount, tax = self.simulator.calculate_capital_gains_tax(
            high_pmc,
            current_price,
            target_net
        )
        
        self.assertEqual(
            tax,
            Decimal('0'),
            "Non dovrebbe esserci tassazione quando il PMC è più alto del prezzo corrente"
        )
        self.assertEqual(
            gross_amount,
            target_net,
            "L'importo lordo dovrebbe essere uguale al netto in caso di perdita"
        )
        
        # Scenario 4: PMC uguale al prezzo corrente (no gain/loss)
        current_price = Decimal('100')
        pmc = Decimal('100')
        target_net = Decimal('1000')
        
        gross_amount, tax = self.simulator.calculate_capital_gains_tax(
            pmc,
            current_price,
            target_net
        )
        
        self.assertEqual(
            tax,
            Decimal('0'),
            "Non dovrebbe esserci tassazione quando PMC = prezzo corrente"
        )
        self.assertEqual(
            gross_amount,
            target_net,
            "L'importo lordo dovrebbe essere uguale al netto quando non c'è gain/loss"
        )

    def test_shares_to_buy_calculation(self):
        """Test del calcolo del numero di azioni da acquistare."""
        # Caso base: acquisto normale
        investment = Decimal('1000')
        price = Decimal('10')
        shares = self.simulator.calculate_shares_to_buy(investment, price)
        self.assertEqual(
            shares,
            Decimal('100'),
            "Dovrebbe acquistare il massimo numero intero di azioni possibile"
        )

        # Caso con decimali: deve arrotondare per difetto
        investment = Decimal('1005')
        price = Decimal('10')
        shares = self.simulator.calculate_shares_to_buy(investment, price)
        self.assertEqual(
            shares,
            Decimal('100'),
            "Dovrebbe arrotondare per difetto il numero di azioni"
        )

        # Caso con investimento zero
        investment = Decimal('0')
        price = Decimal('10')
        shares = self.simulator.calculate_shares_to_buy(investment, price)
        self.assertEqual(
            shares,
            Decimal('0'),
            "Con investimento zero dovrebbe restituire zero azioni"
        )

        # Caso con prezzo zero (dovrebbe gestire l'errore)
        investment = Decimal('1000')
        price = Decimal('0')
        shares = self.simulator.calculate_shares_to_buy(investment, price)
        self.assertEqual(
            shares,
            Decimal('0'),
            "Con prezzo zero dovrebbe restituire zero azioni"
        )

    def test_shares_to_sell_calculation(self):
        """Test del calcolo del numero di azioni da vendere."""
        # Caso base: vendita normale
        gross_amount = Decimal('1000')
        price = Decimal('10')
        available_shares = Decimal('200')
        shares = self.simulator.calculate_shares_to_sell(
            gross_amount, price, available_shares
        )
        self.assertEqual(
            shares,
            Decimal('100'),
            "Dovrebbe vendere il numero corretto di azioni"
        )

        # Caso con azioni insufficienti
        gross_amount = Decimal('3000')
        price = Decimal('10')
        available_shares = Decimal('200')
        shares = self.simulator.calculate_shares_to_sell(
            gross_amount, price, available_shares
        )
        self.assertEqual(
            shares,
            available_shares,
            "Non dovrebbe vendere più azioni di quelle disponibili"
        )

        # Caso con decimali: deve arrotondare per eccesso
        gross_amount = Decimal('995')
        price = Decimal('10')
        available_shares = Decimal('200')
        shares = self.simulator.calculate_shares_to_sell(
            gross_amount, price, available_shares
        )
        self.assertEqual(
            shares,
            Decimal('100'),
            "Dovrebbe arrotondare per eccesso per garantire l'importo lordo richiesto"
        )

    def test_average_price_update(self):
        """Test dell'aggiornamento del prezzo medio di acquisto."""
        # Caso base: nuovo acquisto
        position = ETFPosition(
            shares=Decimal('100'),
            price=Decimal('10'),
            avg_price=Decimal('10')
        )
        new_shares = Decimal('50')
        new_price = Decimal('12')
        
        new_avg_price = self.simulator.update_average_price(
            position, new_shares, new_price
        )
        
        # Verifica che il nuovo prezzo medio sia corretto
        expected_avg = (Decimal('100') * Decimal('10') + Decimal('50') * Decimal('12')) / Decimal('150')
        self.assertAlmostEqual(
            float(new_avg_price),
            float(expected_avg),
            places=2,
            msg="Il nuovo prezzo medio dovrebbe essere calcolato correttamente"
        )

        # Caso con zero nuove shares
        new_avg_price = self.simulator.update_average_price(
            position, Decimal('0'), new_price
        )
        self.assertEqual(
            new_avg_price,
            position.avg_price,
            "Con zero nuove shares il prezzo medio non dovrebbe cambiare"
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)