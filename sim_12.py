import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Typing imports
from typing import Union, Dict, List, Optional, Tuple

# Standard library imports
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, ROUND_UP
from multiprocessing import Pool, cpu_count
from datetime import datetime
from pathlib import Path
from io import StringIO
import sys
import math
import yaml
import logging

# Third party imports
from tqdm import tqdm
from fpdf import FPDF
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from portfolio.reporting import PortfolioPDFExporter
from portfolio.models import RiskMetrics, ExecutionConfig, ETFPosition, SimulationConfig
from portfolio.models.risk_metrics import RiskMetrics
from portfolio.stress_testing import StressTestManager
from portfolio.analyzers import PortfolioAnalyzer
from reporting import ReportFormatter
import logging
logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> 'SimulationConfig':
    """Load simulation configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
            # Estrai e converti le impostazioni di esecuzione
            exec_settings = config_dict.pop('execution_settings', {})
            execution_settings = ExecutionConfig(
                debug_mode=exec_settings.get('debug_mode', False),
                parallel_enabled=exec_settings.get('parallel_execution', {}).get('enabled', True),
                max_cores=exec_settings.get('parallel_execution', {}).get('max_cores', "auto"),
                chunk_size=exec_settings.get('parallel_execution', {}).get('chunk_size', 100),
                memory_limit_mb=exec_settings.get('parallel_execution', {}).get('memory_limit_mb', 1024)
            )
            
            # Crea la configurazione
            config = SimulationConfig(
                **config_dict,
                execution_settings=execution_settings
            )
            
            logger.info(f"Configurazione caricata con successo. "
                    f"Modalità debug: {config.execution_settings.debug_mode}, "
                    f"Cores disponibili: {config.execution_settings.max_cores}")
            
            return config
            
    except FileNotFoundError:
        raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Errore nel parsing del file YAML: {e}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise
     
@classmethod
def from_yaml(cls, config_path: str = 'config.yaml') -> 'SimulationConfig':
        """
        Carica la configurazione da un file YAML.
            
        Args:
            config_path: Percorso del file di configurazione YAML
                
        Returns:
            SimulationConfig: Oggetto configurazione
                
        Raises:
            Exception: Se ci sono errori nel caricamento o nella validazione
            """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
                return cls(**config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"File di configurazione non trovato: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Errore nel parsing del file YAML: {e}")
        except Exception as e:
            raise Exception(f"Errore nel caricamento della configurazione: {e}")

class MonteCarloSimulator:
    def __init__(self, config: SimulationConfig):
        print("\nCONFIG CHECK:")
        print(f"Config type: {type(config)}")
        print(f"Has execution_settings: {hasattr(config, 'execution_settings')}")
        if hasattr(config, 'execution_settings'):
            print(f"Execution settings: {vars(config.execution_settings)}")
        self.config = config
        self.analyzer = PortfolioAnalyzer(config)
        self.current_year = datetime.now().year
        self.initial_positions = {
            "ETF SWDA": ETFPosition(
                shares=Decimal('2285'),
                price=Decimal('99.89'),
                avg_price=Decimal('77.43')
            ),
            "ETF S&P500": ETFPosition(
                shares=Decimal('47'),
                price=Decimal('571.03'),
                avg_price=Decimal('439.76')
            ),
            "ETF Eur stoxx50": ETFPosition(
                shares=Decimal('302'),
                price=Decimal('84.11'),
                avg_price=Decimal('71.78')
            )
        }
        self.allocations = [0.8, 0.1, 0.1]
        self._validate_simulator()

    def _worker_simulation(self, args):
                """
                Funzione worker per eseguire una singola simulazione in un processo separato.
                """
                initial_positions, allocations = args
                return self.run_single_simulation(initial_positions, allocations)

    def run_batch_simulations(self, initial_positions: Dict[str, ETFPosition],
                         allocations: List[float],
                         num_simulations: int) -> List[List[Decimal]]:
        """Esegue le simulazioni in modalità parallela o sequenziale."""
        try:
            print("\nBATCH SIMULATION START")
            print(f"Modalità parallela: {self.config.execution_settings.parallel_enabled}")
            print(f"Cores disponibili: {self.config.execution_settings.max_cores}")
            
            # Esegui prima simulazione sempre in modo dettagliato
            print("\nEsecuzione prima simulazione...")
            all_simulations = []
            first_sim = self.run_single_simulation(initial_positions, allocations, True)
            all_simulations.append(first_sim)
            
            # Calcola il numero di simulazioni rimanenti
            remaining_sims = num_simulations - 1
            
            if remaining_sims > 0:
                if self.config.execution_settings.parallel_enabled:
                    print(f"\nAvvio {remaining_sims} simulazioni in parallelo su {self.config.execution_settings.max_cores} cores")
                    chunk_size = self.config.get_optimal_chunk_size(remaining_sims)
                    print(f"Chunk size: {chunk_size}")
                    
                    args = [(initial_positions, allocations)] * remaining_sims
                    with Pool(processes=self.config.execution_settings.max_cores) as pool:
                        for result in tqdm(
                            pool.imap_unordered(self._worker_simulation, args, chunksize=chunk_size),
                            total=remaining_sims,
                            desc="Simulazioni parallele"
                        ):
                            all_simulations.append(result)
                    print("Simulazioni parallele completate")
                else:
                    print(f"\nEsecuzione {remaining_sims} simulazioni in sequenziale")
                    for i in range(remaining_sims):
                        if i % 100 == 0:
                            print(f"Simulazione {i+1}/{remaining_sims}")
                        sim = self.run_single_simulation(initial_positions, allocations, False)
                        all_simulations.append(sim)
            
            return all_simulations
            
        except Exception as e:
            print(f"Errore nelle simulazioni batch: {str(e)}")
            logger.error(f"Errore nelle simulazioni batch: {e}")
            raise

    def update_average_price(self, position: ETFPosition, new_shares: Decimal,
                           new_price: Decimal) -> Decimal:
        """Calculate the new average price after buying additional shares."""
        try:
            if new_shares == 0:
                return position.avg_price
            total_shares = position.shares + new_shares
            return ((position.shares * position.avg_price + new_shares * new_price)
                   / total_shares).quantize(Decimal('0.01'), ROUND_HALF_UP)
        except Exception as e:
            logger.error(f"Error calculating average price: {e}")
            raise

    def calculate_capital_gains_tax(self, avg_price: Decimal, current_price: Decimal,
                                  target_net: Decimal) -> Tuple[Decimal, Decimal]:
        """Calculate the gross amount needed to withdraw and the tax amount."""
        try:
            if current_price <= avg_price or target_net <= 0:
                return target_net, Decimal('0')
            
            tax_rate = Decimal('0.26')  # 26% in Italy
            gain_ratio = (current_price - avg_price) / current_price
            theoretical_gross = target_net / (1 - (gain_ratio * tax_rate))
            tax_amount = theoretical_gross - target_net
            
            return (theoretical_gross.quantize(Decimal('0.01'), ROUND_HALF_UP),
                   tax_amount.quantize(Decimal('0.01'), ROUND_HALF_UP))
        except Exception as e:
            logger.error(f"Error calculating capital gains tax: {e}")
            raise

    def calculate_shares_to_buy(self, investment: Decimal, price: Decimal) -> Decimal:
        """Calculate how many whole shares can be bought with the given investment."""
        try:
            if price <= 0 or investment <= 0:
                return Decimal('0')
            shares = (investment / price).quantize(Decimal('1.'), rounding=ROUND_DOWN)
            return shares
        except Exception as e:
            logger.error(f"Error calculating shares to buy: {e}")
            raise

    def calculate_shares_to_sell(self, gross_amount: Decimal, price: Decimal,
                               available_shares: Decimal) -> Decimal:
        """Calculate whole number of shares to sell based on the gross amount needed."""
        try:
            if price <= 0 or gross_amount <= 0:
                return Decimal('0')
            theoretical_shares = (gross_amount / price).quantize(Decimal('1.'), rounding=ROUND_UP)
            return min(theoretical_shares, available_shares)
        except Exception as e:
            logger.error(f"Error calculating shares to sell: {e}")
            raise

    def _calculate_withdrawal_start_year(self) -> int:
            """Calcola l'anno di inizio della fase di prelievo."""
            return (self.current_year + 
                    self.config.accumulation_years + 
                    (self.config.maintenance_years if self.config.maintenance_years > 0 else 0))

    def _validate_simulator(self):
        """Validate simulator configuration."""
        if not isinstance(self.config, SimulationConfig):
            raise ValueError("Config must be an instance of SimulationConfig")

    def _validate_allocations(self, allocations: List[float]):
        """Validate that allocations sum to 1."""
        if not math.isclose(sum(allocations), 1.0, rel_tol=1e-9):
            raise ValueError("Allocations must sum to 1.0")

    def _check_black_swan_event(self) -> bool:
        """Check if a black swan event occurs based on the configured probability."""
        return np.random.random() < float(self.config.black_swan_probability) / 100

    def _apply_black_swan_impact(self, positions: Dict[str, ETFPosition], 
                               is_first_simulation: bool) -> None:
        """Apply black swan impact to all positions."""
        impact_multiplier = 1 + (self.config.black_swan_impact / 100)
        for etf_name, position in positions.items():
            new_price = position.price * Decimal(str(impact_multiplier))
            positions[etf_name] = ETFPosition(
                shares=position.shares,
                price=new_price,
                avg_price=position.avg_price
            )
            if is_first_simulation:
                print(f"\n=== BLACK SWAN EVENT OCCURRED ===")
                print(f"Impact on {etf_name}:")
                print(f"  Previous Price: €{position.price:.2f}")
                print(f"  New Price: €{new_price:.2f}")
                print(f"  Value Impact: {self.config.black_swan_impact}%")

    def run_single_simulation(self, initial_positions: Dict[str, ETFPosition],
                         allocations: List[float],
                         is_first_simulation: bool = False) -> List[Decimal]:
        """Run a single simulation and return the capital values."""
        try:
            self._validate_allocations(allocations)
            positions = {k: ETFPosition(v.shares, v.price, v.avg_price) 
                        for k, v in initial_positions.items()}
            capitals = []
            total_years = (self.config.accumulation_years + 
                        self.config.maintenance_years + 
                        self.config.withdrawal_years)

            if is_first_simulation:
                self._print_initial_portfolio(positions)

            # Run accumulation phase if years > 0
            if self.config.accumulation_years > 0:
                accumulation_capitals = self._run_accumulation_phase(
                    positions, allocations, is_first_simulation)
                capitals.extend(accumulation_capitals)

            # Run maintenance phase if years > 0
            if self.config.maintenance_years > 0:
                maintenance_capitals = self._run_maintenance_phase(
                    positions, allocations, is_first_simulation)
                capitals.extend(maintenance_capitals)

            # Run withdrawal phase if years > 0
            if self.config.withdrawal_years > 0:
                withdrawal_capitals = self._run_withdrawal_phase(
                    positions, allocations, is_first_simulation)
                capitals.extend(withdrawal_capitals)

            return capitals
        except Exception as e:
            logger.error(f"Error in simulation: {e}")
            raise

    def _calculate_simulation_years(self) -> List[int]:
        """Calcola la sequenza degli anni per la simulazione."""
        years = []
        current_year = self.current_year
        
        # Anni di accumulo
        if self.config.accumulation_years > 0:
            years.extend(range(current_year, 
                            current_year + self.config.accumulation_years))

        # Anni di mantenimento
        if self.config.maintenance_years > 0:
            maintenance_start = current_year + self.config.accumulation_years
            years.extend(range(maintenance_start, 
                            maintenance_start + self.config.maintenance_years))

        # Anni di prelievo
        if self.config.withdrawal_years > 0:
            withdrawal_start = (current_year + 
                            self.config.accumulation_years + 
                            (self.config.maintenance_years if self.config.maintenance_years > 0 else 0))
            years.extend(range(withdrawal_start, 
                            withdrawal_start + self.config.withdrawal_years))
        
        return years

    def _run_maintenance_phase(self, positions: Dict[str, ETFPosition],
                         allocations: List[float],
                         is_first_simulation: bool) -> List[Decimal]:
        """Run the maintenance phase of the simulation."""
        capitals = []
        maintenance_start_year = self.current_year + self.config.accumulation_years
    
        for year in range(1, self.config.maintenance_years + 1):
            simulation_year = maintenance_start_year + year - 1
            
            if self._check_black_swan_event():
                self._apply_black_swan_impact(positions, is_first_simulation)
                
            growth_rate = Decimal(str(np.random.normal(
                float(self.config.mean_return),
                float(self.config.std_dev_return)
            ) / 100))
            
            total_value = self._process_maintenance_year(
                positions, allocations, year, simulation_year, 
                growth_rate, is_first_simulation)
            
            if total_value <= 0:
                remaining_years = (self.config.maintenance_years - len(capitals))
                capitals.extend([Decimal('0')] * remaining_years)
                if is_first_simulation:
                    print("\n=== PORTFOLIO DEPLETED DURING MAINTENANCE ===")
                break
                
            capitals.append(total_value)
    
        return capitals

    def _print_initial_portfolio(self, positions: Dict[str, ETFPosition]):
        """Print initial portfolio positions."""
        print(f"\n=== INITIAL PORTFOLIO POSITIONS ({self.current_year}) ===")
        total_initial_value = Decimal('0')
        
        for etf_name, position in positions.items():
            value = position.value
            total_initial_value += value
            print(f"{etf_name}:")
            print(f"  Shares: {position.shares:,.0f}")
            print(f"  Price: €{position.price:.2f}")
            print(f"  Avg Price: €{position.avg_price:.2f}")
            print(f"  Total Value: €{value:,.2f}")
        
        print(f"Total Portfolio Value: €{total_initial_value:,.2f}\n")

    def _run_accumulation_phase(self, positions: Dict[str, ETFPosition],
                              allocations: List[float],
                              is_first_simulation: bool) -> List[Decimal]:
        """Run the accumulation phase of the simulation."""
        capitals = []
        for year in range(1, self.config.accumulation_years + 1):
            simulation_year = self.current_year + year - 1
            
            if self._check_black_swan_event():
                self._apply_black_swan_impact(positions, is_first_simulation)
            
            growth_rate = Decimal(str(np.random.normal(
                float(self.config.mean_return),
                float(self.config.std_dev_return)
            ) / 100))
            
            total_value = self._process_accumulation_year(
                positions, allocations, growth_rate, simulation_year, is_first_simulation)
            capitals.append(total_value)
        
        return capitals

    def _run_withdrawal_phase(self, positions: Dict[str, ETFPosition],
                         allocations: List[float],
                         is_first_simulation: bool) -> List[Decimal]:
        """Run the withdrawal phase of the simulation."""
        capitals = []
        withdrawal_start_year = self._calculate_withdrawal_start_year()
        
        for year in range(1, self.config.withdrawal_years + 1):
            simulation_year = withdrawal_start_year + year - 1
                        
            if self._check_black_swan_event():
                self._apply_black_swan_impact(positions, is_first_simulation)
                
            growth_rate = Decimal(str(np.random.normal(
                float(self.config.mean_return),
                float(self.config.std_dev_return)
            ) / 100))
            
            total_value = self._process_withdrawal_year(
                positions, allocations, year, simulation_year, 
                growth_rate, is_first_simulation)
            
            if total_value <= 0:
                remaining_years = (self.config.withdrawal_years - len(capitals))
                capitals.extend([Decimal('0')] * remaining_years)
                if is_first_simulation:
                    print("\n=== PORTFOLIO DEPLETED ===")
                break
                
            capitals.append(total_value)
        
        return capitals

    def _process_accumulation_year(self, positions: Dict[str, ETFPosition],
                                 allocations: List[float], growth_rate: Decimal,
                                 simulation_year: int,
                                 is_first_simulation: bool) -> Decimal:
        """Process a single year during the accumulation phase."""
        if is_first_simulation:
            print(f"\n=== ACCUMULATION YEAR {simulation_year} ===")
            print(f"Market Return: {growth_rate*100:.2f}%")

        total_value = Decimal('0')
        uninvested_cash = Decimal('0')
        
        for etf_name, position in positions.items():
            # Apply market growth
            new_price = position.price * (1 + growth_rate)
            if new_price <= 0:
                logger.warning(f"Invalid market price for {etf_name}: {new_price}. Using minimum price of 0.01")
                new_price = Decimal('0.01')
            
            # Calculate investment
            allocation_index = list(positions.keys()).index(etf_name)
            investment = (self.config.investment_amount * 
                        Decimal(str(allocations[allocation_index])))
            
            # Execute investment with whole shares only
            if investment > 0 and new_price > 0:
                new_shares = self.calculate_shares_to_buy(investment, new_price)
                actual_investment = new_shares * new_price
                uninvested_cash += investment - actual_investment
                
                if new_shares > 0:
                    new_avg_price = self.update_average_price(position, new_shares, new_price)
                else:
                    new_avg_price = position.avg_price
            else:
                new_shares = Decimal('0')
                new_avg_price = position.avg_price
                uninvested_cash += investment
            
            positions[etf_name] = ETFPosition(
                shares=position.shares + new_shares,
                price=new_price,
                avg_price=new_avg_price
            )
            
            total_value += positions[etf_name].value
            
            if is_first_simulation:
                self._print_accumulation_details(etf_name, new_price, investment,
                                              new_shares, positions[etf_name],
                                              actual_investment if new_shares > 0 else Decimal('0'))

        if is_first_simulation:
            if uninvested_cash > 0:
                print(f"\nUninvested Cash: €{uninvested_cash:,.2f}")
            print(f"Total Portfolio Value: €{total_value:,.2f}")
        
        return total_value

    def _print_accumulation_details(self, etf_name: str, new_price: Decimal,
                                  investment: Decimal, new_shares: Decimal,
                                  position: ETFPosition, actual_investment: Decimal):
        """Print details for accumulation phase."""
        print(f"\n{etf_name}:")
        print(f"  New Price: €{new_price:.2f}")
        print(f"  Target Investment: €{investment:,.2f}")
        print(f"  Actual Investment: €{actual_investment:,.2f}")
        print(f"  New Shares Purchased: {new_shares:,.0f}")
        print(f"  Total Shares: {position.shares:,.0f}")
        print(f"  New Avg Price: €{position.avg_price:.2f}")
        print(f"  Position Value: €{position.value:,.2f}")

    def _process_maintenance_year(self, positions: Dict[str, ETFPosition],
                                allocations: List[float], year: int,
                                simulation_year: int, growth_rate: Decimal,
                                is_first_simulation: bool) -> Decimal:
        """Process a single year during the maintenance phase."""
        # Calculate required withdrawal with inflation adjustment
        inflated_withdrawal = self.config.maintenance_withdrawal * (
            (1 + self.config.inflation_rate / 100) ** year)

        if is_first_simulation:
            self._print_maintenance_year_header(
                simulation_year, growth_rate, inflated_withdrawal)

        return self._process_maintenance_positions(
            positions, allocations, growth_rate, inflated_withdrawal, is_first_simulation)

    def _print_maintenance_year_header(self, simulation_year: int,
                                     growth_rate: Decimal,
                                     inflated_withdrawal: Decimal):
        """Print header information for maintenance year."""
        print(f"\n=== MAINTENANCE YEAR {simulation_year} ===")
        print(f"Market Return: {growth_rate*100:.2f}%")
        print(f"Required Withdrawal (Inflation Adjusted): €{inflated_withdrawal:,.2f}")

    def _process_maintenance_positions(self, positions: Dict[str, ETFPosition],
                                    allocations: List[float],
                                    growth_rate: Decimal,
                                    target_withdrawal: Decimal,
                                    is_first_simulation: bool) -> Decimal:
        """Process positions during maintenance phase with whole shares only."""
        total_value = Decimal('0')
        total_withdrawn = Decimal('0')
        total_tax = Decimal('0')
        
        # First pass: apply growth and calculate minimum shares needed
        updated_positions = {}
        min_shares_needed = {}
        max_additional_shares = {}
        
        for etf_name, position in positions.items():
            # Apply market growth
            new_price = position.price * (1 + growth_rate)
            if new_price <= 0:
                logger.warning(f"Invalid market price for {etf_name}: {new_price}. Using minimum price of 0.01")
                new_price = Decimal('0.01')
            
            # Calculate initial target withdrawal for this ETF
            allocation_index = list(positions.keys()).index(etf_name)
            target_portion = (target_withdrawal * 
                            Decimal(str(allocations[allocation_index])))
            
            # Calculate minimum shares needed
            theoretical_gross, _ = self.calculate_capital_gains_tax(
                position.avg_price, new_price, target_portion)
            min_shares = self.calculate_shares_to_sell(
                theoretical_gross, new_price, position.shares)
            
            # Calculate maximum additional shares available
            max_additional = position.shares - min_shares
            
            min_shares_needed[etf_name] = min_shares
            max_additional_shares[etf_name] = max_additional
            updated_positions[etf_name] = (position, new_price)

            # Execute sales and update positions
        for etf_name, (position, new_price) in updated_positions.items():
            shares_to_sell = min_shares_needed[etf_name]
            actual_gross = shares_to_sell * new_price
            actual_gain = max(Decimal('0'), 
                            shares_to_sell * (new_price - position.avg_price))
            actual_tax = actual_gain * Decimal('0.26')
            actual_net = actual_gross - actual_tax
            
            total_withdrawn += actual_net
            total_tax += actual_tax
            
            # Update position
            new_shares = position.shares - shares_to_sell
            positions[etf_name] = ETFPosition(
                shares=new_shares,
                price=new_price,
                avg_price=position.avg_price
            )
            
            total_value += positions[etf_name].value
            
            if is_first_simulation:
                self._print_maintenance_position_details(
                    etf_name, new_price, 
                    target_withdrawal * Decimal(str(allocations[list(positions.keys()).index(etf_name)])),
                    shares_to_sell, actual_tax, positions[etf_name], actual_net)
        
        if is_first_simulation:
            self._print_maintenance_summary(total_withdrawn, total_tax, target_withdrawal, total_value)
        
        return total_value

    def _print_maintenance_position_details(self, etf_name: str, new_price: Decimal,
                                          target_portion: Decimal,
                                          shares_sold: Decimal,
                                          tax_amount: Decimal,
                                          position: ETFPosition,
                                          actual_net: Decimal):
        """Print details for each position during maintenance."""
        print(f"\n{etf_name}:")
        print(f"  New Price: €{new_price:.2f}")
        print(f"  Target Withdrawal: €{target_portion:,.2f}")
        print(f"  Actual Net Withdrawal: €{actual_net:,.2f}")
        print(f"  Shares Sold: {shares_sold:,.0f}")
        print(f"  Remaining Shares: {position.shares:,.0f}")
        print(f"  Capital Gains Tax: €{tax_amount:,.2f}")
        print(f"  Position Value: €{position.value:,.2f}")

    def _print_maintenance_summary(self, total_withdrawn: Decimal,
                                 total_tax: Decimal,
                                 target_withdrawal: Decimal,
                                 total_value: Decimal):
        """Print summary information for maintenance year."""
        remaining_withdrawal = max(Decimal('0'), target_withdrawal - total_withdrawn)
        if remaining_withdrawal > 0:
            print(f"\nWARNING: Unable to withdraw full amount.")
            print(f"Requested withdrawal: €{target_withdrawal:,.2f}")
            print(f"Actual withdrawal: €{total_withdrawn:,.2f}")
            print(f"Shortfall: €{remaining_withdrawal:,.2f}")
        print(f"Total tax paid: €{total_tax:,.2f}")
        print(f"Total amount withdrawn (net + tax): €{(total_withdrawn + total_tax):,.2f}")
        print(f"\nTotal Portfolio Value: €{total_value:,.2f}")

    def _process_withdrawal_year(self, positions: Dict[str, ETFPosition],
                               allocations: List[float], year: int,
                               simulation_year: int, growth_rate: Decimal,
                               is_first_simulation: bool) -> Decimal:
        """Process a single year during the withdrawal phase."""
        # Calculate required withdrawal
        inflated_mandatory_pension = self.config.mandatory_pension * (
            (1 + self.config.inflation_rate / 100) ** year)
        total_pension = inflated_mandatory_pension + self.config.complementary_pension
        base_withdrawal = self.config.withdrawal_amount * (
            (1 + self.config.inflation_rate / 100) ** year)
        target_net_withdrawal = max(Decimal('0'), base_withdrawal - total_pension)

        if is_first_simulation:
            self._print_withdrawal_year_header(
                simulation_year, growth_rate, inflated_mandatory_pension,
                total_pension, base_withdrawal, target_net_withdrawal)

        return self._process_withdrawal_positions(
            positions, allocations, growth_rate, target_net_withdrawal, is_first_simulation)

    def _print_withdrawal_year_header(self, simulation_year: int,
                                    growth_rate: Decimal,
                                    inflated_mandatory_pension: Decimal,
                                    total_pension: Decimal,
                                    base_withdrawal: Decimal,
                                    target_net_withdrawal: Decimal):
        """Print header information for withdrawal year."""
        print(f"\n=== WITHDRAWAL YEAR {simulation_year} ===")
        print(f"Market Return: {growth_rate*100:.2f}%")
        print(f"Inflated Mandatory Pension: €{inflated_mandatory_pension:,.2f}")
        print(f"Complementary Pension: €{self.config.complementary_pension:,.2f}")
        print(f"Total Pension Income: €{total_pension:,.2f}")
        print(f"Required Withdrawal Before Pensions: €{base_withdrawal:,.2f}")
        print(f"Required Withdrawal After Pensions: €{target_net_withdrawal:,.2f}")

    def _process_withdrawal_positions(self, positions: Dict[str, ETFPosition],
                                    allocations: List[float],
                                    growth_rate: Decimal,
                                    target_net_withdrawal: Decimal,
                                    is_first_simulation: bool) -> Decimal:
        """Process positions during withdrawal phase with whole shares only."""
        total_value = Decimal('0')
        total_withdrawn = Decimal('0')
        total_tax = Decimal('0')
        # First pass: apply growth and calculate minimum shares needed
        updated_positions = {}
        min_shares_needed = {}
        max_additional_shares = {}
        
        for etf_name, position in positions.items():
            # Apply market growth
            new_price = position.price * (1 + growth_rate)
            if new_price <= 0:
                logger.warning(f"Invalid market price for {etf_name}: {new_price}. Using minimum price of 0.01")
                new_price = Decimal('0.01')
            
            # Calculate initial target withdrawal for this ETF
            allocation_index = list(positions.keys()).index(etf_name)
            target_net_portion = (target_net_withdrawal * 
                                Decimal(str(allocations[allocation_index])))
            
            # Calculate minimum shares needed
            theoretical_gross, _ = self.calculate_capital_gains_tax(
                position.avg_price, new_price, target_net_portion)
            min_shares = self.calculate_shares_to_sell(
                theoretical_gross, new_price, position.shares)
            
            # Calculate maximum additional shares available
            max_additional = position.shares - min_shares
            
            min_shares_needed[etf_name] = min_shares
            max_additional_shares[etf_name] = max_additional
            updated_positions[etf_name] = (position, new_price)
        
        # Second pass: try to optimize share distribution to minimize shortfall
        final_shares_to_sell = min_shares_needed.copy()
        current_withdrawn = Decimal('0')
        
        # Calculate initial withdrawal with minimum shares
        for etf_name, (position, new_price) in updated_positions.items():
            shares = min_shares_needed[etf_name]
            actual_gross = shares * new_price
            actual_gain = max(Decimal('0'), 
                            shares * (new_price - position.avg_price))
            actual_tax = actual_gain * Decimal('0.26')
            current_withdrawn += actual_gross - actual_tax
        
        # If there's still a shortfall and we have additional shares available
        if current_withdrawn < target_net_withdrawal:
            remaining_needed = target_net_withdrawal - current_withdrawn
            
            # Try each ETF for additional shares
            for etf_name, (position, new_price) in updated_positions.items():
                while (max_additional_shares[etf_name] > 0 and 
                       current_withdrawn < target_net_withdrawal):
                    # Calculate impact of selling one more share
                    additional_gross = new_price
                    additional_gain = max(Decimal('0'), 
                                       new_price - position.avg_price)
                    additional_net = additional_gross - (additional_gain * Decimal('0.26'))
                    
                    if (target_net_withdrawal - current_withdrawn) >= additional_net:
                        final_shares_to_sell[etf_name] += 1
                        max_additional_shares[etf_name] -= 1
                        current_withdrawn += additional_net
                    else:
                        break
        
        # Final pass: execute the optimized sales
        for etf_name, (position, new_price) in updated_positions.items():
            shares_to_sell = final_shares_to_sell[etf_name]
            actual_gross = shares_to_sell * new_price
            actual_gain = max(Decimal('0'), 
                            shares_to_sell * (new_price - position.avg_price))
            actual_tax = actual_gain * Decimal('0.26')
            actual_net = actual_gross - actual_tax
            
            total_withdrawn += actual_net
            total_tax += actual_tax
            
            # Update position
            new_shares = position.shares - shares_to_sell
            positions[etf_name] = ETFPosition(
                shares=new_shares,
                price=new_price,
                avg_price=position.avg_price
            )
            
            total_value += positions[etf_name].value
            
            if is_first_simulation:
                self._print_withdrawal_position_details(
                    etf_name, new_price, 
                    target_net_withdrawal * Decimal(str(allocations[list(positions.keys()).index(etf_name)])),
                    shares_to_sell, actual_tax, positions[etf_name], actual_net)
        
        if is_first_simulation:
            remaining_withdrawal = max(Decimal('0'), target_net_withdrawal - total_withdrawn)
            if remaining_withdrawal > 0:
                print(f"\nWARNING: Unable to withdraw full amount.")
                print(f"Requested withdrawal: €{target_net_withdrawal:,.2f}")
                print(f"Actual withdrawal: €{total_withdrawn:,.2f}")
                print(f"Shortfall: €{remaining_withdrawal:,.2f}")
            print(f"Total tax paid: €{total_tax:,.2f}")
            print(f"Total amount withdrawn (net + tax): €{(total_withdrawn + total_tax):,.2f}")
            print(f"\nTotal Portfolio Value: €{total_value:,.2f}")
        
        return total_value

    def _print_withdrawal_position_details(self, etf_name: str, new_price: Decimal,
                                         target_net_portion: Decimal,
                                         shares_to_sell: Decimal,
                                         tax_amount: Decimal,
                                         position: ETFPosition,
                                         actual_net: Decimal):
        """Print details for each position during withdrawal."""
        print(f"\n{etf_name}:")
        print(f"  New Price: €{new_price:.2f}")
        print(f"  Target Net Withdrawal: €{target_net_portion:,.2f}")
        print(f"  Actual Net Withdrawal: €{actual_net:,.2f}")
        print(f"  Shares Sold: {shares_to_sell:,.0f}")
        print(f"  Remaining Shares: {position.shares:,.0f}")
        print(f"  Capital Gains Tax: €{tax_amount:,.2f}")
        print(f"  Position Value: €{position.value:,.2f}")

    def plot_monte_carlo_results(self, all_simulations: List[List[Decimal]], num_simulations: int):
        """Genera e visualizza i risultati della simulazione Monte Carlo."""
        try:
            # Genera il report
            report = self.generate_report(all_simulations)
            print("\nReport Dettagliato:")
            print(ReportFormatter.format_report(report))  # Modificato qui
            print("\nInterpretazione:")
            print(ReportFormatter().get_interpretation(report))
            
            # Converti i decimali in float per plotting
            simulations_array = np.array([[float(val) for val in simulation] 
                                        for simulation in all_simulations])
            
            # Ottieni gli anni totali e crea la sequenza corretta
            years = self._calculate_simulation_years()
            
            # Crea e salva il grafico timeline
            timeline_fig = plt.figure(figsize=(12, 6))
            timeline_ax = timeline_fig.add_subplot(111)
            self._plot_failure_timeline(timeline_ax, simulations_array, years, num_simulations)
            timeline_fig.savefig("timeline.png", 
                            bbox_inches='tight',
                            dpi=300,
                            pad_inches=0.5)
            plt.close(timeline_fig)
            
            # Crea e salva il grafico distribuzione
            distribution_fig = plt.figure(figsize=(12, 6))
            distribution_ax = distribution_fig.add_subplot(111)
            self._plot_distribution(distribution_ax, simulations_array, num_simulations)
            distribution_fig.savefig("distribution.png",
                                bbox_inches='tight',
                                dpi=300,
                                pad_inches=0.5)
            plt.close(distribution_fig)
            
            return report
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")
            raise

    def plot_unified_analysis(self, all_simulations: List[List[Decimal]], num_simulations: int):
        """Crea una singola finestra con i grafici principali dell'analisi e salva le immagini."""
        try:
            # Converti i decimali in float per il plotting e salva in self.simulations
            self.simulations = np.array([[float(val) for val in simulation] 
                                        for simulation in all_simulations])
            simulations_array = self.simulations
            
            # Crea una figura unica con 2 righe per i due grafici principali
            fig = plt.figure(figsize=(20, 24))  # Aumentata l'altezza per accomodare le note interpretative
            
            # Configura la griglia per i grafici con più spazio tra loro
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3, top=0.95, bottom=0.05)
            
            # Aggiungi gli assi per i grafici
            ax_timeline = fig.add_subplot(gs[0, :])
            ax_distribution = fig.add_subplot(gs[1, :])
            
            # Ottieni gli anni totali della simulazione
            years = self._calculate_simulation_years()
            
            # Creazione dei grafici
            self._plot_failure_timeline(ax_timeline, simulations_array, years, num_simulations)
            self._plot_distribution(ax_distribution, simulations_array, num_simulations)
            
            # Aggiungi note interpretative sotto ogni grafico
            timeline_interpretation = self._get_timeline_interpretation(simulations_array)
            distribution_interpretation = self._get_distribution_interpretation(simulations_array)
            
            # Aggiungi le note interpretative come testo sotto i grafici
            ax_timeline.text(0.02, -0.15, timeline_interpretation, transform=ax_timeline.transAxes,
                        fontsize=10, verticalalignment='top', bbox=dict(facecolor='wheat', alpha=0.5))
            
            ax_distribution.text(0.02, -0.15, distribution_interpretation, transform=ax_distribution.transAxes,
                            fontsize=10, verticalalignment='top', bbox=dict(facecolor='wheat', alpha=0.5))
            
            # Titolo principale
            fig.suptitle('Analisi del Portafoglio: Timeline e Distribuzione', fontsize=20, y=0.98)
            
            # Salva i grafici come immagini separate
            timeline_fig = plt.figure(figsize=(20, 12))
            timeline_ax = timeline_fig.add_subplot(111)
            self._plot_failure_timeline(timeline_ax, simulations_array, years, num_simulations)
            timeline_fig.savefig("timeline.png", bbox_inches='tight', dpi=300)
            plt.close(timeline_fig)
            
            distribution_fig = plt.figure(figsize=(20, 12))
            distribution_ax = distribution_fig.add_subplot(111)
            self._plot_distribution(distribution_ax, simulations_array, num_simulations)
            distribution_fig.savefig("distribution.png", bbox_inches='tight', dpi=300)
            plt.close(distribution_fig)
            
            plt.close(fig)  # Chiude la figura principale
            
            return fig
            
        except Exception as e:
            logger.error(f"Errore nella creazione dei grafici: {e}")
            raise

    def _plot_failure_timeline(self, ax, simulations_array, years, num_simulations):
        """Plot the distribution of simulation failures over time."""
        # Imposta la dimensione della figura prima del plotting
        fig = ax.get_figure()
        fig.set_size_inches(12, 6)  # Proporzioni 2:1
        
        # Get withdrawal phase data
        withdrawal_data = simulations_array[:, self.config.accumulation_years:]
        
        # Calculate failures per year
        failures_by_year = []
        active_simulations = num_simulations
        
        for year_idx in range(withdrawal_data.shape[1]):
            failures_this_year = np.sum((withdrawal_data[:, year_idx] <= 0) & 
                                    (withdrawal_data[:, year_idx-1] > 0 if year_idx > 0 
                                    else np.full(withdrawal_data.shape[0], True)))
            failures_by_year.append((failures_this_year / num_simulations) * 100)
            active_simulations -= failures_this_year
        
        # Assicurati che gli anni corrispondano ai dati
        withdrawal_years = years[self.config.accumulation_years:
                            self.config.accumulation_years + len(failures_by_year)]
        
        # Create the bar plot
        bars = ax.bar(withdrawal_years, failures_by_year, 
                    color='darkred', alpha=0.7)
        
        # Calculate cumulative failure percentage
        cumulative_failures = np.cumsum(failures_by_year)
        ax2 = ax.twinx()
        ax2.plot(withdrawal_years, cumulative_failures, 
                color='black', linewidth=2, label='Cumulative Failure %')
        
        # Customize the plot
        ax.set_title('Distribution of Portfolio Failures Over Time', 
                    pad=20, fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12, labelpad=10)
        ax.set_ylabel('Percentage of New Failures (%)', fontsize=12, labelpad=10)
        ax2.set_ylabel('Cumulative Failure Percentage (%)', fontsize=12, labelpad=10)
        
        # Migliora la leggibilità e mostra tutti gli anni
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xticks(withdrawal_years)  # Mostra tutti gli anni
        ax.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        
        # Add value labels on the bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show non-zero values
                ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        # Add summary statistics
        success_rate = 100 - cumulative_failures[-1]
        median_failure_year = withdrawal_years[np.searchsorted(
            cumulative_failures, 50)] if cumulative_failures[-1] > 50 else None
        
        stats_text = (
            f'Summary Statistics:\n'
            f'Overall Success Rate: {success_rate:.1f}%\n'
            f'Total Failure Rate: {cumulative_failures[-1]:.1f}%\n'
        )
        if median_failure_year:
            stats_text += f'Median Failure Year: {median_failure_year}'
        
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend for cumulative line
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines2, labels2, loc='upper right')
        
        # Aggiungi padding e regola i margini
        plt.tight_layout(pad=2.0)
        
        return fig

    def _plot_distribution(self, ax, simulations_array, num_simulations):
        """Plot distribution of final portfolio values with improved styling."""
        try:
            # Calcola i valori finali
            final_values = simulations_array[:, -1]
            final_values_millions = final_values / 1_000_000
            
            # Crea histogram con stile migliorato
            sns.histplot(final_values_millions, bins=50, ax=ax,
                        color='royalblue', alpha=0.7,
                        stat='density')
            
            # Aggiungi kernel density estimate
            sns.kdeplot(data=final_values_millions, ax=ax,
                    color='darkblue', linewidth=2)
            
            # Personalizza il plot
            ax.set_title('Distribuzione dei Valori Finali del Portafoglio', 
                        fontsize=14, pad=20, fontweight='bold')
            ax.set_xlabel('Valore Finale del Portafoglio (milioni di EUR)', 
                        fontsize=12, labelpad=10)
            ax.set_ylabel('Densita', fontsize=12, labelpad=10)
            
            # Migliora la leggibilità degli assi
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Formatta l'asse x per mostrare i milioni con separatori delle migliaia
            def millions_formatter(x, p):
                return f"{x:,.1f}M EUR"
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
            
            # Calcola le statistiche
            median_value = np.median(final_values)
            mean_value = np.mean(final_values)
            p10 = np.percentile(final_values, 10)
            p90 = np.percentile(final_values, 90)
            
            # Preparazione del box statistiche
            stats_text = (
                f'Statistiche Riassuntive:\n'
                f'Mediana: {median_value/1_000_000:,.1f}M EUR\n'
                f'Media: {mean_value/1_000_000:,.1f}M EUR\n'
                f'10° percentile: {p10/1_000_000:,.1f}M EUR\n'
                f'90° percentile: {p90/1_000_000:,.1f}M EUR'
            )
            
            # Aggiungi il box statistiche
            ax.text(0.98, 0.95, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            alpha=0.9,
                            edgecolor='gray'),
                    fontsize=10)
            
            # Aggiungi linee verticali per le statistiche chiave
            ax.axvline(median_value/1_000_000, color='red', 
                    linestyle='--', alpha=0.5, label='Mediana')
            ax.axvline(p10/1_000_000, color='orange', 
                    linestyle='--', alpha=0.5, label='10° percentile')
            ax.axvline(p90/1_000_000, color='green', 
                    linestyle='--', alpha=0.5, label='90° percentile')
            
            # Aggiungi la legenda
            ax.legend(fontsize=10, framealpha=0.9,
                    loc='upper right',
                    bbox_to_anchor=(0.98, 0.60))
            
            plt.tight_layout(pad=2.0)
            
            return ax
            
        except Exception as e:
            logger.error(f"Errore nella creazione del grafico distribuzione: {e}")
            raise    

    def _get_timeline_interpretation(self, simulations_array: np.ndarray) -> str:
        """Genera l'interpretazione della timeline dei fallimenti."""
        # Calcola le metriche chiave per l'interpretazione
        withdrawal_start_idx = (self.config.accumulation_years + 
                            self.config.maintenance_years)
        withdrawal_phase = simulations_array[:, withdrawal_start_idx:]
        
        total_sims = len(withdrawal_phase)
        failures_by_year = []
        active_sims = total_sims
        
        for year in range(withdrawal_phase.shape[1]):
            failures = np.sum((withdrawal_phase[:, year] <= 0) & 
                            (withdrawal_phase[:, year-1] > 0 if year > 0 
                            else np.full(withdrawal_phase.shape[0], True)))
            failure_rate = failures / total_sims
            failures_by_year.append(failure_rate)
            active_sims -= failures
        
        cumulative_failures = np.cumsum(failures_by_year)
        final_failure_rate = cumulative_failures[-1]
        early_failures = cumulative_failures[4] if len(cumulative_failures) > 4 else 0
        
        # Costruisci l'interpretazione
        interpretation = "INTERPRETAZIONE TIMELINE:\n"
        
        # Valuta il rischio complessivo
        if final_failure_rate <= 0.05:
            interpretation += "• Rischio di fallimento molto basso nel lungo termine\n"
        elif final_failure_rate <= 0.15:
            interpretation += "• Rischio di fallimento moderato nel lungo termine\n"
        else:
            interpretation += "• Rischio di fallimento significativo nel lungo termine\n"
        
        # Valuta i fallimenti precoci
        if early_failures <= 0.02:
            interpretation += "• Rischio minimo di fallimento nei primi 5 anni\n"
        elif early_failures <= 0.05:
            interpretation += "• Basso rischio di fallimento nei primi 5 anni\n"
        else:
            interpretation += "• Attenzione: rischio elevato di fallimento precoce\n"
        
        # Analizza la distribuzione dei fallimenti
        max_yearly_rate = max(failures_by_year)
        if max_yearly_rate <= 0.02:
            interpretation += "• Distribuzione uniforme dei fallimenti nel tempo\n"
        elif max_yearly_rate <= 0.05:
            interpretation += "• Alcuni picchi di fallimento in anni specifici\n"
        else:
            interpretation += "• Presenza di anni critici con alto tasso di fallimento\n"
        
        # Aggiungi raccomandazioni
        if final_failure_rate > 0.15 or early_failures > 0.05:
            interpretation += "\nRACCOMANDAZIONI:\n"
            if final_failure_rate > 0.15:
                interpretation += "• Considerare una riduzione dei prelievi annuali\n"
            if early_failures > 0.05:
                interpretation += "• Valutare un cuscinetto di liquidità per i primi anni\n"
            if max_yearly_rate > 0.05:
                interpretation += "• Pianificare strategie di mitigazione per gli anni critici"
        
        return interpretation

    def _get_distribution_interpretation(self, simulations_array: np.ndarray) -> str:
        """Genera l'interpretazione della distribuzione dei valori finali."""
        final_values = simulations_array[:, -1]
        
        # Calcola le statistiche chiave
        mean_value = np.mean(final_values)
        median_value = np.median(final_values)
        std_dev = np.std(final_values)
        skewness = (mean_value - median_value) / std_dev if std_dev > 0 else 0
        p10 = np.percentile(final_values, 10)
        p90 = np.percentile(final_values, 90)
        zero_portfolios = np.sum(final_values <= 0) / len(final_values)
        
        # Costruisci l'interpretazione
        interpretation = "INTERPRETAZIONE DISTRIBUZIONE:\n"
        
        # Analizza la forma della distribuzione
        if abs(skewness) < 0.2:
            interpretation += "• Distribuzione sostanzialmente simmetrica\n"
        elif skewness > 0:
            interpretation += "• Presenza di code positive (potenziali guadagni elevati)\n"
        else:
            interpretation += "• Presenza di code negative (rischi di perdite significative)\n"
        
        # Analizza la dispersione
        cv = std_dev / mean_value if mean_value > 0 else float('inf')
        if cv < 0.5:
            interpretation += "• Bassa volatilità dei risultati finali\n"
        elif cv < 1.0:
            interpretation += "• Moderata volatilità dei risultati finali\n"
        else:
            interpretation += "• Alta volatilità dei risultati finali\n"
        
        # Analizza il rischio di rovina
        if zero_portfolios <= 0.05:
            interpretation += "• Rischio minimo di esaurimento del capitale\n"
        elif zero_portfolios <= 0.15:
            interpretation += "• Rischio moderato di esaurimento del capitale\n"
        else:
            interpretation += "• Rischio significativo di esaurimento del capitale\n"
        
        # Aggiungi raccomandazioni
        interpretation += "\nRACCOMANDAZIONI:\n"
        if zero_portfolios > 0.15:
            interpretation += "• Considerare una strategia più conservativa\n"
        if cv > 1.0:
            interpretation += "• Valutare una diversificazione maggiore\n"
        if skewness < -0.2:
            interpretation += "• Implementare strategie di protezione dal rischio"
        
        return interpretation

    def perform_stress_test(self, positions: Dict[str, ETFPosition], allocations: List[float]) -> Dict:
        """Esegue tutti gli stress test sul portafoglio."""
        stress_tester = StressTestManager(self.config)
        return stress_tester.perform_stress_test(positions, allocations)

    def generate_report(self, all_simulations):
        """Genera il report delle simulazioni."""
        try:
            # Converti i decimali in float per i calcoli
            simulations_array = np.array([[float(val) for val in sim] 
                                        for sim in all_simulations])
            
            # Calcola le metriche usando l'analyzer
            risk_metrics = self.analyzer.calculate_risk_metrics(simulations_array)
            withdrawal_analysis = self.analyzer.analyze_withdrawal_sustainability(simulations_array)
            stress_test = self.perform_stress_test(self.initial_positions, self.allocations)
        
            # Crea il report
            report = {
                'risk_metrics': risk_metrics,
                'withdrawal_analysis': withdrawal_analysis,
                'stress_test': stress_test
            }
        
            return report
            
        except Exception as e:
            logger.error(f"Errore nella generazione del report: {e}")
            raise
                
    def _find_max_negative_sequence(self, returns: np.ndarray) -> int:
        """Trova la sequenza negativa più lunga nei rendimenti."""
        try:
            max_length = 0
            current_length = 0
            
            for value in returns.flatten():
                if value < 0:
                    current_length += 1
                    max_length = max(max_length, current_length)
                else:
                    current_length = 0
                    
            return max_length
            
        except Exception as e:
            logger.error(f"Error finding max negative sequence: {e}")
            return 0

class PortfolioReport:
    """Sistema avanzato di reporting per l'analisi del portafoglio."""

    def __init__(self, simulations: List[List[Decimal]], config: SimulationConfig):
        if not isinstance(simulations, list) or not all(isinstance(sim, list) for sim in simulations):
            raise ValueError("simulations deve essere una Lista di Liste")
        if not isinstance(config, SimulationConfig):
            raise ValueError("config deve essere un'istanza di SimulationConfig")
        if not simulations or not simulations[0]:
            raise ValueError("simulations non può essere vuoto")
            
        try:
            self.simulations = np.array([[float(val) for val in sim] for sim in simulations])
            if np.any(~np.isfinite(self.simulations)):
                raise ValueError("I dati delle simulazioni contengono valori non finiti")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Errore nella conversione dei dati delle simulazioni: {e}")
            
        self.config = config
        self.initial_positions = {
            "ETF SWDA": ETFPosition(
                shares=Decimal('2285'),
                price=Decimal('99.89'),
                avg_price=Decimal('77.43')
            ),
            "ETF S&P500": ETFPosition(
                shares=Decimal('47'),
                price=Decimal('571.03'),
                avg_price=Decimal('439.76')
            ),
            "ETF Eur stoxx50": ETFPosition(
                shares=Decimal('302'),
                price=Decimal('84.11'),
                avg_price=Decimal('71.78')
            )
        }
        self.allocations = [0.8, 0.1, 0.1]

    def generate_complete_report(self) -> Dict:
        """
        Genera un report completo con tutte le metriche e analisi.
        
        Returns:
            Dict: Dizionario contenente tutte le analisi
            
        Raises:
            ValueError: Se si verifica un errore durante la generazione del report
        """
        try:
            return {
                'risk_metrics': self.calculate_risk_metrics(),
                'withdrawal_analysis': self.analyze_withdrawal_sustainability(),
                'market_impact': self.analyze_market_impact(),
                'stress_test': self.perform_stress_test()
            }
        except Exception as e:
            logger.error(f"Errore nella generazione del report completo: {e}")
            raise ValueError(f"Errore nella generazione del report completo: {e}")

    def analyze_withdrawal_sustainability(self) -> Dict:
        """Analisi della sostenibilità dei prelievi."""
        try:
            analyzer = PortfolioAnalyzer(self.config)
            logger.info(f"Config: accumulation_years={self.config.accumulation_years}, "
                    f"maintenance_years={self.config.maintenance_years}, "
                    f"withdrawal_years={self.config.withdrawal_years}")
            
            # Otteniamo l'analisi completa
            failure_analysis = analyzer.analyze_portfolio_failure(self.simulations)
            logger.info(f"Chiavi in failure_analysis: {failure_analysis.keys()}")
            logger.info(f"Contenuto maintenance_phase: {failure_analysis.get('maintenance_phase')}")
            logger.info(f"Contenuto withdrawal_phase: {failure_analysis.get('withdrawal_phase')}")
            logger.info(f"Contenuto combined_analysis: {failure_analysis.get('combined_analysis')}")
            
            # Creiamo il risultato
            result = {
                'median_portfolio_life': failure_analysis['withdrawal_phase']['median_portfolio_life'],
                'failure_probability_by_year': failure_analysis['withdrawal_phase']['failure_probability_by_year'],
                'maintenance_phase_analysis': failure_analysis['maintenance_phase'],
                'combined_phase_analysis': failure_analysis['combined_analysis']
            }

            logger.info(f"Struttura finale result: {result.keys()}")
            return result

        except Exception as e:
            logger.error(f"Errore nell'analisi dei prelievi: {e}", exc_info=True)
            raise ValueError(f"Errore nell'analisi dei prelievi: {e}")

    def analyze_market_impact(self) -> Dict:
        """Analisi dell'impatto delle condizioni di mercato."""
        try:
            simulator = MonteCarloSimulator(self.config)
            return simulator.perform_stress_test(self.initial_positions, self.allocations)
        except Exception as e:
            logger.error(f"Errore nell'analisi dell'impatto di mercato: {e}")
            return {
                'high_inflation': {'portfolio_impact': -0.15},
                'market_crash': {'portfolio_impact': -0.40},
                'prolonged_bear': {'portfolio_impact': -0.25},
                'combined_stress': {'portfolio_impact': -0.50}
            }

    def _calculate_median_portfolio_life(self) -> float:
        """Calcola la vita mediana del portafoglio durante la fase di prelievo."""
        try:
            withdrawal_start_idx = (self.config.accumulation_years + 
                                (self.config.maintenance_years if self.config.maintenance_years > 0 else 0))
            
            portfolio_lives = []
            for simulation in self.simulations:
                withdrawal_phase = simulation[withdrawal_start_idx:]
                if withdrawal_phase[-1] > 0:
                    portfolio_lives.append(len(withdrawal_phase))
                else:
                    zero_point = np.where(withdrawal_phase <= 0)[0][0]
                    portfolio_lives.append(zero_point)
            
            return float(np.median(portfolio_lives)) if portfolio_lives else 0.0
        except Exception as e:
            logger.warning(f"Errore nel calcolo della vita mediana del portafoglio: {e}")
            return 0.0

    def _find_max_negative_sequence(self, returns: np.ndarray) -> int:
        """Trova la sequenza negativa più lunga nei rendimenti."""
        try:
            max_length = 0
            for sim in returns:
                current_length = 0
                max_sim_length = 0
                for ret in sim:
                    if ret < 0:
                        current_length += 1
                        max_sim_length = max(max_sim_length, current_length)
                    else:
                        current_length = 0
                max_length = max(max_length, max_sim_length)
            return max_length
        except Exception as e:
            logger.warning(f"Errore nel calcolo della sequenza negativa massima: {e}")
            return 0

    def plot_analysis_dashboard(self, all_simulations: List[List[Decimal]]):
        """Crea un dashboard unificato con tutti i grafici usando Matplotlib."""
        try:
            # Crea la figura principale con i subplot
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Crea i quattro subplot
            ax1 = fig.add_subplot(gs[0, 0])  # Distribuzione
            ax2 = fig.add_subplot(gs[0, 1])  # Prelievi
            ax3 = fig.add_subplot(gs[1, 0])  # Drawdown
            ax4 = fig.add_subplot(gs[1, 1])  # Stress Test
            
            # Plot Distribuzione Valori Finali
            self._plot_distribution_mpl(ax1)
            
            # Plot Sostenibilità Prelievi
            self._plot_withdrawal_mpl(ax2)
            
            # Plot Analisi Drawdown
            self._plot_drawdown_mpl(ax3)
            
            # Plot Stress Test
            self._plot_stress_test_mpl(ax4)
            
            # Titolo principale
            fig.suptitle('Dashboard Analisi Portafoglio', fontsize=16, y=0.95)
            
            return fig
            
        except Exception as e:
            logger.error(f"Errore nella creazione del dashboard: {e}")
            raise

    def _get_distribution_interpretation(self, mean, median, std_dev, p10, p90):
        interpretation = "INTERPRETAZIONE:<br>"
        skewness = (mean - median) / std_dev
        
        if abs(skewness) < 0.2:
            interpretation += "- Distribuzione simmetrica<br>"
        elif skewness > 0:
            interpretation += "- Presenza di code positive<br>"
        else:
            interpretation += "- Presenza di code negative<br>"
        
        range_width = (p90 - p10) / median
        if range_width < 0.5:
            interpretation += "- Risultati molto concentrati<br>"
        elif range_width < 1.0:
            interpretation += "- Dispersione moderata<br>"
        else:
            interpretation += "- Alta variabilità<br>"
        
        if median > mean:
            interpretation += "- Prospettive conservative"
        else:
            interpretation += "- Potenziale di crescita"
        return interpretation

    def _get_withdrawal_interpretation(self, final_prob, median_life, early_prob):
        interpretation = "INTERPRETAZIONE:<br>"
        
        if final_prob < 0.05:
            interpretation += "- Rischio fallimento molto basso<br>"
        elif final_prob < 0.15:
            interpretation += "- Rischio fallimento accettabile<br>"
        elif final_prob < 0.25:
            interpretation += "- Rischio fallimento moderato<br>"
        else:
            interpretation += "- Rischio fallimento elevato<br>"
        
        if median_life >= self.config.withdrawal_years:
            interpretation += "- Durata portafoglio ottimale<br>"
        elif median_life >= self.config.withdrawal_years * 0.8:
            interpretation += "- Durata portafoglio adeguata<br>"
        else:
            interpretation += "- Durata portafoglio critica<br>"
        
        if early_prob < 0.01:
            interpretation += "- Rischio precoce trascurabile"
        elif early_prob < 0.05:
            interpretation += "- Basso rischio precoce"
        else:
            interpretation += "- Alto rischio precoce"
        return interpretation

    def _get_drawdown_interpretation(self, max_drawdown, avg_max_drawdown, avg_recovery):
        interpretation = "INTERPRETAZIONE:<br>"
        
        if max_drawdown < 0.2:
            interpretation += "- Drawdown max contenuto<br>"
        elif max_drawdown < 0.35:
            interpretation += "- Drawdown max significativo<br>"
        else:
            interpretation += "- Drawdown max critico<br>"
        
        if avg_max_drawdown < 0.15:
            interpretation += "- Drawdown medi moderati<br>"
        elif avg_max_drawdown < 0.25:
            interpretation += "- Drawdown medi rilevanti<br>"
        else:
            interpretation += "- Drawdown medi preoccupanti<br>"
        
        if avg_recovery < 2:
            interpretation += "- Recupero veloce"
        elif avg_recovery < 4:
            interpretation += "- Recupero nella norma"
        else:
            interpretation += "- Recupero prolungato"
        return interpretation

    def _get_stress_test_interpretation(self, worst_case, avg_impact, combined_impact):
        interpretation = "INTERPRETAZIONE:<br>"
        
        if worst_case > -0.25:
            interpretation += "- Impatto peggiore contenuto<br>"
        elif worst_case > -0.40:
            interpretation += "- Impatto peggiore significativo<br>"
        else:
            interpretation += "- Impatto peggiore severo<br>"
        
        if avg_impact > -0.20:
            interpretation += "- Impatto medio moderato<br>"
        elif avg_impact > -0.30:
            interpretation += "- Impatto medio rilevante<br>"
        else:
            interpretation += "- Impatto medio critico<br>"
        
        if worst_case < -0.40 or combined_impact < -0.50:
            interpretation += "- Rivedere strategia"
        elif avg_impact < -0.30:
            interpretation += "- Valutare hedging"
        else:
            interpretation += "- Strategia resiliente"
        return interpretation

class UTF8PDF(FPDF):
    def __init__(self, num_simulations):
        super().__init__()
        self.num_simulations = num_simulations
        self.set_auto_page_break(auto=True, margin=15)

    def footer(self):
        """Aggiunge il numero di pagina in basso a destra"""
        self.set_y(-15)  # 15 mm dalla fine
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}/{{nb}}', 0, 0, 'R')

class SimulationCapture:
    """Classe per catturare l'output della simulazione"""
    def __init__(self):
        self.captured_output = StringIO()
        self.old_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self.captured_output
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout
        return False

def main():
    try:
        print("Caricamento configurazione...")
        config = load_config()
        print("Configurazione caricata con successo")
        
        # Test della configurazione
        print("\nTest configurazione:")
        print(f"Debug mode: {config.execution_settings.debug_mode}")
        print(f"Parallel enabled: {config.execution_settings.parallel_enabled}")
        print(f"Max cores: {config.execution_settings.max_cores}")
        
        # Get number of simulations from user
        while True:
            try:
                num_simulations = int(input("Inserisci il numero di simulazioni da eseguire: "))
                if num_simulations <= 0:
                    raise ValueError("Il numero di simulazioni deve essere positivo")
                break
            except Exception as e:
                print(f"Errore: {e}. Riprova.")

        print("Inizializzazione simulatore...")
        simulator = MonteCarloSimulator(config)

        print("\nStato simulatore prima dell'esecuzione:")
        print(f"Attributi config: {dir(simulator.config)}")
        print(f"Attributi execution_settings: {dir(simulator.config.execution_settings)}")
        print(f"parallel_enabled: {simulator.config.execution_settings.parallel_enabled}")
        print(f"max_cores: {simulator.config.execution_settings.max_cores}")
        
        # Setup initial positions and allocations
        print("Configurazione posizioni iniziali...")
        initial_positions = {
            "ETF SWDA": ETFPosition(
                shares=Decimal('2285'),
                price=Decimal('99.89'),
                avg_price=Decimal('77.43')
            ),
            "ETF S&P500": ETFPosition(
                shares=Decimal('47'),
                price=Decimal('571.03'),
                avg_price=Decimal('439.76')
            ),
            "ETF Eur stoxx50": ETFPosition(
                shares=Decimal('302'),
                price=Decimal('84.11'),
                avg_price=Decimal('71.78')
            )
        }
        allocations = [0.8, 0.1, 0.1]
        
        # Cattura l'output della prima simulazione per il PDF
        with SimulationCapture() as capture:
            print("Avvio simulazioni...")
            all_simulations = simulator.run_batch_simulations(
                initial_positions=initial_positions,
                allocations=allocations,
                num_simulations=num_simulations
            )
            first_simulation_output = capture.captured_output.getvalue()
        
        print("Generazione report e grafici...")
        # Converti le simulazioni in array numpy se non lo sono già
        simulations_array = np.array([[float(val) for val in sim] for sim in all_simulations])
        report = simulator.plot_monte_carlo_results(all_simulations, num_simulations)
        
        # Aggiungi esplicitamente i dati delle simulazioni al report
        report['simulations'] = simulations_array

        # Export to PDF
        if input("\nVuoi esportare il report in PDF? (s/n): ").lower() == 's':
            try:
                filename = input("Inserisci il nome del file PDF (default: portfolio_report.pdf): ").strip()
                if not filename:
                    filename = "portfolio_report.pdf"
                if not filename.endswith('.pdf'):
                    filename += '.pdf'
                
                print("Creazione PDF...")
                pdf_exporter = PortfolioPDFExporter(config, num_simulations)
                pdf_exporter.export(
                    report_data=report,
                    simulation_details=first_simulation_output,
                    filename=filename
                )
                print(f"\nReport PDF generato con successo: {filename}")
            except Exception as e:
                print(f"Errore durante l'esportazione del PDF: {e}")
                logger.error(f"PDF Export error: {e}", exc_info=True)

    except Exception as e:
        print(f"Errore durante l'esecuzione: {e}")
        logger.error("Main execution error", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Errore critico: {e}")
        logger.error("Critical error", exc_info=True)
        import traceback
        traceback.print_exc()