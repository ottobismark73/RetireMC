import numpy as np
from typing import Dict, Optional
import logging
from decimal import Decimal
from .models.simulation_config import SimulationConfig
from .models.risk_metrics import RiskMetrics

logger = logging.getLogger(__name__)

class PortfolioAnalyzer:
    """Analyzer for portfolio performance and risk metrics."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config

    def calculate_risk_metrics(self, simulations_array: np.ndarray) -> RiskMetrics:
        """Calculate comprehensive risk metrics for the portfolio."""
        try:
            # Identificazione delle fasi
            accumulation_end = self.config.accumulation_years
            maintenance_end = accumulation_end + self.config.maintenance_years
            
            # Calcolo success rate per fase di mantenimento (se presente)
            maintenance_success_rate = 1.0
            maintenance_success = np.ones(len(simulations_array), dtype=bool)
            if self.config.maintenance_years > 0:
                maintenance_phase = simulations_array[:, accumulation_end:maintenance_end]
                maintenance_success = np.all(maintenance_phase > 0, axis=1)
                maintenance_success_rate = float(np.mean(maintenance_success))
                
            # Calcolo success rate per fase di prelievo
            withdrawal_phase = simulations_array[:, maintenance_end:]
            withdrawal_success = np.all(withdrawal_phase > 0, axis=1)
            withdrawal_success_rate = float(np.mean(withdrawal_success))
            
            # Success rate complessivo
            success_rate = float(np.mean(maintenance_success & withdrawal_success))
            
            # Calcolo statistiche dettagliate
            success_stats = {
                'total_simulations': len(simulations_array),
                'successful_simulations': int(np.sum(maintenance_success & withdrawal_success)),
                'maintenance_success_rate': maintenance_success_rate,
                'withdrawal_success_rate': withdrawal_success_rate,
                'overall_success_rate': success_rate
            }
            
            # Calcolo drawdowns
            drawdowns = self._calculate_drawdowns(simulations_array)
            max_drawdown = float(np.min(drawdowns))
            avg_drawdown = float(np.mean(np.min(drawdowns, axis=1)))
            drawdown_duration = self._calculate_drawdown_duration(drawdowns)
            recovery_time = self._calculate_average_recovery_time(simulations_array)
            underwater_periods = self._count_underwater_periods(drawdowns)
            
            # Calcolo rendimenti per le metriche di tail risk
            returns = np.nan_to_num(np.diff(simulations_array, axis=1) / 
                                np.maximum(simulations_array[:, :-1], 1e-9), 
                                nan=0.0)
            
            # Calcolo metriche di tail risk
            tail_threshold = np.percentile(returns.flatten(), 5)
            tail_prob = float(np.mean(returns.flatten() < tail_threshold))
            tail_loss = float(np.mean(returns[returns < tail_threshold]))
            skewness = float(np.mean(((returns - np.mean(returns)) / np.std(returns))**3))
            kurtosis = float(np.mean(((returns - np.mean(returns)) / np.std(returns))**4))
            
            # Calcolo metriche di stress e resilienza
            stress_resilience = self._calculate_stress_resilience(simulations_array, returns)
            black_swan_impact = self._calculate_black_swan_impact(simulations_array)
            worst_recovery = self._calculate_worst_case_recovery(drawdowns)
            
            # Calcolo metriche di stabilità
            stability_index = self._calculate_return_stability(returns)
            
            # Calcolo Risk Score
            risk_score = self._calculate_portfolio_risk_score(simulations_array, 
                                                            float(simulations_array[0, 0]))

            # Calcolo delle metriche del portafoglio finale
            portfolio_metrics = self.calculate_end_portfolio_metrics(simulations_array)
            
            return RiskMetrics(
                risk_score=risk_score,
                max_drawdown=max_drawdown,
                avg_drawdown=avg_drawdown,
                drawdown_duration=float(drawdown_duration),
                recovery_time=float(recovery_time) if recovery_time else 0.0,
                underwater_periods=int(underwater_periods),
                tail_loss_probability=tail_prob,
                expected_tail_loss=tail_loss,
                skewness=skewness,
                kurtosis=kurtosis,
                stress_resilience_score=float(stress_resilience),
                black_swan_impact=float(black_swan_impact),
                worst_case_recovery=float(worst_recovery),
                return_stability_index=float(stability_index),
                success_rate=success_rate,
                success_stats=success_stats,
                portfolio_metrics=portfolio_metrics
            )
                
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            raise

    def calculate_end_portfolio_metrics(self, simulations_array: np.ndarray) -> Dict:
        """Calcola le metriche del valore finale del portafoglio."""
        try:
            initial_values = simulations_array[:, 0]
            final_values = simulations_array[:, -1]
            total_simulations = len(simulations_array)

            # Calcolo Large End Portfolio Value (100% o più del valore iniziale)
            large_end_mask = final_values >= (2 * initial_values)  # 100% più del valore iniziale
            large_end_count = np.sum(large_end_mask)
            large_end_percentage = (large_end_count / total_simulations) * 100

            # Calcolo Small End Portfolio Value (50% o meno del valore iniziale)
            small_end_mask = (final_values <= (0.5 * initial_values)) & (final_values > 0)  # 50% o meno ma non zero
            small_end_count = np.sum(small_end_mask)
            small_end_percentage = (small_end_count / total_simulations) * 100

            return {
                'large_end_portfolio': {
                    'count': int(large_end_count),
                    'total': total_simulations,
                    'percentage': float(large_end_percentage)
                },
                'small_end_portfolio': {
                    'count': int(small_end_count),
                    'total': total_simulations,
                    'percentage': float(small_end_percentage)
                }
            }
        except Exception as e:
            logger.error(f"Error calculating end portfolio metrics: {e}")
            raise

    def _calculate_portfolio_risk_score(self, simulations_array: np.ndarray, initial_value: float) -> float:
        """Calculate the composite risk score."""
        try:
            # Calcolo success rate (40%)
            final_values = simulations_array[:, -1]
            success_rate = float(np.mean(final_values >= initial_value))
            
            # Calcolo recovery capability (30%)
            drawdowns = self._calculate_drawdowns(simulations_array)
            recovery_times = self._calculate_average_recovery_time(simulations_array)
            recovery_capability = 1 / (1 + min(recovery_times, 10)) if recovery_times else 0.5
            
            # Calcolo path stability (30%)
            returns = np.diff(simulations_array, axis=1) / np.maximum(simulations_array[:, :-1], 1e-9)
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
            path_stability = max(0, 1 - np.std(returns))
            
            # Calcolo score finale
            score = (
                0.4 * success_rate +
                0.3 * recovery_capability +
                0.3 * path_stability
            ) * 100
            
            return float(max(0, min(score, 100)))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 50.0  # Valore di default in caso di errore

    def analyze_portfolio_failure(self, simulations_array: np.ndarray) -> Dict:
        """Analyze portfolio failure across maintenance and withdrawal phases."""
        try:
            # Calcola gli indici delle diverse fasi
            maintenance_start_idx = self.config.accumulation_years
            withdrawal_start_idx = maintenance_start_idx + self.config.maintenance_years
            
            logger.info(f"Analisi fallimento - maintenance_start_idx: {maintenance_start_idx}, withdrawal_start_idx: {withdrawal_start_idx}")
            
            # Inizializza i risultati
            results = {
                'maintenance_phase': {
                    'failure_probability_by_year': [],
                    'total_failure_probability': 0.0,
                    'median_portfolio_life': 0.0
                },
                'withdrawal_phase': {
                    'failure_probability_by_year': [],
                    'total_failure_probability': 0.0,
                    'median_portfolio_life': 0.0
                },
                'combined_analysis': {
                    'total_failure_probability': 0.0,
                    'first_failure_year_distribution': {},
                    'median_survival_time': 0.0
                }
            }

            # Analisi fase di mantenimento
            if self.config.maintenance_years > 0:
                logger.info("Analizzando fase di mantenimento...")
                maintenance_phase = simulations_array[:, 
                                                maintenance_start_idx:withdrawal_start_idx]
                results['maintenance_phase'] = self._analyze_phase_failures(
                    maintenance_phase, 
                    'Maintenance'
                )
                logger.info(f"Risultati mantenimento: {results['maintenance_phase']}")

            # Analisi fase di prelievo
            if self.config.withdrawal_years > 0:
                logger.info("Analizzando fase di prelievo...")
                withdrawal_phase = simulations_array[:, withdrawal_start_idx:]
                results['withdrawal_phase'] = self._analyze_phase_failures(
                    withdrawal_phase, 
                    'Withdrawal'
                )
                logger.info(f"Risultati prelievo: {results['withdrawal_phase']}")

            # Analisi combinata
            logger.info("Eseguendo analisi combinata...")
            results['combined_analysis'] = self._analyze_combined_failures(
                simulations_array[:, maintenance_start_idx:],
                maintenance_start_idx
            )
            logger.info(f"Risultati analisi combinata: {results['combined_analysis']}")

            return results

        except Exception as e:
            logger.error(f"Error in portfolio failure analysis: {e}")
            raise

    def _analyze_phase_failures(self, phase_data: np.ndarray, phase_name: str) -> Dict:
        """Analyze failures for a specific phase."""
        try:
            num_simulations = len(phase_data)
            logger.info(f"Analizzando fase {phase_name} con {num_simulations} simulazioni")
            
            failure_probs = []
            portfolio_lives = []
            active_simulations = num_simulations
            cumulative_failures = 0

            for year in range(phase_data.shape[1]):
                # Calcola i nuovi fallimenti per questo anno
                current_failures = np.sum(
                    (phase_data[:, year] <= 0) & 
                    (phase_data[:, year-1] > 0 if year > 0 else np.full(phase_data.shape[0], True))
                )
                
                cumulative_failures += current_failures
                failure_prob = current_failures / active_simulations if active_simulations > 0 else 1.0
                failure_probs.append(failure_prob)
                
                # Aggiorna il conteggio delle simulazioni attive
                active_simulations -= current_failures

            # Calcola la vita del portafoglio per ogni simulazione
            for sim in phase_data:
                if sim[-1] > 0:
                    portfolio_lives.append(len(sim))
                else:
                    zero_point = np.where(sim <= 0)[0][0]
                    portfolio_lives.append(zero_point)

            results = {
                'failure_probability_by_year': failure_probs,
                'total_failure_probability': cumulative_failures / num_simulations,
                'median_portfolio_life': float(np.median(portfolio_lives)) if portfolio_lives else 0.0
            }
            
            logger.info(f"Risultati fase {phase_name}: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing {phase_name} phase: {e}")
            raise

    def _analyze_combined_failures(self, combined_data: np.ndarray, 
                                maintenance_start_idx: int) -> Dict:
        """Analyze failures across both phases combined."""
        try:
            num_simulations = len(combined_data)
            first_failures = {}
            survival_times = []

            for sim_idx, simulation in enumerate(combined_data):
                if np.any(simulation <= 0):
                    failure_year = np.where(simulation <= 0)[0][0] + maintenance_start_idx
                    first_failures[failure_year] = first_failures.get(failure_year, 0) + 1
                    survival_times.append(failure_year - maintenance_start_idx)
                else:
                    survival_times.append(len(simulation))

            total_failures = sum(first_failures.values())
            results = {
                'total_failure_probability': total_failures / num_simulations,
                'first_failure_year_distribution': {
                    year: count/num_simulations 
                    for year, count in sorted(first_failures.items())
                },
                'median_survival_time': float(np.median(survival_times)) if survival_times else 0.0
            }
            
            logger.info(f"Risultati analisi combinata: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in combined analysis: {e}")
            raise

    def _calculate_drawdowns(self, simulations_array: np.ndarray) -> np.ndarray:
        """Calculate drawdowns for all simulations."""
        rolling_max = np.maximum.accumulate(simulations_array, axis=1)
        drawdowns = np.zeros_like(simulations_array)
        mask = rolling_max != 0
        drawdowns[mask] = (simulations_array[mask] - rolling_max[mask]) / rolling_max[mask]
        return drawdowns

    def _calculate_drawdown_duration(self, drawdowns: np.ndarray) -> float:
        """Calculate average duration of significant drawdowns."""
        threshold = float(self.config.drawdown_threshold)
        durations = []
        
        for sim in drawdowns:
            in_drawdown = False
            current_duration = 0
            
            for dd in sim:
                if dd < -threshold:
                    in_drawdown = True
                    current_duration += 1
                elif in_drawdown:
                    durations.append(current_duration)
                    in_drawdown = False
                    current_duration = 0
                    
        return np.mean(durations) if durations else 0.0

    def _calculate_average_recovery_time(self, simulations_array: np.ndarray) -> Optional[float]:
        try:
            recovery_times = []
            threshold = float(self.config.drawdown_threshold)
            recovery_threshold = 0.85
            
            for simulation in simulations_array:
                peak = simulation[0]
                max_drawdown_point = 0
                drawdown_value = 0
                recovery_start = 0
                
                # Trova il punto di massimo drawdown e inizio recupero
                for t, value in enumerate(simulation[1:], 1):
                    current_drawdown = (value - peak) / peak
                    if current_drawdown < drawdown_value:
                        drawdown_value = current_drawdown
                        max_drawdown_point = t
                    elif t > max_drawdown_point and value > simulation[t-1]:
                        # Identifica l'inizio del recupero quando il valore inizia a risalire
                        recovery_start = t
                        break
                
                # Se abbiamo trovato un drawdown significativo
                if drawdown_value < -threshold and recovery_start > 0:
                    # Cerca il punto di recupero partendo dall'inizio del recupero
                    for t, value in enumerate(simulation[recovery_start:], recovery_start):
                        if value >= recovery_threshold * peak:
                            recovery_time = t - recovery_start
                            if recovery_time > 0:
                                recovery_times.append(recovery_time)
                            break
            
            return float(np.mean(recovery_times)) if recovery_times else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating recovery time: {e}")
            return 0.0

    def _count_underwater_periods(self, drawdowns: np.ndarray) -> int:
        """
        Count number of distinct underwater periods across all simulations.
        Returns the AVERAGE number of underwater periods per simulation.
        """
        try:
            threshold = float(self.config.drawdown_threshold)
            total_periods = 0
            num_simulations = len(drawdowns)
            
            # Per ogni simulazione
            for sim in drawdowns:
                in_drawdown = False
                periods_this_sim = 0
                
                # Per ogni punto temporale
                for dd in sim:
                    if not in_drawdown and dd < -threshold:
                        # Nuovo periodo underwater
                        periods_this_sim += 1
                        in_drawdown = True
                    elif in_drawdown and dd >= -threshold:
                        # Fine del periodo underwater
                        in_drawdown = False
                
                total_periods += periods_this_sim
            
            # Calcola la media di periodi per simulazione
            avg_periods = total_periods / num_simulations if num_simulations > 0 else 0
            # Arrotonda al numero intero più vicino
            return int(round(avg_periods))
            
        except Exception as e:
            logger.error(f"Error calculating underwater periods: {e}")
            return 0

    def _calculate_stress_resilience(self, simulations_array: np.ndarray, returns: np.ndarray) -> float:
        """Calculate stress resilience score (0-100)."""
        try:
            # Calcolo velocità di recupero
            drawdowns = self._calculate_drawdowns(simulations_array)
            recovery_speed = 1 / (self._calculate_average_recovery_time(simulations_array) + 1)
            
            # Calcolo stabilità durante stress
            stress_returns = returns[returns < np.percentile(returns, 25)]
            stability = 1 - (np.std(stress_returns) / np.std(returns)) if len(stress_returns) > 0 else 1
            
            # Calcolo frequenza rimbalzi
            rebounds = np.sum(np.where(returns[:-1] < 0, returns[1:] > 0, 0)) / max(np.sum(returns[:-1] < 0), 1)
            
            # Combinazione dei fattori
            score = (0.4 * recovery_speed + 0.4 * stability + 0.2 * rebounds) * 100
            return min(max(score, 0), 100)
        except Exception as e:
            logger.error(f"Error calculating stress resilience: {e}")
            return 0.0

    def _calculate_black_swan_impact(self, simulations_array: np.ndarray) -> float:
        try:
            # Evita divisione per zero
            denominator = simulations_array[:, :-1]
            denominator = np.where(denominator == 0, np.nan, denominator)
            
            returns = np.diff(simulations_array, axis=1) / denominator
            # Rimuovi i NaN prima di calcolare il percentile
            returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                logger.warning("No valid returns for black swan calculation")
                return 0.0
                
            worst_events = np.percentile(returns, 1)
            return float(worst_events)
            
        except Exception as e:
            logger.error(f"Error calculating black swan impact: {e}")
            return 0.0

    def _calculate_worst_case_recovery(self, drawdowns: np.ndarray) -> float:
        """Calculate worst case recovery time."""
        try:
            threshold = float(self.config.drawdown_threshold)
            recovery_times = []
            
            for sim in drawdowns:
                if np.min(sim) < -threshold:
                    worst_idx = np.argmin(sim)
                    recovery_indices = np.where(sim[worst_idx:] >= 0)[0]
                    if len(recovery_indices) > 0:
                        recovery_times.append(recovery_indices[0])
                        
            return float(np.max(recovery_times)) if recovery_times else float('inf')
        except Exception as e:
            logger.error(f"Error calculating worst case recovery: {e}")
            return float('inf')

    def _calculate_return_stability(self, returns: np.ndarray) -> float:
        """Calculate return stability index."""
        try:
            rolling_std = np.std(returns, axis=1)
            return float(1 - np.std(rolling_std) / np.mean(rolling_std))
        except Exception as e:
            logger.error(f"Error calculating return stability: {e}")
            return 0.0

    def analyze_withdrawal_sustainability(self, simulations_array: np.ndarray) -> Dict:
        """Analyze withdrawal sustainability of the portfolio."""
        try:
            # Calcola gli indici delle fasi
            maintenance_start_idx = self.config.accumulation_years
            withdrawal_start_idx = (self.config.accumulation_years + 
                                self.config.maintenance_years)
            
            result = {}
            
            # Analisi fase di mantenimento (se presente)
            if self.config.maintenance_years > 0:
                maintenance_phase = simulations_array[:, 
                                                maintenance_start_idx:withdrawal_start_idx]
                
                maintenance_analysis = self._analyze_single_phase(
                    maintenance_phase,
                    self.config.maintenance_withdrawal,
                    'maintenance'
                )
                result['maintenance_phase_analysis'] = maintenance_analysis
            
            # Analisi fase di prelievo
            withdrawal_phase = simulations_array[:, withdrawal_start_idx:]
            withdrawal_analysis = self._analyze_single_phase(
                withdrawal_phase,
                self.config.withdrawal_amount,
                'withdrawal'
            )
            
            # Mantiene la compatibilità con il codice esistente
            result.update({
                'median_portfolio_life': withdrawal_analysis['median_portfolio_life'],
                'failure_probabilities': withdrawal_analysis['failure_probabilities']
            })
            
            # Analisi combinata (se presente)
            if self.config.maintenance_years > 0:
                combined_phase = simulations_array[:, maintenance_start_idx:]
                combined_analysis = self._analyze_combined_phases(
                    combined_phase,
                    self.config.maintenance_years
                )
                result['combined_phase_analysis'] = combined_analysis
            
            return result
                
        except Exception as e:
            logger.error(f"Error in withdrawal analysis: {e}")
            raise

    def _analyze_single_phase(self, phase_data: np.ndarray, target_amount: Decimal, phase_type: str) -> Dict:
        """Analyze a single phase (maintenance or withdrawal)."""
        try:
            # Verifica se i dati sono vuoti
            if phase_data.size == 0 or len(phase_data.shape) == 0:
                return {
                    'phase_type': phase_type,
                    'median_portfolio_life': 0.0,
                    'failure_probabilities': [],  # Nota: chiave consistente
                    'total_failure_probability': 0.0,
                    'early_failure_probability': 0.0,
                    'target_amount': float(target_amount)
                }
                
            num_simulations = len(phase_data)
            failure_probs = []  # Lista per le probabilità di fallimento
            portfolio_lives = []
            
            # Per ogni simulazione
            for simulation in phase_data:
                if len(simulation) > 0:
                    if simulation[-1] > 0:
                        portfolio_lives.append(len(simulation))
                    else:
                        zero_points = np.where(simulation <= 0)[0]
                        if len(zero_points) > 0:
                            portfolio_lives.append(zero_points[0])
                        else:
                            portfolio_lives.append(len(simulation))
            
            # Calcola probabilità di fallimento per anno
            active_simulations = num_simulations
            for year in range(phase_data.shape[1]):
                failures_this_year = np.sum(
                    (phase_data[:, year] <= 0) & 
                    (phase_data[:, year-1] > 0 if year > 0 else np.full(phase_data.shape[0], True))
                )
                
                failure_prob = failures_this_year / active_simulations if active_simulations > 0 else 0.0
                failure_probs.append(float(failure_prob))
                active_simulations -= failures_this_year
            
            # Calcola metriche finali
            median_life = float(np.median(portfolio_lives)) if portfolio_lives else 0.0
            total_failures = sum(failure_probs)
            early_failures = sum(failure_probs[:min(5, len(failure_probs))])
            
            return {
                'phase_type': phase_type,
                'median_portfolio_life': median_life,
                'failure_probabilities': failure_probs,  # Nota: chiave consistente
                'total_failure_probability': float(total_failures),
                'early_failure_probability': float(early_failures),
                'target_amount': float(target_amount)
            }
                
        except Exception as e:
            logger.error(f"Error in single phase analysis: {e}")
            return {
                'phase_type': phase_type,
                'median_portfolio_life': 0.0,
                'failure_probabilities': [],  # Nota: chiave consistente
                'total_failure_probability': 0.0,
                'early_failure_probability': 0.0,
                'target_amount': float(target_amount)
            }

    def _analyze_combined_phases(self, combined_data: np.ndarray, 
                            maintenance_years: int) -> Dict:
        """Analyze the combined maintenance and withdrawal phases."""
        try:
            num_simulations = len(combined_data)
            total_years = combined_data.shape[1]
            
            # Traccia quando avvengono i primi fallimenti
            first_failures = {}
            for sim_idx in range(num_simulations):
                failure_points = np.where(combined_data[sim_idx, :] <= 0)[0]
                if len(failure_points) > 0:
                    first_failures[sim_idx] = failure_points[0] + 1
            
            # Calcola la distribuzione dei primi fallimenti
            failure_distribution = {}
            for year in range(1, total_years + 1):
                count = sum(1 for v in first_failures.values() if v == year)
                failure_distribution[year] = count / num_simulations
            
            # Calcola il tempo mediano di sopravvivenza
            survival_times = []
            for sim in combined_data:
                if sim[-1] > 0:
                    survival_times.append(total_years)
                else:
                    zero_point = np.where(sim <= 0)[0][0]
                    survival_times.append(zero_point)
            
            median_survival = float(np.median(survival_times))
            total_failure_prob = len(first_failures) / num_simulations
            
            return {
                'total_failure_probability': total_failure_prob,
                'median_survival_time': median_survival,
                'first_failure_year_distribution': failure_distribution,
                'maintenance_to_withdrawal_transition': {
                    'success_rate': float(np.mean(combined_data[:, maintenance_years-1] > 0)),
                    'portfolio_values': {
                        'mean': float(np.mean(combined_data[:, maintenance_years-1])),
                        'median': float(np.median(combined_data[:, maintenance_years-1])),
                        'std': float(np.std(combined_data[:, maintenance_years-1]))
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error in combined phase analysis: {e}")
            raise