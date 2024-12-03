from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
import numpy as np
from ..models.risk_metrics import RiskMetrics
from ..models.simulation_config import SimulationConfig

logger = logging.getLogger(__name__)

@dataclass
class ReportSection:
    """Container for a report section."""
    title: str
    content: Dict
    interpretation: Optional[str] = None

@dataclass
class ReportGenerator:
    """Generator for portfolio analysis reports."""
    config: SimulationConfig
    num_simulations: int

    def generate_complete_report(self, report_data: Dict) -> List[ReportSection]:
        """Generate a complete report with all sections."""
        try:
            sections = []
            
            # Risk Metrics Section
            if 'risk_metrics' in report_data:
                sections.append(self.generate_risk_metrics_section(report_data['risk_metrics']))
            
            # Withdrawal Analysis Section
            if 'withdrawal_analysis' in report_data:
                sections.append(self.generate_withdrawal_section(report_data['withdrawal_analysis']))
            
            # Stress Test Section
            if 'stress_test' in report_data:
                sections.append(self.generate_stress_test_section(report_data['stress_test']))
            
            return sections
            
        except Exception as e:
            logger.error(f"Error generating complete report: {e}")
            raise

    def generate_risk_metrics_section(self, metrics: RiskMetrics) -> ReportSection:
        """Genera la sezione delle metriche di rischio."""
        content = {
            'risk_score': f"{metrics.risk_score:.1f}/100",
            'max_drawdown': f"{metrics.max_drawdown:.2%}",
            'avg_drawdown': f"{metrics.avg_drawdown:.2%}",
            'drawdown_duration': f"{metrics.drawdown_duration:.1f} anni",
            'recovery_time': f"{metrics.recovery_time:.1f} anni",
            'underwater_periods': str(metrics.underwater_periods),
            'tail_loss_probability': f"{metrics.tail_loss_probability:.2%}",
            'expected_tail_loss': f"{metrics.expected_tail_loss:.2%}",
            'skewness': f"{metrics.skewness:.2f}",
            'kurtosis': f"{metrics.kurtosis:.2f}",
            'stress_resilience_score': f"{metrics.stress_resilience_score:.1f}/100",
            'black_swan_impact': f"{metrics.black_swan_impact:.2%}",
            'worst_case_recovery': f"{metrics.worst_case_recovery:.1f} anni",
            'return_stability_index': f"{metrics.return_stability_index:.2f}",
            'success_rate': f"{metrics.success_rate:.2%}"
        }

        interpretation = self._interpret_risk_metrics(metrics)

        return ReportSection(
            title="Metriche di Rischio",
            content=content,
            interpretation=interpretation
        )

    def generate_withdrawal_section(self, withdrawal_data: Dict) -> ReportSection:
        """Generate withdrawal analysis section of the report."""
        try:
            content = {}
            
            # Fase di mantenimento (se presente)
            if 'maintenance_phase_analysis' in withdrawal_data:
                maint = withdrawal_data['maintenance_phase_analysis']
                content['maintenance'] = {
                    'median_life': f"{maint.get('median_portfolio_life', 0):.1f}",
                    'failure_probabilities': [
                        f"{prob:.1%}" for prob in maint.get('failure_probabilities', [])
                    ],
                    'total_failure_prob': f"{maint.get('total_failure_probability', 0):.1%}",
                    'early_failure_prob': f"{maint.get('early_failure_probability', 0):.1%}",
                    'target_amount': f"EUR {maint.get('target_amount', 0):,.2f}"
                }
            
            # Fase di prelievo
            content['withdrawal'] = {
                'median_life': f"{withdrawal_data.get('median_portfolio_life', 0):.1f}",
                'failure_probabilities': [
                    f"{prob:.1%}" for prob in withdrawal_data.get('failure_probabilities', [])
                ]
            }
            
            # Analisi combinata (se presente)
            if 'combined_phase_analysis' in withdrawal_data:
                combined = withdrawal_data['combined_phase_analysis']
                content['combined'] = {
                    'total_failure_prob': f"{combined.get('total_failure_probability', 0):.1%}",
                    'median_survival': f"{combined.get('median_survival_time', 0):.1f}",
                    'transition_success': f"{combined.get('maintenance_to_withdrawal_transition', {}).get('success_rate', 0):.1%}",
                    'first_failure_distribution': {
                        year: f"{prob:.1%}"
                        for year, prob in combined.get('first_failure_year_distribution', {}).items()
                    }
                }

            interpretation = self._interpret_withdrawal_analysis(withdrawal_data)
            
            return ReportSection(
                title="Analisi Sostenibilità Prelievi",
                content=content,
                interpretation=interpretation
            )
                
        except Exception as e:
            logger.error(f"Error generating withdrawal section: {e}")
            raise

    def generate_stress_test_section(self, stress_data: Dict) -> ReportSection:
        """Generate stress test section of the report."""
        try:
            content = {
                scenario: f"{results['portfolio_impact']:.2%}"
                for scenario, results in stress_data.items()
            }

            interpretation = self._interpret_stress_test(stress_data)
            
            return ReportSection(
                title="Analisi degli Scenari di Stress",
                content=content,
                interpretation=interpretation
            )
        except Exception as e:
            logger.error(f"Error generating stress test section: {e}")
            raise

    def _interpret_withdrawal_analysis(self, withdrawal_data: Dict) -> str:
        """Generate interpretation for withdrawal analysis."""
        try:
            interpretation = []
            
            # Analisi fase di mantenimento
            if 'maintenance_phase_analysis' in withdrawal_data:
                maint = withdrawal_data['maintenance_phase_analysis']
                interpretation.append("FASE DI MANTENIMENTO:")
                
                # Valuta la probabilità di fallimento
                failure_prob = maint.get('total_failure_probability', 0)
                if failure_prob > 0.15:
                    interpretation.append(f"- CRITICITÀ: Alta probabilità di fallimento ({failure_prob:.1%})")
                elif failure_prob > 0.10:
                    interpretation.append(f"- ATTENZIONE: Moderata probabilità di fallimento ({failure_prob:.1%})")
                else:
                    interpretation.append(f"- Bassa probabilità di fallimento ({failure_prob:.1%})")
                
                # Valuta i fallimenti precoci
                early_prob = maint.get('early_failure_probability', 0)
                if early_prob > 0.05:
                    interpretation.append(f"- RISCHIO: Significativa probabilità di fallimento precoce ({early_prob:.1%})")
                
                interpretation.append(f"- Vita mediana del portafoglio: {maint.get('median_portfolio_life', 0):.1f} anni")
                interpretation.append("")  # Linea vuota per separazione
            
            # Analisi fase di prelievo
            interpretation.append("FASE DI PRELIEVO:")
            withdrawal_life = withdrawal_data.get('median_portfolio_life', 0)
            target_years = self.config.withdrawal_years
            
            if withdrawal_life < target_years:
                interpretation.append(
                    f"- ATTENZIONE: Vita mediana ({withdrawal_life:.1f} anni) "
                    f"inferiore all'obiettivo ({target_years} anni)"
                )
            else:
                interpretation.append(
                    f"- Vita mediana del portafoglio adeguata: {withdrawal_life:.1f} anni"
                )
            
            # Analisi combinata
            if 'combined_phase_analysis' in withdrawal_data:
                combined = withdrawal_data['combined_phase_analysis']
                interpretation.append("\nANALISI COMBINATA:")
                
                total_prob = combined.get('total_failure_probability', 0)
                if total_prob > 0.15:
                    interpretation.append(f"- CRITICITÀ: Probabilità totale di fallimento {total_prob:.1%}")
                else:
                    interpretation.append(f"- Probabilità totale di fallimento: {total_prob:.1%}")
                
                interpretation.append(
                    f"- Tempo mediano di sopravvivenza: {combined.get('median_survival_time', 0):.1f} anni"
                )
                
                # Analisi transizione tra fasi
                transition = combined.get('maintenance_to_withdrawal_transition', {})
                transition_rate = transition.get('success_rate', 0)
                interpretation.append(
                    f"- Tasso di successo transizione mantenimento-prelievo: {transition_rate:.1%}"
                )
            
            # Aggiungi raccomandazioni se necessario
            interpretation.append("\nRACCOMANDAZIONI:")
            recommendations = []
            
            if 'maintenance_phase_analysis' in withdrawal_data:
                if withdrawal_data['maintenance_phase_analysis'].get('total_failure_probability', 0) > 0.1:
                    recommendations.append("- Riconsiderare la strategia di mantenimento")
            
            if withdrawal_life < target_years:
                recommendations.append("- Valutare una riduzione dei prelievi annuali")
            
            if not recommendations:
                recommendations.append("- Strategia attuale adeguata agli obiettivi")
            
            interpretation.extend(recommendations)
            
            return "\n".join(interpretation)
            
        except Exception as e:
            logger.error(f"Error in withdrawal interpretation: {e}")
            return "Errore nell'interpretazione dell'analisi dei prelievi."

    def _interpret_stress_test(self, stress_data: Dict) -> str:
        """Generate interpretation for stress test results."""
        try:
            interpretation = []
            
            # Analyze worst case
            worst_impact = min(
                results['portfolio_impact'] 
                for results in stress_data.values()
            )
            
            if worst_impact < -0.5:
                interpretation.append(
                    f"CRITICITÀ: Impatto severo negli scenari peggiori ({worst_impact:.1%})"
                )
            elif worst_impact < -0.3:
                interpretation.append(
                    f"ATTENZIONE: Impatto significativo negli scenari peggiori ({worst_impact:.1%})"
                )
                
            # Analyze specific scenarios
            if stress_data['high_inflation']['portfolio_impact'] < -0.3:
                interpretation.append("Elevata vulnerabilità all'inflazione")
            if stress_data['market_crash']['portfolio_impact'] < -0.4:
                interpretation.append("Significativa esposizione al rischio di mercato")
                
            # Add recommendations
            if interpretation:
                interpretation.append("\nRACCOMANDAZIONI:")
                if stress_data['high_inflation']['portfolio_impact'] < -0.3:
                    interpretation.append("- Considerare strumenti di protezione dall'inflazione")
                if stress_data['market_crash']['portfolio_impact'] < -0.4:
                    interpretation.append("- Valutare strategie di hedging")
                    
            return "\n".join(interpretation)
        except Exception as e:
            logger.error(f"Error in stress test interpretation: {e}")
            return "Errore nell'interpretazione degli stress test."

    def _interpret_risk_metrics(self, metrics: RiskMetrics) -> str:
        """Interpreta le metriche di rischio."""
        try:
            interpretation = []

            # Interpretazione del Success Rate
            success_stats = metrics.success_stats
            interpretation.append("ANALISI DEL SUCCESS RATE:")
            interpretation.append(f"- Simulazioni analizzate: {success_stats['total_simulations']:,}")
            interpretation.append(f"- Simulazioni riuscite: {success_stats['successful_simulations']:,}")
            
            maintenance_success_rate = success_stats['maintenance_success_rate']
            if maintenance_success_rate < 0.95:
                interpretation.append(
                    f"- ATTENZIONE: Significativa probabilità di fallimento "
                    f"durante la fase di mantenimento "
                    f"({1 - maintenance_success_rate:.1%})"
                )
                
            withdrawal_success_rate = success_stats['withdrawal_success_rate']
            if withdrawal_success_rate < 0.95:
                interpretation.append(
                    f"- ATTENZIONE: Significativa probabilità di fallimento "
                    f"durante la fase di prelievo "
                    f"({1 - withdrawal_success_rate:.1%})"
                )
            
            overall_success = success_stats['overall_success_rate']
            if overall_success >= 0.95:
                interpretation.append(f"- Piano finanziario robusto ({overall_success:.1%} success rate)")
            elif overall_success >= 0.85:
                interpretation.append(f"- Piano finanziario accettabile ({overall_success:.1%} success rate)")
            else:
                interpretation.append(f"- CRITICITÀ: Piano finanziario a rischio ({overall_success:.1%} success rate)")

            # Interpretazione Risk Score
            risk_score = metrics.risk_score
            if risk_score >= 80:
                risk_msg = "eccellente"
            elif risk_score >= 60:
                risk_msg = "buono"
            elif risk_score >= 40:
                risk_msg = "moderato"
            else:
                risk_msg = "critico"

            interpretation.append(f"\nRISK SCORE: {risk_msg.title()} ({risk_score:.1f}/100)")

            # Interpretazione Resilienza
            resilience_score = metrics.stress_resilience_score
            if resilience_score >= 80:
                resilience_msg = "eccellente"
            elif resilience_score >= 60:
                resilience_msg = "buona"
            elif resilience_score >= 40:
                resilience_msg = "moderata"
            else:
                resilience_msg = "debole"

            interpretation.append(f"\nRESILIENZA: {resilience_msg.title()}")
            interpretation.append(f"- Il portafoglio mostra una resilienza {resilience_msg}")
            interpretation.append(f"- Capacità di recupero da eventi estremi: {metrics.worst_case_recovery:.1f} anni")

            # Interpretazione Profilo di Rischio
            if metrics.max_drawdown > -0.3:
                dd_msg = "elevato rischio di perdite significative"
            elif metrics.max_drawdown > -0.2:
                dd_msg = "rischio di perdite moderato"
            else:
                dd_msg = "buon controllo delle perdite"

            interpretation.append(f"\nPROFILO DI RISCHIO:")
            interpretation.append(f"- Drawdown: {dd_msg}")
            interpretation.append(f"- Tempo medio di recupero: {metrics.recovery_time:.1f} anni")

            # Interpretazione Stabilità
            stability = metrics.return_stability_index
            if stability > 0.7:
                stab_msg = "molto stabile"
            elif stability > 0.5:
                stab_msg = "moderatamente stabile"
            else:
                stab_msg = "volatile"

            interpretation.append(f"\nSTABILITÀ:")
            interpretation.append(f"- Il portafoglio è {stab_msg}")
            interpretation.append(f"- Probabilità di successo: {metrics.success_rate:.1%}")

            return "\n".join(interpretation)
            
        except Exception as e:
            logger.error(f"Error interpreting risk metrics: {e}")
            return "Errore nell'interpretazione delle metriche di rischio."