from dataclasses import dataclass
from typing import Dict, List
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ReportFormatter:
    @staticmethod
    def format_report(report: Dict) -> str:
        """Formatta il report completo."""
        try:
            output = []
            
            # Report Header
            output.append("\nReport Dettagliato:")
            output.append(f"Data generazione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            output.append("\nMETRICHE DI RISCHIO:")
            
            # Risk Metrics
            risk_metrics = report['risk_metrics']
            output.append(f"- Risk Score: {risk_metrics.risk_score:.1f}/100")
            output.append(f"- Maximum Drawdown: {risk_metrics.max_drawdown:.2%}")
            output.append(f"- Average Drawdown: {risk_metrics.avg_drawdown:.2%}")
            output.append(f"- Recovery Time: {risk_metrics.recovery_time:.1f} anni")
            output.append(f"- Stress Resilience: {risk_metrics.stress_resilience_score:.1f}/100")
            output.append(f"- Return Stability: {risk_metrics.return_stability_index:.2f}")
            output.append(f"- Success Rate: {risk_metrics.success_rate:.2%}")

            # Withdrawal Analysis
            output.append("\nANALISI DEI PRELIEVI:")
            withdrawal = report['withdrawal_analysis']
            
            if 'maintenance_phase_analysis' in withdrawal:
                maint = withdrawal['maintenance_phase_analysis']
                output.append("\nFase di Mantenimento:")
                output.append(f"- Vita mediana portafoglio: {maint['median_portfolio_life']:.1f} anni")
                output.append(f"- Probabilità fallimento: {maint['total_failure_probability']:.2%}")
            
            output.append("\nFase di Prelievo:")
            output.append(f"- Vita mediana portafoglio: {withdrawal['median_portfolio_life']:.1f} anni")
            
            if len(withdrawal.get('failure_probabilities', [])) > 0:  # Usa la nuova chiave
                max_prob = max(withdrawal['failure_probabilities'])
                max_year = withdrawal['failure_probabilities'].index(max_prob) + 1
                output.append(f"- Massima probabilità di fallimento: {max_prob:.2%} (Anno {max_year})")
            
            # Stress Test
            output.append("\nSTRESS TEST:")
            stress = report['stress_test']
            output.append(f"- Impatto inflazione elevata: {stress['high_inflation']['portfolio_impact']:.2%}")
            output.append(f"- Impatto crollo di mercato: {stress['market_crash']['portfolio_impact']:.2%}")
            output.append(f"- Impatto bear market: {stress['prolonged_bear']['portfolio_impact']:.2%}")
            output.append(f"- Impatto scenario combinato: {stress['combined_stress']['portfolio_impact']:.2%}")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Errore nella formattazione del report: {e}")
            return f"Errore nella formattazione del report: {str(e)}"

    @staticmethod
    def get_interpretation(report: Dict) -> str:
        """Genera l'interpretazione del report."""
        try:
            risk_metrics = report['risk_metrics']
            withdrawal = report['withdrawal_analysis']
            
            output = ["=== INTERPRETAZIONE REPORT ==="]
            
            # Interpretazione Resilienza
            resilience_score = risk_metrics.stress_resilience_score
            if resilience_score >= 80:
                resilience_msg = "eccellente"
            elif resilience_score >= 60:
                resilience_msg = "buona"
            elif resilience_score >= 40:
                resilience_msg = "moderata"
            else:
                resilience_msg = "debole"
                
            output.append(f"RESILIENZA: {resilience_msg.title()}")
            output.append(f"- Il portafoglio mostra una resilienza {resilience_msg}")
            output.append(f"- Capacità di recupero da eventi estremi: {risk_metrics.worst_case_recovery:.1f} anni")
            
            # Interpretazione Profilo di Rischio
            output.append("\nPROFILO DI RISCHIO:")
            if risk_metrics.max_drawdown > -0.3:
                output.append("- Drawdown: elevato rischio di perdite significative")
            elif risk_metrics.max_drawdown > -0.2:
                output.append("- Drawdown: rischio di perdite moderato")
            else:
                output.append("- Drawdown: buon controllo delle perdite")
            output.append(f"- Tempo medio di recupero: {risk_metrics.recovery_time:.1f} anni")
            
            # Interpretazione Stabilità
            output.append("\nSTABILITÀ:")
            output.append(f"- Il portafoglio è {resilience_msg}")
            output.append(f"- Probabilità di successo: {risk_metrics.success_rate:.1%}")
            
            # Analisi Sostenibilità
            output.append("\nANALISI SOSTENIBILITÀ:")
            
            # Usa la nuova chiave per le probabilità di fallimento
            failure_probabilities = withdrawal.get('failure_probabilities', [])
            if failure_probabilities:
                failure_end = failure_probabilities[-1]
                if failure_end < 0.01:
                    output.append("- Bassa probabilità di fallimento")
                else:
                    output.append("- Moderata probabilità di fallimento")
                output.append(f"  Probabilità: {failure_end:.1%}")
            
            output.append(f"- Vita mediana portafoglio: {withdrawal['median_portfolio_life']:.1f} anni")
            
            return "\n".join(output)
            
        except Exception as e:
            logger.error(f"Errore nell'interpretazione: {e}")
            return f"Errore nell'interpretazione: {str(e)}"