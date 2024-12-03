from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, List, Dict, Union

@dataclass
class RiskMetrics:
    """Metriche di rischio avanzate per l'analisi del portafoglio."""
    
    # Risk Score composito
    risk_score: float
    
    # Metriche di drawdown
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: float
    recovery_time: float
    underwater_periods: int
    
    # Metriche di tail risk
    tail_loss_probability: float
    expected_tail_loss: float
    skewness: float
    kurtosis: float
    
    # Metriche di stress e resilienza
    stress_resilience_score: float
    black_swan_impact: float
    worst_case_recovery: float
    
    # Metriche di stabilità
    return_stability_index: float
    success_rate: float
    
    # Metriche dettagliate per il success rate
    success_stats: Dict[str, Union[int, float]] = field(default_factory=dict)
    
    # Metriche di valore finale del portafoglio
    portfolio_metrics: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Converte le metriche in un dizionario."""
        return {
            'risk_score': self.risk_score,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'drawdown_duration': self.drawdown_duration,
            'recovery_time': self.recovery_time,
            'underwater_periods': self.underwater_periods,
            'tail_loss_probability': self.tail_loss_probability,
            'expected_tail_loss': self.expected_tail_loss,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'stress_resilience_score': self.stress_resilience_score,
            'black_swan_impact': self.black_swan_impact,
            'worst_case_recovery': self.worst_case_recovery,
            'return_stability_index': self.return_stability_index,
            'success_rate': self.success_rate,
            'success_stats': self.success_stats,
            'portfolio_metrics': self.portfolio_metrics
        }
        
    def get_success_details(self) -> str:
        """Fornisce una descrizione dettagliata delle statistiche di successo."""
        if not self.success_stats:
            return "Statistiche di successo non disponibili"
            
        return f"""
        Analisi Dettagliata del Success Rate:
        - Simulazioni Totali: {self.success_stats['total_simulations']:,}
        - Simulazioni di Successo: {self.success_stats['successful_simulations']:,}
        - Success Rate Fase Mantenimento: {self.success_stats['maintenance_success_rate']:.1%}
        - Success Rate Fase Prelievo: {self.success_stats['withdrawal_success_rate']:.1%}
        - Success Rate Complessivo: {self.success_stats['overall_success_rate']:.1%}
        """