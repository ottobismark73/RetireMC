from fpdf import FPDF
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from ..models.simulation_config import SimulationConfig
from ..models.risk_metrics import RiskMetrics
from .report_generator import ReportGenerator, ReportSection

try:
    import seaborn as sns
    sns.set_style('whitegrid')
except ImportError:
    logging.warning("Seaborn non installato. Verranno usati stili di default.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UTF8PDF(FPDF):
    def __init__(self, num_simulations):
        super().__init__()
        self.num_simulations = num_simulations
        self.set_auto_page_break(auto=True, margin=15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}/{{nb}}', 0, 0, 'R')

class PortfolioPDFExporter:    # Cambiato qui da SimplePortfolioPDFExporter a PortfolioPDFExporter
    """Classe per l'esportazione dei report in PDF."""
    
    def __init__(self, config: SimulationConfig, num_simulations: int):
        """Inizializza l'esportatore PDF."""
        self.config = config
        self.num_simulations = num_simulations
        self.pdf = UTF8PDF(num_simulations)
        self.simulation_details = None
        self.report_data = None
        self.simulations = None  # Aggiunto questo attributo
        self.report_generator = ReportGenerator(config, num_simulations)

    def export(self, report_data, simulation_details=None, filename="portfolio_report.pdf"):
        """Metodo pubblico per l'esportazione del report."""
        try:
            self.report_data = report_data
            self.simulation_details = simulation_details
            
            # Aggiungi i dati delle simulazioni al report_data se non presenti
            if isinstance(report_data, dict) and 'simulations' not in report_data:
                logger.warning("Dati simulazioni non trovati nel report, verranno aggiunti")
                if hasattr(self.report_data, 'risk_metrics'):
                    # Estrai i dati delle simulazioni dalle metriche di rischio se possibile
                    self.report_data['simulations'] = np.array(self.report_data['risk_metrics'].simulations)
            
            # Genera le sezioni usando il ReportGenerator
            risk_section = self.report_generator.generate_risk_metrics_section(report_data['risk_metrics'])
            withdrawal_section = self.report_generator.generate_withdrawal_section(report_data['withdrawal_analysis'])
            stress_section = self.report_generator.generate_stress_test_section(report_data['stress_test'])
            
            # Genera il PDF
            self._generate_pdf(filename, [risk_section, withdrawal_section, stress_section])
            
            logger.info(f"Report PDF generato con successo: {filename}")
        except Exception as e:
            logger.error(f"Errore nell'esportazione del PDF: {e}")
            raise

    def _generate_pdf(self, filename, sections):
        """Metodo interno per la generazione del PDF."""
        try:
            if not self.report_data:
                raise ValueError("Nessun dato di report fornito")

            # Inizializzazione
            self.pdf.alias_nb_pages()
            self.pdf.add_page()

            # Sezioni fisse iniziali
            self._add_header()
            self._add_config_details()
            
            # Aggiungi la sezione di sintesi
            self._add_monte_carlo_summary()

            # Sezioni principali
            for section in sections:
                # Nuova pagina per ogni sezione principale
                self.pdf.add_page()
                self._add_section_title(section.title)
                
                if section.title == "Metriche di Rischio":
                    self._add_risk_metrics(section.content)
                    if section.interpretation:
                        self._add_interpretation_box(section.interpretation)
                        
                elif section.title == "Analisi Sostenibilità Prelievi":
                    self._add_withdrawal_analysis(section.content)
                    if section.interpretation:
                        self._add_interpretation_box(section.interpretation)

            # Sezione grafici
            self.pdf.add_page()
            self._add_section_title("Analisi Grafica del Portafoglio")
            self._add_timeline_chart(self.pdf.w - 40, 100)
            self._add_distribution_chart(self.pdf.w - 40, 100)

            # Sezione stress test (spostata qui)
            self.pdf.add_page()
            self._add_section_title("Analisi degli Scenari di Stress")
            self._add_stress_test_section()

            # Dettagli prima simulazione (se disponibili)
            if self.simulation_details:
                self.pdf.add_page()
                self._add_section_title("Dettagli Prima Simulazione")
                self._add_simulation_details()

            # Salvataggio finale
            self.pdf.output(filename, 'F')

        except Exception as e:
            logger.error(f"Errore nella generazione del PDF: {e}")
            raise

    def _add_stress_test_section(self):
        """Aggiunge la sezione degli stress test con grafico e interpretazione."""
        try:
            # NON forzare una nuova pagina, ma controlla dove siamo
            current_y = self.pdf.get_y()
            
            # Solo se siamo verso la fine della pagina, iniziamo una nuova
            if current_y > (self.pdf.h * 0.75):
                self.pdf.add_page()
                current_y = self.pdf.get_y()
            
            # Crea il grafico degli stress test
            results = self.report_data['stress_test']
            scenarios = {
                'Inflazione\nElevata': results['high_inflation']['portfolio_impact'] * 100,
                'Crollo di\nMercato': results['market_crash']['portfolio_impact'] * 100,
                'Bear Market\nProlungato': results['prolonged_bear']['portfolio_impact'] * 100,
                'Stress\nCombinato': results['combined_stress']['portfolio_impact'] * 100
            }

            # Crea e salva il grafico
            plt.figure(figsize=(12, 6))
            sorted_items = sorted(scenarios.items(), key=lambda x: x[1], reverse=True)
            names = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]

            y_pos = np.arange(len(names))
            bars = plt.barh(y_pos, values,
                        color=['#ef4444' if v <= -50 else '#f97316' if v <= -30
                                else '#eab308' for v in values],
                        alpha=0.8)

            plt.title('Impatto degli Scenari di Stress sul Portafoglio', 
                    pad=20, fontsize=14, fontweight='bold')
            plt.xlabel('Impatto (%)', fontsize=12)
            plt.yticks(y_pos, names, fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.3)

            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width - 1 if width < 0 else width + 1,
                        bar.get_y() + bar.get_height()/2,
                        f'{values[i]:.1f}%',
                        va='center',
                        ha='right' if width < 0 else 'left',
                        fontweight='bold',
                        fontsize=11)

            min_value = min(values)
            plt.xlim(min_value * 1.1, max(5, -min_value * 0.1))
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)

            plt.tight_layout()
            plt.savefig('stress_test.png', dpi=300, bbox_inches='tight', 
                        facecolor='white', pad_inches=0.5)
            plt.close()

            # Calcola le dimensioni del grafico (25% dell'altezza pagina)
            page_height = self.pdf.h - self.pdf.t_margin - self.pdf.b_margin
            graph_height = page_height * 0.25
            image_width = self.pdf.w - 20

            # Memorizza la posizione Y prima di aggiungere il grafico
            before_graph_y = self.pdf.get_y()

            # Aggiungi il grafico
            self.pdf.image('stress_test.png',
                        x=(self.pdf.w - image_width) / 2,
                        y=before_graph_y,
                        w=image_width,
                        h=graph_height)

            # Sposta il cursore sotto il grafico
            after_graph_y = before_graph_y + graph_height + 5
            self.pdf.set_y(after_graph_y)

            # Controlla se c'è abbastanza spazio per l'interpretazione
            interpretation_height = 100  # stima dell'altezza necessaria
            if (self.pdf.h - after_graph_y) < interpretation_height:
                self.pdf.add_page()
                self.pdf.set_y(self.pdf.t_margin)

            # Memorizza la posizione Y prima dell'interpretazione
            before_interpretation_y = self.pdf.get_y()

            # Aggiungi l'interpretazione
            interpretation = self._create_stress_test_interpretation(results)
            self._add_formatted_stress_interpretation(interpretation)

        except Exception as e:
            logger.error(f"Errore nella creazione della sezione stress test: {e}")
            raise

    def _add_formatted_stress_interpretation(self, interpretation_text):
        """Aggiunge l'interpretazione formattata degli stress test al PDF."""
        try:
            # Calcola margini e larghezza
            margin = 10
            box_width = self.pdf.w - (2 * margin)
            
            # Calcola l'altezza necessaria
            text_lines = interpretation_text.split('\n')
            line_height = 5
            text_height = len(text_lines) * line_height + 10
            
            # Background grigio chiaro per il box
            self.pdf.set_fill_color(245, 245, 245)
            self.pdf.rect(margin, self.pdf.get_y(), box_width, text_height, 'F')

            # Titolo "INTERPRETAZIONE"
            self.pdf.set_font('Helvetica', 'B', 11)
            self.pdf.set_xy(margin + 5, self.pdf.get_y() + 5)
            self.pdf.cell(box_width - 10, 7, "INTERPRETAZIONE DEGLI STRESS TEST:", 0, 1)

            # Contenuto
            self.pdf.set_font('Helvetica', '', 10)
            for line in text_lines:
                if line.strip():
                    self.pdf.set_x(margin + 5)
                    self.pdf.cell(box_width - 10, line_height, line.strip(), 0, 1)
                else:
                    self.pdf.ln(line_height)

            # Spazio dopo il box
            self.pdf.ln(10)

        except Exception as e:
            logger.error(f"Errore nell'aggiunta dell'interpretazione: {e}")
            raise

    def _add_consolidated_interpretation_box(self, text):
        """Aggiunge un box interpretativo consolidato che mantiene tutto il testo insieme."""
        try:
            # Margini e dimensioni
            margin = 10
            box_width = self.pdf.w - (2 * margin)
            
            # Preparazione del testo
            self.pdf.set_font('Helvetica', '', 10)
            lines = []
            current_line = ""
            
            # Dividi il testo in linee che si adattano alla larghezza
            for word in text.split():
                test_line = current_line + " " + word if current_line else word
                if self.pdf.get_string_width(test_line) < (box_width - 20):
                    current_line = test_line
                else:
                    lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            # Calcola l'altezza necessaria
            line_height = 6
            text_height = (len(lines) + 2) * line_height  # +2 per il titolo e spaziatura
            
            # Verifica se c'è abbastanza spazio nella pagina corrente
            if (self.pdf.get_y() + text_height + 20) > self.pdf.h:
                self.pdf.add_page()
            
            # Disegna il box
            self.pdf.set_fill_color(245, 245, 245)
            self.pdf.rect(margin, self.pdf.get_y(), box_width, text_height, 'F')
            
            # Aggiungi il titolo
            self.pdf.set_font('Helvetica', 'B', 11)
            self.pdf.set_xy(margin + 5, self.pdf.get_y() + 5)
            self.pdf.cell(0, line_height, "INTERPRETAZIONE DEGLI STRESS TEST", 0, 1)
            
            # Aggiungi il contenuto
            self.pdf.set_font('Helvetica', '', 10)
            for line in lines:
                self.pdf.set_x(margin + 5)
                self.pdf.cell(box_width - 10, line_height, line, 0, 1)
            
            # Aggiungi spazio dopo il box
            self.pdf.ln(10)
            
        except Exception as e:
            logger.error(f"Errore nell'aggiunta del box interpretativo: {e}")
            raise

    def _create_stress_test_chart(self):
        """Crea il grafico dello stress test e lo salva come immagine."""
        try:
            # Crea la figura con dimensioni ottimizzate
            plt.style.use('default')  # Usa lo stile default invece di seaborn
            fig = plt.figure(figsize=(12, 8))
            
            # Crea due subplot affiancati con proporzioni diverse
            gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.15)
            ax_bars = fig.add_subplot(gs[0])  # Grafico principale
            ax_info = fig.add_subplot(gs[1])  # Panel informativo
            
            # Ottieni i risultati degli stress test
            results = self.report_data['stress_test']
            
            # Prepara i dati per il grafico
            scenarios = {
                'Inflazione\nElevata': results['high_inflation']['portfolio_impact'] * 100,
                'Crollo di\nMercato': results['market_crash']['portfolio_impact'] * 100,
                'Bear Market\nProlungato': results['prolonged_bear']['portfolio_impact'] * 100,
                'Stress\nCombinato': results['combined_stress']['portfolio_impact'] * 100
            }
            
            # Ordina gli scenari dal più negativo al meno negativo
            sorted_scenarios = dict(sorted(scenarios.items(), key=lambda x: x[1]))
            
            # Crea una palette di colori basata sulla severità
            colors = []
            for value in sorted_scenarios.values():
                if value <= -50:
                    colors.append('#ef4444')     # Rosso: impatto severo
                elif value <= -30:
                    colors.append('#f97316')     # Arancione: impatto significativo
                elif value <= -15:
                    colors.append('#eab308')     # Giallo: impatto moderato
                else:
                    colors.append('#22c55e')     # Verde: impatto contenuto
            
            # Crea il grafico a barre orizzontale
            bars = ax_bars.barh(list(sorted_scenarios.keys()), 
                            list(sorted_scenarios.values()),
                            color=colors,
                            alpha=0.8)
            
            # Personalizza il grafico principale
            ax_bars.set_title('Impatto degli Scenari di Stress', 
                            pad=20, fontsize=14, fontweight='bold')
            ax_bars.set_xlabel('Impatto sul Portafoglio (%)', fontsize=11)
            
            # Aggiungi griglia verticale
            ax_bars.grid(axis='x', linestyle='--', alpha=0.3)
            ax_bars.set_axisbelow(True)
            
            # Rimuovi i bordi superflui
            ax_bars.spines['top'].set_visible(False)
            ax_bars.spines['right'].set_visible(False)
            
            # Aggiungi le etichette dei valori sulle barre
            for bar in bars:
                width = bar.get_width()
                x_position = width - 1 if width < 0 else width + 1
                ha = 'right' if width < 0 else 'left'
                ax_bars.text(x_position,
                            bar.get_y() + bar.get_height()/2,
                            f'{width:.1f}%',
                            va='center',
                            ha=ha,
                            fontweight='bold',
                            color='black')
            
            # Imposta i limiti dell'asse x con un po' di padding
            most_negative = min(scenarios.values())
            ax_bars.set_xlim([most_negative * 1.2, 5])
            
            # Aggiungi il pannello informativo
            ax_info.set_axis_off()
            
            # Calcola il punteggio di resilienza
            resilience_score = 100 + sum(scenarios.values()) / len(scenarios)
            
            # Crea il testo informativo
            info_text = "VALUTAZIONE RESILIENZA\n\n"
            
            # Aggiungi indicatore visuale del punteggio
            if resilience_score >= 70:
                score_color = '#22c55e'  # Verde
                status = "(+) ROBUSTA"
            elif resilience_score >= 50:
                score_color = '#eab308'  # Giallo
                status = "(!) MODERATA"
            else:
                score_color = '#ef4444'  # Rosso
                status = "(X) CRITICA"
                
            info_text += f"Resilience Score: {resilience_score:.0f}/100\n"
            info_text += f"Status: {status}\n\n"
            
            # Aggiungi metriche chiave
            info_text += "METRICHE CHIAVE:\n\n"
            info_text += f"Max Drawdown: {abs(min(scenarios.values())):.1f}%\n"
            info_text += f"Media Impatti: {abs(sum(scenarios.values())/len(scenarios)):.1f}%\n"
            info_text += f"Scenari Critici: {sum(1 for v in scenarios.values() if v <= -30)}/4\n\n"
            
            # Aggiungi legenda severità
            info_text += "LEGENDA SEVERITA:\n\n"
            info_text += "- Impatto Severo    (<= -50%)\n"
            info_text += "- Impatto Alto      (<= -30%)\n"
            info_text += "- Impatto Moderato  (<= -15%)\n"
            info_text += "- Impatto Contenuto (> -15%)"
            
            # Aggiungi il testo informativo in un box
            ax_info.text(0.05, 0.95, info_text,
                        fontsize=10,
                        va='top',
                        bbox=dict(facecolor='white',
                                alpha=0.8,
                                edgecolor='gray',
                                boxstyle='round'))
            
            # Layout finale e salvataggio
            plt.savefig('stress_test.png', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return self._create_stress_test_interpretation(results)
            
        except Exception as e:
            logger.error(f"Errore nella creazione del grafico stress test: {e}")
            raise

    def _create_stress_test_interpretation(self, results: dict) -> str:
        """
        Genera un'interpretazione avanzata dei risultati dello stress test.
        """
        try:
            # Estrai i valori percentuali
            inflation_impact = results['high_inflation']['portfolio_impact'] * 100
            market_crash = results['market_crash']['portfolio_impact'] * 100
            bear_market = results['prolonged_bear']['portfolio_impact'] * 100
            combined = results['combined_stress']['portfolio_impact'] * 100
            
            # Calcola il punteggio di resilienza complessivo (0-100)
            resilience_score = 100 + (inflation_impact + market_crash + bear_market + combined) / 4
            
            # SEZIONE 1: ANALISI DETTAGLIATA
            analysis = "ANALISI DETTAGLIATA DEGLI STRESS TEST:\n\n"
            
            # 1.1 Resilienza all'Inflazione
            analysis += "Impatto Inflazione Elevata:\n"
            if inflation_impact <= -70:
                analysis += f"- CRITICITA SEVERA: Perdita potenziale del {abs(inflation_impact):.1f}%\n"
                analysis += "- Il portafoglio mostra estrema vulnerabilita a scenari inflazionistici\n"
            elif inflation_impact <= -50:
                analysis += f"- ATTENZIONE: Perdita potenziale del {abs(inflation_impact):.1f}%\n"
                analysis += "- Significativa sensibilita all'inflazione elevata\n"
            else:
                analysis += f"- Impatto contenuto: Perdita limitata al {abs(inflation_impact):.1f}%\n"
                analysis += "- Discreta protezione contro scenari inflazionistici\n"
            
            # 1.2 Resistenza ai Crolli di Mercato
            analysis += "\nImpatto Crollo di Mercato:\n"
            if market_crash <= -60:
                analysis += f"- RISCHIO CRITICO: Drawdown del {abs(market_crash):.1f}%\n"
                analysis += "- Esposizione eccessiva al rischio di mercato\n"
            elif market_crash <= -40:
                analysis += f"- ATTENZIONE: Drawdown del {abs(market_crash):.1f}%\n"
                analysis += "- Significativa sensibilita alle correzioni di mercato\n"
            else:
                analysis += f"- Drawdown contenuto: {abs(market_crash):.1f}%\n"
                analysis += "- Buona resilienza alle correzioni di mercato\n"
            
            # 1.3 Analisi Bear Market
            analysis += "\nScenario Bear Market Prolungato:\n"
            if bear_market <= -45:
                analysis += f"- RISCHIO ELEVATO: Erosione del {abs(bear_market):.1f}%\n"
                analysis += "- Alta vulnerabilita a mercati ribassisti prolungati\n"
            elif bear_market <= -30:
                analysis += f"- ATTENZIONE: Erosione del {abs(bear_market):.1f}%\n"
                analysis += "- Moderata resistenza ai mercati ribassisti\n"
            else:
                analysis += f"- Impatto gestibile: {abs(bear_market):.1f}%\n"
                analysis += "- Buona tenuta in scenari ribassisti prolungati\n"
            
            # 1.4 Scenario Combinato
            analysis += "\nScenario di Stress Combinato:\n"
            if combined <= -75:
                analysis += f"- ALLARME: Potenziale perdita del {abs(combined):.1f}%\n"
                analysis += "- Il portafoglio e altamente vulnerabile a shock multipli\n"
            elif combined <= -50:
                analysis += f"- CRITICITA: Potenziale perdita del {abs(combined):.1f}%\n"
                analysis += "- Moderata resilienza a shock multipli\n"
            else:
                analysis += f"- Impatto significativo ma gestibile: {abs(combined):.1f}%\n"
                analysis += "- Discreta resilienza a shock multipli\n"
            
            # SEZIONE 2: VALUTAZIONE COMPLESSIVA
            analysis += f"\nVALUTAZIONE COMPLESSIVA (Resilience Score: {resilience_score:.0f}/100):\n"
            if resilience_score >= 70:
                analysis += "(+) Portafoglio robusto con buona resilienza agli stress\n"
                analysis += "(+) Struttura difensiva adeguata\n"
            elif resilience_score >= 50:
                analysis += "(!) Portafoglio con resilienza moderata\n"
                analysis += "(!) Necessarie alcune ottimizzazioni difensive\n"
            else:
                analysis += "(X) Portafoglio con vulnerabilita significative\n"
                analysis += "(X) Necessaria revisione strutturale della strategia\n"
            
            # SEZIONE 3: RACCOMANDAZIONI SPECIFICHE
            analysis += "\nRACCOMANDAZIONI STRATEGICHE:\n"
            
            # 3.1 Protezione Inflazione
            if inflation_impact <= -50:
                analysis += "1. Protezione Inflazione:\n"
                analysis += "   - Incrementare allocazione in TIPS (7-10%)\n"
                analysis += "   - Valutare REITs e commodities (5-8%)\n"
                analysis += "   - Considerare azioni value con pricing power\n"
            
            # 3.2 Gestione Rischio di Mercato
            if market_crash <= -40:
                analysis += "\n2. Mitigazione Rischio di Mercato:\n"
                analysis += "   - Implementare strategia di hedging dinamico\n"
                analysis += "   - Aumentare allocazione in strumenti decorrelati\n"
                analysis += "   - Valutare put protettive su indici principali\n"
            
            # 3.3 Gestione della Liquidita
            buffer_size = "20-25%" if combined <= -60 else "15-20%" if combined <= -40 else "10-15%"
            analysis += f"\n3. Gestione della Liquidita:\n"
            analysis += f"   - Mantenere buffer di liquidita del {buffer_size}\n"
            analysis += "   - Strutturare ladder di depositi/titoli di stato\n"
            
            # 3.4 Diversificazione
            if bear_market <= -30:
                analysis += "\n4. Ottimizzazione Diversificazione:\n"
                analysis += "   - Aumentare esposizione a strategie alternative\n"
                analysis += "   - Valutare allocazione in absolute return\n"
                analysis += "   - Considerare strategie di volatilita gestita\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Errore nell'interpretazione degli stress test: {e}")
            return "Errore nell'analisi dei risultati dello stress test."

    def _add_timeline_chart(self, page_width, page_height):
            """Aggiunge il grafico timeline con interpretazione."""
            self.pdf.set_font('Helvetica', 'B', 12)
            self.pdf.cell(0, 8, 'Timeline delle Simulazioni', ln=True)
            self.pdf.set_font('Helvetica', 'I', 8)
            self.pdf.cell(0, 4, 'Distribuzione dei fallimenti nel tempo e tasso cumulativo', ln=True)
            self.pdf.ln(2)
            
            # Aggiunge il grafico
            self.pdf.image("timeline.png", 
                        x=10,
                        y=self.pdf.get_y(),
                        w=page_width,
                        h=page_height)
            
            # Sposta il cursore e aggiunge interpretazione
            self.pdf.set_y(self.pdf.get_y() + page_height + 5)
            self._add_interpretation_box(self._get_timeline_interpretation())

    def _add_distribution_chart(self, page_width, page_height):
        """Aggiunge il grafico distribuzione con interpretazione."""
        try:
            self.pdf.add_page()
            self.pdf.set_font('Helvetica', 'B', 12)
            self.pdf.cell(0, 8, 'Distribuzione dei Valori Finali', ln=True)
            self.pdf.set_font('Helvetica', 'I', 8)
            self.pdf.cell(0, 4, 'Analisi della distribuzione dei valori finali del portafoglio', ln=True)
            self.pdf.ln(2)
            
            # Get values from final positions in report_data
            if 'simulations' in self.report_data:
                final_values = self.report_data['simulations'][:, -1]
            else:
                # In caso non ci siano i dati, log dell'errore e usa valori di esempio
                logger.error("Dati delle simulazioni non trovati nel report")
                raise ValueError("Dati delle simulazioni mancanti")
            
            # Aggiungi il grafico
            self.pdf.image("distribution.png",
                        x=10,
                        y=self.pdf.get_y(),
                        w=page_width,
                        h=page_height)
            
            # Sposta il cursore e aggiunge interpretazione
            self.pdf.set_y(self.pdf.get_y() + page_height + 5)
            self._add_interpretation_box(self._get_distribution_interpretation(final_values))
        except Exception as e:
            logger.error(f"Errore nell'aggiunta del grafico distribuzione: {e}")
            raise

    def _add_header(self):
        """Aggiunge l'intestazione del report."""
        self.pdf.set_font('Helvetica', 'B', 24)
        self.pdf.cell(0, 15, 'RetireMC', ln=True, align='C')
        
        self.pdf.set_font('Helvetica', 'I', 12)
        self.pdf.cell(0, 10, f'Simulazione finanziaria Monte Carlo basata su {self.num_simulations:,} iterazioni', 
                     ln=True, align='C')
        self.pdf.ln(5)
        
        self.pdf.set_font('Helvetica', 'B', 16)
        self.pdf.cell(0, 10, 'Report Analisi Portafoglio', ln=True, align='C')
        self.pdf.set_font('Helvetica', '', 10)
        self.pdf.cell(0, 10, f"Generato il: {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
        self.pdf.ln(5)

    def _add_monte_carlo_summary(self):
        """Aggiunge la sezione di sintesi delle simulazioni Monte Carlo"""
        # Titolo principale con sfondo verde
        self.pdf.set_font('Helvetica', 'B', 14)
        self.pdf.set_fill_color(144, 238, 144)  # Verde chiaro, stesso degli altri titoli
        self.pdf.cell(0, 10, 'Sintesi delle simulazioni Monte Carlo', ln=True, fill=True)
        self.pdf.ln(5)

        # Success Rate Box
        stats = self.report_data['risk_metrics'].success_stats
        success_rate = stats['overall_success_rate'] * 100
        total_sims = stats['total_simulations']
        successful_sims = stats['successful_simulations']

        # Percentuale grande con font più piccolo
        self.pdf.set_font('Helvetica', 'B', 20)  # Ridotto da 24 a 20
        self.pdf.cell(0, 10, f"{success_rate:.1f}%", ln=True)
        
        # Sottotitolo
        self.pdf.set_font('Helvetica', 'B', 12)
        self.pdf.cell(0, 6, "Success Rate", ln=True)

        # Testo esplicativo Success Rate
        self.pdf.set_font('Helvetica', '', 10)
        explanation = (f"{successful_sims:,} out of {total_sims:,} retirement simulations "
                    f"were able to sustain withdrawals for the entire retirement.")
        self.pdf.multi_cell(0, 5, explanation)
        self.pdf.ln(10)

        # Large End Portfolio Value
        portfolio_metrics = self.report_data['risk_metrics'].portfolio_metrics
        large_end = portfolio_metrics['large_end_portfolio']
        
        self.pdf.set_font('Helvetica', 'B', 20)  # Ridotto da 24 a 20
        self.pdf.cell(0, 10, f"{large_end['percentage']:.1f}%", ln=True)
        
        # Sottotitolo
        self.pdf.set_font('Helvetica', 'B', 12)
        self.pdf.cell(0, 6, "Large End Portfolio Value", ln=True)
        
        self.pdf.set_font('Helvetica', '', 10)
        large_explanation = (f"{large_end['count']:,} out of {large_end['total']:,} retirement simulations "
                            f"had an end portfolio value that was at least 100% more than the initial portfolio value.")
        self.pdf.multi_cell(0, 5, large_explanation)
        self.pdf.ln(10)

        # Small End Portfolio Value
        small_end = portfolio_metrics['small_end_portfolio']
        
        self.pdf.set_font('Helvetica', 'B', 20)  # Ridotto da 24 a 20
        self.pdf.cell(0, 10, f"{small_end['percentage']:.1f}%", ln=True)
        
        # Sottotitolo
        self.pdf.set_font('Helvetica', 'B', 12)
        self.pdf.cell(0, 6, "Small End Portfolio Value", ln=True)
        
        self.pdf.set_font('Helvetica', '', 10)
        small_explanation = (f"{small_end['count']:,} out of {small_end['total']:,} retirement simulations "
                            f"had a nonzero end portfolio value that was at least 50% less than the initial portfolio value.")
        self.pdf.multi_cell(0, 5, small_explanation)
        self.pdf.ln(10)

    def export_report(self, report_data, filename="portfolio_report.pdf"):
        """Esporta i dati del report in PDF con layout corretto."""
        try:
            self.report_data = report_data
            self.pdf.alias_nb_pages()
            
            # === SEZIONE 1: INTESTAZIONE E CONFIGURAZIONE ===
            self.pdf.add_page()
            self._add_header()
            self._add_config_details()
            
            # === SEZIONE 2: METRICHE DI RISCHIO ===
            self._add_section_title("Metriche di Rischio")
            self._add_risk_metrics(report_data['risk_metrics'])
            
            # === SEZIONE 3: ANALISI PRELIEVI ===
            self.pdf.add_page()  # Nuova pagina
            self._add_section_title("Analisi Sostenibilità Prelievi")
            self._add_withdrawal_analysis(report_data['withdrawal_analysis'])
            
            # === SEZIONE 4: GRAFICI MONTE CARLO ===
            self.pdf.add_page()  # Nuova pagina
            self._add_section_title("Analisi Grafica del Portafoglio")
            self._add_monte_carlo_charts()
            
            # === SEZIONE 5: TIMELINE ===
            self._add_timeline_chart(self.pdf.w - 40, 100)  # Dimensioni del grafico timeline
            
            # === SEZIONE 6: DISTRIBUZIONE ===
            self.pdf.add_page()  # Nuova pagina
            self._add_distribution_chart(self.pdf.w - 40, 100)  # Dimensioni del grafico distribuzione
            
            # === SEZIONE 7: STRESS TEST ===
            self.pdf.add_page()  # Nuova pagina
            self._add_stress_test_section()
            
            # Salvataggio finale
            self.pdf.output(filename, 'F')
            logger.info(f"Report PDF generato con successo: {filename}")
            
        except Exception as e:
            logger.error(f"Errore nell'esportazione del PDF: {e}")
            raise

    def _add_interpretation_box(self, text):
        """Aggiunge un box interpretativo compatto sotto il contenuto."""
        try:
            # Calcola margini e dimensioni
            margin = 10
            box_width = self.pdf.w - (2 * margin)
            
            # Split del testo in linee e calcolo altezza necessaria
            lines = text.split('\n')
            line_height = 5
            text_height = len(lines) * line_height + 10  # +10 per padding
            
            # Verifica se c'è abbastanza spazio nella pagina corrente
            if (self.pdf.get_y() + text_height + 20) > self.pdf.h:
                self.pdf.add_page()
                
            # Background grigio chiaro per il box
            start_y = self.pdf.get_y() + 5  # Piccolo spazio prima del box
            self.pdf.set_fill_color(245, 245, 245)
            self.pdf.rect(margin, start_y, box_width, text_height, 'F')
            
            # Imposta il font per il contenuto
            self.pdf.set_font('Helvetica', '', 10)
            
            # Aggiungi il testo riga per riga
            current_y = start_y + 5  # Padding superiore
            for line in lines:
                if line.strip():  # Salta linee vuote
                    self.pdf.set_xy(margin + 5, current_y)
                    self.pdf.cell(box_width - 10, line_height, line.strip(), 0, 1)
                    current_y += line_height
            
            # Aggiorna la posizione Y dopo il box
            self.pdf.set_y(current_y + 5)  # Piccolo spazio dopo il box
            
        except Exception as e:
            logger.error(f"Error adding interpretation box: {e}")
            raise

    def _get_timeline_interpretation(self):
        """Genera l'interpretazione della timeline dei fallimenti."""
        try:
            if not self.report_data or 'withdrawal_analysis' not in self.report_data:
                return "Dati insufficienti per l'interpretazione."
                    
            withdrawal = self.report_data['withdrawal_analysis']
            risk_metrics = self.report_data.get('risk_metrics', {})
            
            interpretation = []
            
            # Analisi del rischio complessivo usando success_rate invece di ruin_probability
            if hasattr(risk_metrics, 'success_rate'):
                failure_rate = 1.0 - risk_metrics.success_rate
            else:
                # Fallback su calcolo basato sui dati di withdrawal se risk_metrics non disponibile
                failure_rate = withdrawal.get('total_failure_probability', 0.0)
                
            if failure_rate <= 0.05:
                interpretation.append("- Rischio di fallimento molto basso (< 5%)")
            elif failure_rate <= 0.15:
                interpretation.append("- Rischio di fallimento moderato (5-15%)")
            else:
                interpretation.append("- ATTENZIONE: Rischio di fallimento elevato (> 15%)")
            
            # Analisi della sostenibilità temporale
            median_life = withdrawal.get('median_portfolio_life', 0)
            target_years = self.config.withdrawal_years
            
            if median_life >= target_years:
                interpretation.append(f"- Il portafoglio raggiunge l'obiettivo di {target_years} anni")
            else:
                interpretation.append(f"- Il portafoglio potrebbe esaurirsi prima dei {target_years} anni target")
            
            # Analisi dei fallimenti precoci
            # Cerchiamo prima in maintenance_phase_analysis, poi in altre strutture
            early_failures = 0
            if 'maintenance_phase_analysis' in withdrawal:
                early_failures = withdrawal['maintenance_phase_analysis'].get('early_failure_probability', 0)
            elif 'failure_probabilities' in withdrawal and len(withdrawal['failure_probabilities']) > 4:
                early_failures = sum(withdrawal['failure_probabilities'][:5]) / 5
                
            if early_failures <= 0.02:
                interpretation.append("- Bassissimo rischio di fallimento nei primi 5 anni")
            elif early_failures <= 0.05:
                interpretation.append("- Moderato rischio di fallimento nei primi 5 anni")
            else:
                interpretation.append("- ATTENZIONE: Alto rischio di fallimento precoce")
            
            # Aggiungi raccomandazioni se necessario
            if failure_rate > 0.15 or early_failures > 0.05:
                interpretation.append("\nRACCOMANDAZIONI:")
                if failure_rate > 0.15:
                    interpretation.append("- Considerare una riduzione del tasso di prelievo")
                if early_failures > 0.05:
                    interpretation.append("- Valutare un cuscinetto di liquidità per i primi anni")
            
            return "\n".join(interpretation)
                
        except Exception as e:
            logger.error(f"Errore nella generazione dell'interpretazione timeline: {e}")
            return "Errore nell'interpretazione dei dati."

    def _get_distribution_interpretation(self, final_values: np.ndarray) -> str:
        """Genera un'interpretazione più dettagliata della distribuzione dei valori finali."""
        try:
            # Calcolo statistiche chiave
            median_value = np.median(final_values)
            mean_value = np.mean(final_values)
            p10 = np.percentile(final_values, 10)
            p90 = np.percentile(final_values, 90)
            std_dev = np.std(final_values)
            
            interpretation = []
            
            # Analisi delle statistiche di base
            interpretation.append("ANALISI DISTRIBUZIONE VALORI FINALI:")
            interpretation.append(f"- Valore mediano del portafoglio: EUR {median_value:,.0f}")
            interpretation.append(f"- Valore medio del portafoglio: EUR {mean_value:,.0f}")
            
            # Analisi della dispersione
            interpretation.append("\nDISPERSIONE DEI RISULTATI:")
            interpretation.append(f"- 10° percentile: EUR {p10:,.0f}")
            interpretation.append(f"- 90° percentile: EUR {p90:,.0f}")
            interpretation.append(f"- Deviazione standard: EUR {std_dev:,.0f}")
            
            # Analisi dell'asimmetria
            skewness = (mean_value - median_value) / std_dev if std_dev > 0 else 0
            if abs(skewness) < 0.2:
                interpretation.append("\nFORMA DELLA DISTRIBUZIONE:")
                interpretation.append("- Distribuzione sostanzialmente simmetrica")
            elif skewness > 0:
                interpretation.append("\nFORMA DELLA DISTRIBUZIONE:")
                interpretation.append("- Distribuzione asimmetrica positiva (tendenza a risultati superiori alla media)")
            else:
                interpretation.append("\nFORMA DELLA DISTRIBUZIONE:")
                interpretation.append("- Distribuzione asimmetrica negativa (tendenza a risultati inferiori alla media)")
            
            return "\n".join(interpretation)
            
        except Exception as e:
            logger.error(f"Errore nell'interpretazione della distribuzione: {e}")
            return "Errore nell'analisi della distribuzione dei valori finali."
       
    def _add_section_title(self, title):
        """Aggiunge un titolo di sezione con stile migliorato."""
        self.pdf.ln(5)  # Aggiunge spazio prima del titolo
        self.pdf.set_font('Helvetica', 'B', 14)
        self.pdf.set_fill_color(144, 238, 144)  # Grigio chiaro per lo sfondo
        self.pdf.cell(0, 10, title, ln=True, fill=True)
        self.pdf.set_font('Helvetica', '', 10)
        self.pdf.ln(2)  # Spazio dopo il titolo
        
    def _add_config_details(self):
        """Aggiunge i dettagli della configurazione con layout migliorato."""
        config_items = [
            ('Anni di accumulo', f"{self.config.accumulation_years}"),
            ('Anni di mantenimento', f"{self.config.maintenance_years}"),
            ('Anni di prelievo', f"{self.config.withdrawal_years}"),
            ('Investimento annuo', f"EUR {float(self.config.investment_amount):,.2f}"),
            ('Prelievo fase mantenimento', f"EUR {float(self.config.maintenance_withdrawal):,.2f}"),
            ('Prelievo fase decumulo', f"EUR {float(self.config.withdrawal_amount):,.2f}"),
            ('Rendimento medio atteso', f"{float(self.config.mean_return):.2f}%"),
            ('Volatilita', f"{float(self.config.std_dev_return):.2f}%"),
            ('Inflazione', f"{float(self.config.inflation_rate):.2f}%")
        ]
        
        # Crea una tabella per i dettagli della configurazione
        col_width = self.pdf.w / 2 - 20
        for i, (label, value) in enumerate(config_items):
            if i % 2 == 0:
                self.pdf.set_x(10)
            else:
                self.pdf.set_x(col_width + 20)
            
            self._safe_cell(col_width * 0.6, 7, label + ":", 0)
            self._safe_cell(col_width * 0.4, 7, value, 0)
            
            if i % 2 == 1 or i == len(config_items) - 1:
                self.pdf.ln()
            
    def _add_risk_metrics(self, metrics: Dict):
        """Aggiunge le metriche di rischio con layout migliorato."""
        try:
            metrics_items = [
                ('Risk Score', metrics.get('risk_score', '0')),
                ('Maximum Drawdown', metrics.get('max_drawdown', '0%')),
                ('Average Drawdown', metrics.get('avg_drawdown', '0%')),
                ('Drawdown Duration', metrics.get('drawdown_duration', '0 anni')),
                ('Recovery Time', metrics.get('recovery_time', '0 anni')),
                ('Underwater Periods', str(metrics.get('underwater_periods', '0'))),
                ('Tail Loss Probability', metrics.get('tail_loss_probability', '0%')),
                ('Expected Tail Loss', metrics.get('expected_tail_loss', '0%')),
                ('Skewness', metrics.get('skewness', '0.00')),
                ('Kurtosis', metrics.get('kurtosis', '0.00')),
                ('Stress Resilience Score', metrics.get('stress_resilience_score', '0/100')),
                ('Black Swan Impact', metrics.get('black_swan_impact', '0%')),
                ('Worst Case Recovery', metrics.get('worst_case_recovery', '0 anni')),
                ('Return Stability Index', metrics.get('return_stability_index', '0.00')),
                ('Success Rate', metrics.get('success_rate', '0%'))
            ]
            
            # Layout a due colonne per le metriche
            col_width = self.pdf.w / 2 - 20
            for i, (label, value) in enumerate(metrics_items):
                if i % 2 == 0:
                    self.pdf.set_x(10)
                else:
                    self.pdf.set_x(col_width + 20)
                
                self._safe_cell(col_width * 0.6, 7, label + ":", 0)
                self._safe_cell(col_width * 0.4, 7, str(value), 0)
                
                if i % 2 == 1 or i == len(metrics_items) - 1:
                    self.pdf.ln()
        except Exception as e:
            logger.error(f"Error adding risk metrics to PDF: {e}")
            raise
            
    def _add_withdrawal_analysis(self, withdrawal_data: Dict):
        """Aggiunge l'analisi dei prelievi con layout migliorato."""
        try:
            # Fase di Mantenimento (se presente)
            if 'maintenance' in withdrawal_data:
                self._add_section_subtitle("Fase di Mantenimento")
                maint = withdrawal_data['maintenance']
                
                # Dettagli principali
                self._safe_cell(100, 7, "Vita Mediana Portafoglio:", 0)
                self._safe_cell(0, 7, f"{maint['median_life']} anni", ln=True)
                
                self._safe_cell(100, 7, "Probabilità Totale di Fallimento:", 0)
                self._safe_cell(0, 7, f"{maint['total_failure_prob']}", ln=True)
                
                self._safe_cell(100, 7, "Probabilità Fallimento Precoce:", 0)
                self._safe_cell(0, 7, f"{maint['early_failure_prob']}", ln=True)
                
                # Probabilità di fallimento per anno
                self.pdf.ln(5)
                self.pdf.set_font('Helvetica', 'B', 10)
                self._safe_cell(0, 7, "Probabilità di Fallimento per Anno (Mantenimento):", ln=True)
                self.pdf.set_font('Helvetica', '', 10)
                
                # Layout a due colonne per le probabilità
                col_width = self.pdf.w / 2 - 20
                probabilities = maint['failure_probabilities']
                for i, prob in enumerate(probabilities):
                    if i % 2 == 0:
                        self.pdf.set_x(10)
                    else:
                        self.pdf.set_x(col_width + 20)
                    
                    self._safe_cell(col_width * 0.6, 7, f"Anno {i + 1}:", 0)
                    self._safe_cell(col_width * 0.4, 7, f"{prob}", 0)
                    
                    if i % 2 == 1 or i == len(probabilities) - 1:
                        self.pdf.ln()
                        
                self.pdf.ln(5)

            # Fase di Prelievo
            self._add_section_subtitle("Fase di Prelievo")
            withdrawal = withdrawal_data['withdrawal']
            
            # Dettagli principali prelievo
            self._safe_cell(100, 7, "Vita Mediana Portafoglio:", 0)
            self._safe_cell(0, 7, f"{withdrawal['median_life']} anni", ln=True)
            
            # Probabilità di fallimento per anno (prelievo)
            self.pdf.ln(5)
            self.pdf.set_font('Helvetica', 'B', 10)
            self._safe_cell(0, 7, "Probabilità di Fallimento per Anno (Prelievo):", ln=True)
            self.pdf.set_font('Helvetica', '', 10)
            
            # Layout a due colonne per le probabilità
            col_width = self.pdf.w / 2 - 20
            probabilities = withdrawal['failure_probabilities']
            for i, prob in enumerate(probabilities):
                if i % 2 == 0:
                    self.pdf.set_x(10)
                else:
                    self.pdf.set_x(col_width + 20)
                
                self._safe_cell(col_width * 0.6, 7, f"Anno {i + 1}:", 0)
                self._safe_cell(col_width * 0.4, 7, f"{prob}", 0)
                
                if i % 2 == 1 or i == len(probabilities) - 1:
                    self.pdf.ln()
                    
            self.pdf.ln(5)

            # Analisi Combinata (se presente)
            if 'combined' in withdrawal_data:
                self._add_section_subtitle("Analisi Combinata")
                combined = withdrawal_data['combined']
                
                self._safe_cell(100, 7, "Probabilità Totale di Fallimento:", 0)
                self._safe_cell(0, 7, f"{combined['total_failure_prob']}", ln=True)
                
                self._safe_cell(100, 7, "Tempo Mediano di Sopravvivenza:", 0)
                self._safe_cell(0, 7, f"{combined['median_survival']} anni", ln=True)
                
                self._safe_cell(100, 7, "Successo Transizione Mantenimento-Prelievo:", 0)
                self._safe_cell(0, 7, f"{combined['transition_success']}", ln=True)
                
                # Distribuzione primi fallimenti
                if 'first_failure_distribution' in combined:
                    self.pdf.ln(5)
                    self.pdf.set_font('Helvetica', 'B', 10)
                    self._safe_cell(0, 7, "Distribuzione Primi Fallimenti per Anno:", ln=True)
                    self.pdf.set_font('Helvetica', '', 10)
                    
                    for year, prob in sorted(combined['first_failure_distribution'].items()):
                        self._safe_cell(60, 7, f"Anno {year}:", 0)
                        self._safe_cell(0, 7, f"{prob}", ln=True)

        except Exception as e:
            logger.error(f"Errore nell'aggiunta dell'analisi dei prelievi: {e}")
            raise

    def _add_section_subtitle(self, subtitle: str):
        """Aggiunge un sottotitolo di sezione con stile uniforme."""
        self.pdf.ln(5)
        self.pdf.set_font('Helvetica', 'B', 12)
        self.pdf.set_fill_color(245, 245, 245)  # Grigio molto chiaro
        self._safe_cell(0, 8, subtitle, ln=True, fill=True)
        self.pdf.set_font('Helvetica', '', 10)
        self.pdf.ln(2)

    def _add_stress_test_results(self, stress_data):
        """Aggiunge i risultati degli stress test con layout migliorato."""
        try:
            # Crea il grafico con la nuova interpretazione
            interpretation = self._create_stress_test_chart()
            
            # Aggiungi il box interpretativo con il nuovo testo
            self._add_interpretation_box(interpretation)
            
        except Exception as e:
            logger.error(f"Errore nell'aggiunta dei risultati stress test: {e}")
            return "Errore nell'analisi dei risultati dello stress test."
        
    def _add_analysis_charts(self):
        """Aggiunge i grafici dell'analisi."""
        try:
            self.pdf.add_page()
            
            # Titolo sezione grafici
            self.pdf.set_font('Helvetica', 'B', 16)
            self.pdf.set_fill_color(230, 230, 230)
            self.pdf.cell(0, 12, 'Analisi Grafica del Portafoglio', ln=True, align='C', fill=True)
            self.pdf.ln(5)
            
            # Dimensioni ottimizzate
            page_width = self.pdf.w - 20
            page_height = (self.pdf.h - 60) / 2
            
            # Timeline delle Simulazioni
            self._add_timeline_chart(page_width, page_height)
            
            # Distribuzione dei Valori Finali
            self._add_distribution_chart(page_width, page_height)
            
        except Exception as e:
            logger.error(f"Errore nell'aggiunta dei grafici: {e}")
            raise

    def _add_simulation_details(self):
        """Aggiunge i dettagli della simulazione con formattazione migliorata."""
        if not self.simulation_details:
            return
            
        self.pdf.set_font('Helvetica', '', 9)  # Dimensione font leggermente aumentata
        
        lines = self.simulation_details.split('\n')
        
        for line in lines:
            if not line.strip():
                self.pdf.ln(3)
                continue
                
            if line.startswith('==='):
                self.pdf.ln(3)
                self.pdf.set_font('Helvetica', 'B', 11)
                self.pdf.set_fill_color(245, 245, 245)
                self._safe_cell(0, 6, line.strip('= '), ln=True, fill=True)
                self.pdf.set_font('Helvetica', '', 9)
                continue
            
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                self._safe_cell(indent * 1.5, 5, '')
                line = line.lstrip()
            
            # Gestione testo lungo
            max_width = 180
            if self.pdf.get_string_width(line) > max_width:
                words = line.split()
                current_line = ""
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if self.pdf.get_string_width(test_line) < max_width:
                        current_line = test_line
                    else:
                        self._safe_cell(0, 5, current_line, ln=True)
                        current_line = word
                if current_line:
                    self._safe_cell(0, 5, current_line, ln=True)
            else:
                self._safe_cell(0, 5, line, ln=True)

    def _safe_cell(self, w, h, txt, border=0, ln=0, align='', fill=False):
        """Wrapper sicuro per cell() con gestione migliorata dei caratteri speciali."""
        # Gestione caratteri speciali e formattazione
        txt = (str(txt)
            .replace('€', 'EUR ')      # <-- Da modificare: sostituisce € con EUR
            .replace('à', 'a')
            .replace('è', 'e')
            .replace('ì', 'i')
            .replace('ò', 'o')
            .replace('ù', 'u'))
        
        # Gestione valori monetari
        if 'EUR' in txt and any(c.isdigit() for c in txt):
            align = 'R'  # Allinea a destra i valori monetari
            
        self.pdf.cell(w, h, txt, border, ln, align, fill)