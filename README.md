# RetireMC - Simulatore Monte Carlo per Analisi Finanziaria

## Descrizione
RetireMC è un simulatore finanziario basato sul metodo Monte Carlo che analizza strategie di investimento e prelievo per la pianificazione finanziaria a lungo termine. Il programma valuta la sostenibilità del portafoglio attraverso diverse fasi della vita finanziaria: 
- accumulo: è la fase in cui si accumula il capitale fino al momento in cui si decide di smettere di lavorare ma prima di poter recepire la pensione obbligatoria
- mantenimento: è la fase in cui si preleva il capitale per coprire le spese senza esaurire il capitale investito. In questa fase non si percepisce nè stipendio nè pensione obbligatoria. 
- prelievo: è la fase in cui si preleva il capitale per coprire le spese fino al fine vita. In questa fase si percepisce la pensione obbligatoria e quella complementare.


## Funzionalità Principali

### 1. Simulazione Monte Carlo
- Generazione di migliaia di scenari di mercato possibili
- Analisi di tre fasi distinte: accumulo, mantenimento e prelievo
- Calcolo parallelo per performance ottimizzate

### 2. Metriche di Rischio
- Risk Score composito (0-100)
- Maximum e Average Drawdown
- Recovery Time
- Stress Resilience Score
- Return Stability Index
- Success Rate

### 3. Analisi di Sostenibilità
- Analisi fase di mantenimento
- Analisi fase di prelievo
- Probabilità di fallimento per anno
- Vita mediana del portafoglio

### 4. Stress Testing
- Scenario inflazione elevata
- Scenario crollo di mercato
- Scenario bear market prolungato
- Scenario combinato di stress

### 5. Reporting
- Report dettagliato in PDF
- Grafici di analisi
- Interpretazioni automatiche dei risultati
- Timeline delle simulazioni
- Distribuzione dei valori finali

## Dettagli della Simulazione

### Generazione dei Rendimenti
Il programma simula i rendimenti annuali secondo queste premesse:
- Utilizza una distribuzione normale (Gaussiana) per generare i rendimenti
- Media = mean_return configurato
- Deviazione standard = std_dev_return configurato
- Considera la possibilità di eventi estremi (black swan) con probabilità configurabile

La formula base per ogni anno è:
```
rendimento_anno = random_normal(mean_return, std_dev_return)
```

### Calcolo del Valore del Portafoglio
Per ogni anno, il valore del portafoglio viene aggiornato:
```
nuovo_valore = valore_precedente * (1 + rendimento_anno)
```

### Gestione dei Prelievi e Tassazione

#### Prezzo Medio di Carico (PMC)
Il programma utilizza il metodo del Prezzo Medio di Carico per il calcolo delle plusvalenze:
- Tiene traccia del PMC per ogni posizione nel portafoglio
- Il PMC viene aggiornato ad ogni nuovo investimento secondo la formula:
```
nuovo_PMC = (PMC_precedente * N_quote_precedenti + Prezzo_nuovo * N_quote_nuove) / (N_quote_precedenti + N_quote_nuove)
```

#### Calcolo del Prelievo Lordo e Netto
Quando viene effettuato un prelievo:
1. Si determina l'importo lordo necessario per ottenere il netto desiderato
2. Si calcola la plusvalenza sulla parte venduta usando il PMC:
```
plusvalenza = (prezzo_attuale - PMC) * quote_vendute
imposta = plusvalenza * aliquota_capital_gain  # 26% in Italia
```
3. Il prelievo lordo viene aumentato per compensare l'imposta:
```
prelievo_lordo = prelievo_netto_desiderato / (1 - aliquota_capital_gain)
```

#### Esempio Pratico
Se si desidera prelevare 1.000€ netti:
1. Con un PMC di 90€ e un valore attuale di 100€
2. Plusvalenza per quota = 10€ (100€ - 90€)
3. Imposta per quota = 2,60€ (26% di 10€)
4. Per ottenere 1.000€ netti, sarà necessario vendere quote per circa 1.351€ lordi

### Aggiornamento del Portafoglio Post-Prelievo
Dopo ogni prelievo:
- Il numero di quote viene ridotto
- Il PMC rimane invariato per le quote rimanenti
- Il valore del portafoglio viene aggiornato

Questo approccio:
- Simula realisticamente l'impatto fiscale sui prelievi
- Tiene conto dell'effetto composto della tassazione nel lungo periodo
- Permette una stima più accurata della sostenibilità dei prelievi

## Parametri Configurabili

### Parametri Temporali
```yaml
accumulation_years: numero di anni fase accumulo
maintenance_years: numero di anni fase mantenimento
withdrawal_years: numero di anni fase prelievo
```

### Parametri Finanziari
```yaml
investment_amount: importo investimento annuo
maintenance_withdrawal: prelievo annuo fase mantenimento
withdrawal_amount: prelievo annuo fase prelievo
mandatory_pension: pensione obbligatoria
complementary_pension: pensione complementare
```

### Parametri di Mercato
```yaml
mean_return: rendimento medio atteso
std_dev_return: deviazione standard dei rendimenti
inflation_rate: tasso di inflazione
risk_free_rate: tasso risk-free
```

### Parametri di Rischio
```yaml
black_swan_probability: probabilità eventi estremi
black_swan_impact: impatto eventi estremi
drawdown_threshold: soglia per calcolo drawdown
confidence_level: livello di confidenza
```

### Parametri Stress Test
```yaml
high_inflation_scenario: livello inflazione per stress test
market_crash_impact: impatto crash di mercato
bear_market_years: durata bear market
combined_stress_impact: impatto scenario combinato
```

### Parametri di Esecuzione
```yaml
batch_size: dimensione batch simulazioni
plot_height: altezza grafici
plot_width: larghezza grafici
distribution_bins: numero bin per distribuzioni
```

## Come Modificare i Parametri

I parametri possono essere modificati in due modi:

1. **File di Configurazione (Config.yaml)**
   - Aprire il file Config.yaml
   - Modificare i valori desiderati
   - Salvare il file

2. **Interfaccia Programma**
   - All'avvio del programma è possibile modificare alcuni parametri chiave
   - Seguire le istruzioni a schermo

### Esempio Config.yaml
```yaml
simulation:
  accumulation_years: 20
  maintenance_years: 5
  withdrawal_years: 25
  investment_amount: 10000
  withdrawal_amount: 25000
  mean_return: 4.0
  std_dev_return: 15.0
  inflation_rate: 2.0

risk_parameters:
  drawdown_threshold: 0.1
  confidence_level: 0.95

stress_test:
  high_inflation_scenario: 8.0
  market_crash_impact: -40.0
  bear_market_years: 5
```

## Configurazione del Portafoglio

### Struttura Standard
Il portafoglio di default è composto da 3 ETF con caratteristiche configurabili:
```python
# Configurazione standard in sim_11.py
etf_positions = {
    'ETF1': ETFPosition(shares=Decimal('100'), price=Decimal('1000'), avg_price=Decimal('1000')),
    'ETF2': ETFPosition(shares=Decimal('150'), price=Decimal('800'), avg_price=Decimal('750')),
    'ETF3': ETFPosition(shares=Decimal('200'), price=Decimal('500'), avg_price=Decimal('450'))
}
```

### Personalizzazione del Portafoglio

#### Modificare il Numero di ETF
Per aumentare o ridurre il numero di ETF:
1. Aprire il file sim_12.py
2. Localizzare la sezione di inizializzazione del portafoglio
3. Aggiungere o rimuovere posizioni:
```python
# Esempio con 2 ETF
etf_positions = {
    'ETF1': ETFPosition(shares=Decimal('100'), price=Decimal('1000'), avg_price=Decimal('1000')),
    'ETF2': ETFPosition(shares=Decimal('150'), price=Decimal('800'), avg_price=Decimal('750'))
}

# Esempio con 4 ETF
etf_positions = {
    'ETF1': ETFPosition(shares=Decimal('100'), price=Decimal('1000'), avg_price=Decimal('1000')),
    'ETF2': ETFPosition(shares=Decimal('150'), price=Decimal('800'), avg_price=Decimal('750')),
    'ETF3': ETFPosition(shares=Decimal('200'), price=Decimal('500'), avg_price=Decimal('450')),
    'ETF4': ETFPosition(shares=Decimal('120'), price=Decimal('600'), avg_price=Decimal('580'))
}
```

#### Parametri Configurabili per Ogni ETF
Per ogni ETF si possono configurare:
- `shares`: numero di quote possedute
- `price`: prezzo attuale della quota
- `avg_price`: Prezzo Medio di Carico (PMC)

#### Modificare le Allocazioni
È possibile modificare anche le allocazioni target per ogni ETF:
```python
# In sim_12.py
allocations = [0.4, 0.35, 0.25]  # Deve sommare a 1
```

### Come Apportare le Modifiche

1. **Modifiche Temporanee**:
   - Le modifiche possono essere fatte direttamente nel codice per test specifici
   - Utile per analisi di scenari alternativi

2. **Modifiche Permanenti**:
   - Creare un nuovo file di configurazione (es: portfolio_config.py)
   - Importare la configurazione in sim_12.py
   ```python
   from portfolio_config import etf_positions, allocations
   ```

3. **Validazione**:
   - Il programma verifica automaticamente che:
     - Le allocazioni sommino a 1
     - Ogni ETF abbia valori validi (non negativi)
     - Il PMC sia coerente con il prezzo attuale

### Esempio di File di Configurazione Completo
```python
# portfolio_config.py
from decimal import Decimal
from portfolio.models.position import ETFPosition

etf_positions = {
    'VWCE': ETFPosition(
        shares=Decimal('100'),
        price=Decimal('1000'),
        avg_price=Decimal('950')
    ),
    'AGGH': ETFPosition(
        shares=Decimal('200'),
        price=Decimal('500'),
        avg_price=Decimal('480')
    ),
    'VAGF': ETFPosition(
        shares=Decimal('150'),
        price=Decimal('800'),
        avg_price=Decimal('790')
    )
}

allocations = [0.60, 0.25, 0.15]  # Corrisponde all'ordine degli ETF
```

### Note Importanti
- Le quote devono essere numeri interi
- I prezzi possono avere decimali
- Le allocazioni devono essere espresse in decimali (es: 0.4 per 40%)
- È consigliabile mantenere nomi significativi per gli ETF
- Il PMC dovrebbe riflettere la storia degli acquisti

## Note Importanti
- Tutti i valori monetari sono in EUR
- Le percentuali vanno espresse come numeri decimali (es: 4.0 per 4%)
- I parametri di rischio influenzano significativamente i risultati
- Si consiglia di testare diverse configurazioni per trovare quella ottimale

## Avvio del Programma
```bash
python sim_12.py
```
