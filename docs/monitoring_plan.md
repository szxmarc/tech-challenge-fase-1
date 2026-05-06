# Plano de Monitoramento — Churn Prediction API

## 1. Objetivos do Monitoramento

O modelo de churn opera em um ambiente onde o comportamento dos clientes pode mudar ao longo do tempo
(novos planos, sazonalidade, ações da concorrência). Sem monitoramento contínuo, o modelo pode
degradar silenciosamente — continuando a servir predições, mas com qualidade reduzida.

**Objetivos principais:**

1. Detectar degradação de performance antes que impacte o valor de negócio
2. Identificar data drift nas features de entrada (distribuição diferente do treino)
3. Garantir disponibilidade e latência dentro dos SLOs definidos
4. Acionar retreinamento de forma criteriosa e reprodutível

---

## 2. Métricas a Monitorar

### 2a. Métricas do Modelo

Avaliadas mensalmente sobre um conjunto de dados com rótulos confirmados (clientes que efetivamente
cancelaram ou permaneceram no período seguinte à predição).

| Métrica | Intervalo Normal | Alerta (Warning) | Crítico | Ação |
|---|---|---|---|---|
| AUC-ROC | ≥ 0.80 | 0.76 – 0.80 | < 0.76 | Iniciar processo de retreinamento |
| Recall | ≥ 0.75 | 0.70 – 0.75 | < 0.70 | Revisar threshold de decisão |
| F1-Score | ≥ 0.70 | 0.65 – 0.70 | < 0.65 | Retreinar com dados recentes |
| Valor de Negócio Total | positivo | próximo de zero | negativo | Revisar threshold + retreinar |

### 2b. Métricas de Dados (Features de Entrada)

Monitoradas semanalmente comparando a distribuição das requisições recentes com o dataset de treino.

| Feature | Tipo | O que monitorar |
|---|---|---|
| `monthlyCharges` | Contínua | Média, desvio padrão, % outliers (> 3σ) |
| `totalCharges` | Contínua | Média, desvio padrão, % zeros inesperados |
| `tenureMonths` | Contínua | Média, % clientes novos (tenure < 3) |
| `contractTwoYear` | Binária | Proporção de contratos anuais/bianuais |
| `internetFiberOptic` | Binária | Proporção de clientes com fibra |
| Valores faltantes | Global | % de campos ausentes por feature |

### 2c. Métricas Operacionais

Monitoradas continuamente (a cada request ou por coleta periódica de logs).

| Métrica | SLO | Ferramenta sugerida |
|---|---|---|
| Latência P50 | < 30ms | Logs estruturados da API |
| Latência P95 | < 100ms | Logs estruturados da API |
| Taxa de erros HTTP 5xx | < 0.1% | Logs + alertas |
| Uptime `/health` | ≥ 99% | Health check externo (UptimeRobot, etc.) |

---

## 3. Detecção de Data Drift

### Ferramenta: Evidently (já listada nas dependências de notebooks)

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

# Carregar dataset de referência (treino)
reference = pd.read_csv("data/processed/baseline/X_train.csv")

# Carregar amostra de produção (últimas N requisições logadas)
production = pd.read_csv("data/monitoring/production_sample.csv")

# Gerar relatório de drift
report = Report(metrics=[
    DataDriftPreset(),
    DatasetDriftMetric(),
])
report.run(reference_data=reference, current_data=production)
report.save_html("reports/drift_report.html")
```

**Interpretação:**
- `dataset_drift = False` → distribuição estável, sem ação necessária
- `dataset_drift = True` → investigar quais features drifaram; avaliar retreinamento

**Threshold de drift:** considerar retreinamento quando ≥ 30% das features apresentarem
drift estatisticamente significativo (p-value < 0.05 no teste KS para features contínuas
ou chi-quadrado para binárias).

---

## 4. Alertas e Limites

| Métrica | Normal | Warning | Crítico | Responsável |
|---|---|---|---|---|
| AUC-ROC | ≥ 0.80 | 0.76 – 0.79 | < 0.76 | Time Data Science |
| Recall | ≥ 0.75 | 0.70 – 0.74 | < 0.70 | Time Data Science |
| Valor de Negócio | positivo | < R$ 10.000 | negativo | Gerente de Retenção |
| % features com drift | < 10% | 10% – 30% | > 30% | Time Data Science |
| Latência P95 | < 100ms | 100 – 200ms | > 200ms | Time de TI |
| Taxa de erros 5xx | < 0.1% | 0.1% – 1% | > 1% | Time de TI |
| Dados faltantes (por feature) | < 5% | 5% – 10% | > 10% | Engenharia de Dados |

---

## 5. Frequência de Monitoramento

| O que monitorar | Frequência | Como |
|---|---|---|
| Saúde da API (`/health`) | Contínuo | Health check externo (a cada 1 min) |
| Latência e erros | Contínuo | Análise dos logs de uvicorn |
| Distribuição de features (drift) | Semanal | Script Evidently em batch |
| Métricas do modelo (AUC, Recall) | Mensal | Avaliação em dados rotulados recentes |
| Valor de negócio | Mensal | Junto com métricas do modelo |
| Auditoria de fairness | Trimestral | Análise por gênero e SeniorCitizen |

---

## 6. Processo de Retreinamento

Execute este processo quando um alerta crítico for disparado **ou** mensalmente no calendário fixo.

### Passo a Passo

```
1. Coletar novos dados rotulados
   - Mínimo: 500 registros com churn confirmado (30-90 dias após predição)
   - Formato: mesmo esquema do Telco-Customer-Churn.csv

2. Atualizar o dataset de treino
   - Concatenar novos dados com o histórico existente
   - Salvar em data/raw/ com data no nome (ex: Telco-Churn-2026-06.csv)
   - Atualizar RAW_DATASET em settings.py se necessário

3. Executar o pipeline
   - Reiniciar a API: python run_app.py
   - O startup pipeline re-treina automaticamente LR + MLP
   - Comparação é salva em reports/model_comparison.json

4. Validar o novo modelo
   - Verificar que AUC-ROC ≥ 0.80 e Recall ≥ 0.75
   - Comparar com a run anterior no MLflow: make mlflow
   - Se novo modelo for pior em AUC-ROC e Recall → não promover

5. Registrar a run no MLflow
   - Tag: retreinamento_YYYY-MM
   - Anotar o motivo do retreinamento (drift / calendário / degradação)

6. Atualizar o Model Card
   - Atualizar tabela de resultados em docs/model_card.md
   - Registrar data e motivo do retreinamento
```

### Critério de Promoção do Novo Modelo

| Condição | Decisão |
|---|---|
| Novo AUC-ROC > atual **e** novo Recall > atual | Promover |
| Novo AUC-ROC > atual **e** novo Recall ≈ atual (< 0.02 diff) | Promover |
| Novo Recall > atual **e** novo AUC-ROC ≈ atual (< 0.02 diff) | Promover |
| Novo modelo pior em ambas as métricas | Não promover; investigar dados |

---

## 7. Ferramentas Sugeridas

| Necessidade | Ferramenta | Status |
|---|---|---|
| Drift de dados | Evidently | Disponível (dependência em pyproject.toml) |
| Rastreamento de experimentos | MLflow | Implementado |
| Saúde da API | FastAPI `/health` | Implementado |
| Logging estruturado | Python `logging` | Implementado |
| Coleta de métricas de latência | Middleware uvicorn / logs | A implementar |
| Dashboard operacional | Grafana + Prometheus | A implementar |
| Alertas automáticos | PagerDuty / Slack webhook | A implementar |

---

## 8. Checklist Mensal de Monitoramento

Copie este checklist a cada ciclo mensal e marque cada item:

```markdown
## Monitoramento — [MÊS/ANO]

### Modelo
- [ ] Coletei pelo menos 500 registros rotulados do mês anterior
- [ ] Avaliei AUC-ROC no conjunto recente → resultado: ___
- [ ] Avaliei Recall no conjunto recente → resultado: ___
- [ ] Calculei Valor de Negócio → resultado: R$ ___
- [ ] Comparei com a baseline do mês anterior no MLflow
- [ ] (Se degradação) Iniciei processo de retreinamento

### Dados
- [ ] Executei relatório Evidently de drift semanal (4× no mês)
- [ ] Verifiquei % de features com drift → resultado: ___% 
- [ ] Verifiquei % de valores faltantes por feature → ok / problema em: ___
- [ ] Verifiquei outliers em monthlyCharges e totalCharges

### Operacional
- [ ] Verifiquei logs de erros 5xx → contagem: ___
- [ ] Verifiquei latência P95 → resultado: ___ms
- [ ] Verifiquei uptime no período → resultado: ___%
- [ ] Confirmo que MLflow tem runs registradas no período

### Ações
- [ ] Registrei alertas disparados: ___
- [ ] Registrei ações tomadas: ___
- [ ] Atualizado Model Card se houve retreinamento: sim / não
```

---

**Documento criado por:** Giovanni de Aguirre Tamanini  
**Data:** Maio 2026  
**Versão:** 1.0
