# ML Canvas - Previsão de Churn em Telecomunicações

## 1. PROBLEMA DE NEGÓCIO

### Contexto
Uma operadora de telecomunicações está enfrentando uma taxa acelerada de cancelamento de clientes (churn), impactando diretamente a receita e o crescimento da empresa.

### Objetivo de Negócio
Reduzir a taxa de churn trimestral em 15% através da identificação proativa de clientes em risco de cancelamento, permitindo ações preventivas de retenção.

### Stakeholders Principais

| Stakeholder | Papel | Interesse |
|------------|-------|-----------|
| **Diretoria Executiva** | Tomador de decisão final | Aumento da rentabilidade e redução de custos de aquisição |
| **Gerente de Retenção de Clientes** | Usuário principal do modelo | Identificar clientes em risco para ações de retenção |
| **Equipe de Marketing** | Usuário secundário | Criar campanhas direcionadas de retenção |
| **Equipe de CRM** | Fornecedor de dados | Prover dados de interação com clientes |
| **Equipe de Suporte Técnico** | Fornecedor de dados | Fornecer histórico de reclamações e problemas |
| **Time de Data Science** | Desenvolvedor do modelo | Construir e manter o modelo preditivo |
| **Equipe de TI/Infraestrutura** | Suporte técnico | Prover infraestrutura e deployment |

---

## 2. MÉTRICAS DE NEGÓCIO

### KPIs Principais

1. **Taxa de Churn Evitado**
   - Meta: Reduzir churn em 15% em 3 meses
   - Baseline atual: Assumir 25% de churn trimestral
   - Meta: Reduzir para 21.25% ou menos

2. **ROI da Solução de ML**
   - Custo de ação de retenção: R$ 50 por cliente
   - Custo de aquisição de novo cliente: R$ 500
   - Valor médio mensal do cliente (LTV): R$ 100/mês
   - Break-even: Se evitar churn de 1 cliente > economiza R$ 450

3. **Taxa de Conversão de Retenção**
   - % de clientes identificados pelo modelo que foram retidos após ação
   - Meta: > 30% dos clientes em risco abordados sejam retidos

### Análise de Custo-Benefício

| Cenário | Custo | Benefício | Resultado |
|---------|-------|-----------|-----------|
| **Falso Positivo** | R$ 50 (ação desnecessária) | R$ 0 | -R$ 50 |
| **Falso Negativo** | R$ 0 (sem ação) | -R$ 500 (perda cliente) | -R$ 500 |
| **Verdadeiro Positivo** | R$ 50 (ação) | R$ 450 (retenção) | +R$ 400 |
| **Verdadeiro Negativo** | R$ 0 | R$ 0 | R$ 0 |

**Análise:** Falsos Negativos são 10x mais custosos que Falsos Positivos. O modelo deve priorizar **Recall** para minimizar clientes em risco não identificados.

---

## 3. OBJETIVOS TÉCNICOS DE ML

### Tipo de Problema
Classificação binária supervisionada:
- **Classe 0:** Cliente não vai dar churn (negativo)
- **Classe 1:** Cliente vai dar churn (positivo)

### Métricas Técnicas Principais

1. **AUC-ROC (Area Under ROC Curve)**
   - Meta: ≥ 0.80
   - Mede a capacidade geral de discriminação do modelo

2. **Recall (Sensibilidade)**
   - Meta: ≥ 0.75
   - Prioridade: Identificar o máximo de clientes em risco
   - Justificativa: Custo alto de FN (perder cliente)

3. **Precision-Recall AUC**
   - Meta: ≥ 0.65
   - Importante para datasets desbalanceados (churn é evento raro)

4. **F2-Score**
   - Meta: ≥ 0.70
   - F2 pondera mais Recall que Precision (β=2)
   - Alinhado com a estratégia de negócio

### Threshold de Decisão
- Threshold padrão (0.5) pode não ser ótimo
- Otimizar threshold considerando custo de FP vs FN
- Análise de curva Precision-Recall para definir ponto de operação

---

## 4. DADOS

### Fontes de Dados

| Fonte | Tipo de Dados | Disponibilidade |
|-------|---------------|-----------------|
| **CRM** | Dados demográficos, plano contratado, tempo de contrato | Alta |
| **Sistema de Billing** | Histórico de pagamentos, atrasos, valor da conta | Alta |
| **SAC/Suporte** | Histórico de reclamações, chamados abertos | Média |
| **Sistema de Uso** | Consumo de dados, minutos, SMS | Alta |
| **Dados externos** | Ofertas da concorrência, cobertura de rede | Baixa |

### Variáveis Candidatas

**Demográficas:**
- Idade, gênero, localização
- Tempo como cliente (tenure)

**Comportamentais:**
- Média de consumo mensal (dados, minutos)
- Variação no padrão de uso
- Frequência de contato com SAC

**Contratuais:**
- Tipo de plano
- Valor da conta mensal
- Contrato (pós/pré-pago)
- Presença de dependentes no plano

**Financeiras:**
- Histórico de atrasos
- Mudanças recentes no valor da conta
- Métodos de pagamento

**Engajamento:**
- Uso de app da operadora
- Adesão a serviços adicionais
- Participação em programas de fidelidade

### Data Readiness

- Volume esperado: ≥ 5.000 clientes
- Features esperadas: ≥ 10 variáveis
- Período histórico: 12-24 meses
- Rotulação: Status de churn conhecido (churned sim/não)

---

## 5. SLOs (SERVICE LEVEL OBJECTIVES)

### Performance do Modelo

| Métrica | SLO | Medição |
|---------|-----|---------|
| AUC-ROC | ≥ 0.80 | Avaliação offline no conjunto de teste |
| Recall | ≥ 0.75 | Avaliação offline no conjunto de teste |
| F2-Score | ≥ 0.70 | Avaliação offline no conjunto de teste |
| Latência de Predição | < 100ms | P95 em produção |
| Throughput | ≥ 100 predições/segundo | Pico de uso |

### Qualidade dos Dados

| Aspecto | SLO | Ação |
|---------|-----|------|
| Dados faltantes | < 10% por feature | Imputação ou exclusão de feature |
| Valores outliers | < 5% por feature | Análise e tratamento |
| Atualização dos dados | Diária | Pipeline automatizado |

### Operacional

| Aspecto | SLO | Detalhes |
|---------|-----|----------|
| Uptime da API | ≥ 99% | Disponibilidade do serviço |
| Data Drift Detection | Semanal | Monitoramento de distribuição |
| Retreinamento | Mensal | Ou quando performance cair 5% |
| Tempo de resposta a incidentes | < 2 horas | Durante horário comercial |

---

## 6. PREMISSAS E RESTRIÇÕES

### Premissas

1. Clientes identificados em risco serão contatados pela equipe de retenção em até 48h
2. A equipe de retenção tem capacidade de abordar até 500 clientes/semana
3. Taxa de sucesso de retenção após contato: 30%
4. Dados históricos de churn estão rotulados corretamente
5. Padrões passados de churn são indicativos de comportamento futuro

### Restrições

1. **LGPD/Privacidade:**
   - Uso de dados pessoais deve estar em conformidade com LGPD
   - Consentimento explícito dos clientes
   - Anonimização quando possível

2. **Técnicas:**
   - Infraestrutura limitada (cloud ou on-premise)
   - Prazo de desenvolvimento: 3 meses
   - Orçamento: [A definir]

3. **Negócio:**
   - Modelo deve ser explicável para a equipe de retenção
   - Não pode criar viés discriminatório (age, gender, location)
   - Deve integrar com sistema CRM existente

---

## 7. RISCOS E MITIGAÇÕES

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| **Dados insuficientes ou de baixa qualidade** | Média | Alto | EDA detalhada + validação com SME |
| **Modelo com viés discriminatório** | Média | Alto | Análise de fairness + auditorias |
| **Baixa adoção pela equipe de retenção** | Média | Alto | Treinamento + interface amigável |
| **Mudança no comportamento de churn** | Baixa | Alto | Monitoramento contínuo + retreinamento |
| **Sobrecarga da equipe de retenção** | Alta | Médio | Priorização por score de propensão |
| **Data drift** | Alta | Alto | Monitoramento automatizado + alertas |

---

## 8. PLANO DE VALIDAÇÃO

### Offline Validation

1. **Divisão dos dados:**
   - Train: 60%
   - Validation: 20%
   - Test: 20%
   - Estratificação por classe (churn)

2. **Validação cruzada:**
   - K-Fold estratificada (k=5)
   - Avaliar estabilidade das métricas

3. **Comparação com baselines:**
   - DummyClassifier (estratificado)
   - Regressão Logística
   - Árvore de Decisão

### Online Validation (A/B Testing)

1. **Fase Piloto (1 mês):**
   - 10% dos clientes em risco identificados
   - Comparar com método atual (se existir)
   - Medir taxa de retenção efetiva

2. **Rollout Gradual:**
   - 25% → 50% → 100% em 2 meses
   - Monitorar KPIs continuamente

---

## 9. EVOLUÇÃO DO PROJETO E MODELAGEM

### 9.1 Histórico de Mudanças nas Métricas de Negócio

Durante o desenvolvimento do projeto, as métricas de negócio foram refinadas para melhor refletir a realidade operacional:

#### **Versão 1.0 - Notebook `01_eda_e_baselines.ipynb` (Baseline Simplificado)**

**Objetivo:** Estabelecer linha de base e validar viabilidade do projeto

**Escopo:**
- EDA completa e análise de data readiness
- Modelos baseline: DummyClassifier + Logistic Regression
- MLflow tracking configurado

**Métricas de Negócio (Simplificadas):**
```
Premissas:
- Custo de ação de retenção:  R$ 50
- Custo de perder cliente:    R$ 500 (fixo)
- Benefício de retenção:      R$ 400 (fixo)
- Ratio FN/FP:                10x

Fórmula:
Valor Total = (TP × 400) - (FP × 50) - (FN × 500)
```

**Resultados:**
- Logistic Regression: +R$ 61.500 (lucro líquido)
- AUC-ROC: 0.841 ✅
- F2-Score: 0.704 ✅
- Recall: 0.781 ✅

**Justificativa:** Abordagem direta para comparação rápida de baselines e validação inicial do problema de negócio.

---

#### **Versão 2.0 - Notebook `telco_churn_mlp.ipynb` (Modelagem Avançada)**

**Objetivo:** Superar baselines com modelos complexos e otimizar valor de negócio

**Escopo:**
- Modelos avançados: Random Forest + MLP (PyTorch)
- Explicabilidade com SHAP values
- Otimização de threshold por valor de negócio
- Feature selection baseada em importância

**Métricas de Negócio (Baseadas em Dados Reais):**
```
Premissas Refinadas:
- Receita mensal média:       R$ 65 (dados reais: MonthlyCharges)
- Período de retenção:        12 meses
- Taxa de sucesso campanha:   30%
- Custo por contato:          R$ 50
- LTV (Lifetime Value):       R$ 780 (65 × 12)
- Ratio FN/FP:                15.6x

Fórmula:
Valor Cliente Retido = Receita Mensal × Meses × Taxa Sucesso
Valor TP = (R$ 65 × 12 × 30%) - R$ 50 = +R$ 184
Valor FP = -R$ 50 (campanha desperdiçada)
Valor FN = -R$ 780 (receita perdida em 12 meses)
Valor TN = R$ 0

Lucro Líquido = (TP × 234) - (Clientes Contatados × 50) - (FN × 780)
```

**Resultados:**
- Logistic Regression: +R$ 40.850
- Random Forest: +R$ 38.946
- MLP (threshold=0.5): +R$ 39.878
- AUC-ROC (todos): ≥ 0.836 ✅

**Justificativa:** 
- Modelagem mais realista considerando LTV
- Taxa de sucesso de 30% baseada em benchmarks de indústria
- Permite otimização de threshold por valor de negócio
- Melhor alinhamento com estratégia de retenção

---

### 9.2 Comparação Técnica entre Notebooks

| Aspecto | Notebook 1 (Baseline) | Notebook 2 (Avançado) |
|---------|----------------------|----------------------|
| **Modelos** | Dummy + Logistic Regression | Dummy + LR + Random Forest + MLP |
| **Framework** | Scikit-learn | Scikit-learn + PyTorch |
| **Explicabilidade** | Coeficientes LR | SHAP values + Feature Importance |
| **Threshold** | Fixo (0.5) | Otimizado (0.5 / p80 / otimizado) |
| **Métricas Extras** | Básicas | MCC, Brier, Log Loss, Precision@K, KS |
| **Feature Selection** | Não | Sim (baseada em RF) |
| **Early Stopping** | N/A | Sim (MLP) |
| **Visualizações** | ROC, PR, Confusion | + Curvas de treino, SHAP plots, Threshold × Lucro |

---

### 9.3 Evolução dos Modelos

#### **Random Forest (Novo no Notebook 2)**

**Configuração:**
- 300 estimators
- Class weight balanced
- Análise de feature importance por impureza e permutação
- SHAP values para explicabilidade individual

**Papel:**
1. Baseline robusto para comparação
2. Ranqueamento de features
3. Seleção de features para MLP
4. Explicabilidade via SHAP

#### **MLP - Multi-Layer Perceptron (Novo no Notebook 2)**

**Arquitetura:**
```
Input Layer (features selecionadas)
    ↓
Hidden Layer 1 (128 neurons) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Hidden Layer 2 (64 neurons) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Output Layer (1 neuron) → Sigmoid
```

**Configuração:**
- Framework: PyTorch
- Otimizador: Adam (lr=0.001)
- Loss: BCEWithLogitsLoss (pos_weight para desbalanceamento)
- Early Stopping: patience=10 épocas
- Batch size: 64

**Justificativa:**
- Captura interações não-lineares complexas
- BatchNorm estabiliza treinamento
- Dropout previne overfitting
- Class weights lidam com desbalanceamento

---

### 9.4 Otimização de Threshold (Inovação do Notebook 2)

Análise de **3 estratégias** de threshold:

| Estratégia | Threshold | Lucro Líquido | Uso |
|------------|-----------|---------------|-----|
| **Padrão** | 0.50 | ~R$ 40k | Comparação baseline |
| **Percentil 80 (p80)** | ~0.42 | ~R$ 29k | Top 20% de risco |
| **Otimizado (negócio)** | Variável | Máximo | Maximiza lucro |

**Benefício:** Permite escolher threshold baseado em capacidade operacional da equipe de retenção e objetivo de negócio.

---

### 9.5 Comparação de Resultados Finais

| Métrica | Notebook 1 (Simples) | Notebook 2 (LTV) | Diferença |
|---------|---------------------|------------------|-----------|
| **Logistic Regression** | +R$ 61.500 | +R$ 40.850 | -33% |
| **Custo FN** | R$ 500 | R$ 780 | +56% |
| **Benefício TP** | R$ 400 | R$ 234 (líquido) | -41% |
| **Ratio FN/FP** | 10x | 15.6x | +56% |
| **AUC-ROC (LR)** | 0.841 | 0.846 | +0.6% |

**Observação:** Os valores absolutos diferem devido à modelagem mais conservadora (taxa de sucesso 30%), mas **ambas as abordagens validam** que os modelos geram valor positivo e superam significativamente o baseline aleatório.

---

### 9.6 Decisão Final e Padrão Adotado

**Adotamos a Versão 2.0** como padrão do projeto por:

1. ✅ **Realismo:** Baseada em dados reais do dataset (R$ 65 médio)
2. ✅ **Conservadorismo:** Taxa de sucesso 30% é mais realista que 100%
3. ✅ **Flexibilidade:** Otimização de threshold por objetivo de negócio
4. ✅ **Explicabilidade:** SHAP values para justificar decisões
5. ✅ **Escalabilidade:** MLP pode incorporar novas features facilmente
6. ✅ **Alinhamento:** LTV de 12 meses alinha com estratégias de retenção
7. ✅ **Rastreabilidade:** MLflow tracking completo de experimentos

**Ambos os notebooks são mantidos:**
- **Notebook 1:** Referência de baseline e validação rápida
- **Notebook 2:** Modelagem avançada e deploy em produção

