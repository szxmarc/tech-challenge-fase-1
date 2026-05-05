# Model Card — Previsão de Churn Telco

## 1. Visão Geral do Modelo

| Campo | Valor |
|---|---|
| **Nome** | Churn Prediction MLP PyTorch |
| **Versão** | 1.0 |
| **Tarefa** | Classificação binária — previsão de churn de clientes |
| **Framework** | PyTorch 2.0+ (produção) / Scikit-learn (baseline) |
| **Responsável** | Giovanni de Aguirre Tamanini |
| **Data** | Maio 2026 |
| **Status** | Produção |

---

## 2. Uso Pretendido

### Uso Indicado

- Identificar clientes de uma operadora de telecomunicações com risco de cancelamento (churn)
- Priorizar ações preventivas da equipe de retenção
- Geração de scores de propensão para campanhas de marketing direcionadas

### Usuários Esperados

- Equipe de Retenção de Clientes (usuário principal)
- Equipe de Marketing (usuário secundário)
- Sistemas de CRM que consomem a API

### Usos Fora do Escopo

- Detecção de fraude em tempo real (modelo não é otimizado para latência sub-milissegundo)
- Scoring de crédito ou avaliação de risco financeiro
- Aplicação em outros setores além de telecomunicações sem retreinamento
- Tomada de decisão automatizada sem revisão humana

---

## 3. Arquitetura do Modelo

### Modelo de Produção: MLP PyTorch

```
Entrada (N features selecionadas)
     ↓
Linear(N → 64) + BatchNorm + ReLU + Dropout(0.3)
     ↓
Linear(64 → 32) + BatchNorm + ReLU + Dropout(0.3)
     ↓
Linear(32 → 1) + Sigmoid
     ↓
Saída: probabilidade de churn [0, 1]
```

| Hiperparâmetro | Valor |
|---|---|
| Camadas ocultas | (64, 32) |
| Dropout | 0.3 |
| Função de perda | BCEWithLogitsLoss com `pos_weight` |
| Otimizador | Adam (lr=1e-3) |
| Épocas máximas | 100 |
| Batch size | 64 |
| Early stopping patience | 10 épocas |
| Threshold de decisão | 0.5 |

**Tratamento do desbalanceamento:** `pos_weight = n_negativos / n_positivos` (~2.8) aplicado
no BCEWithLogitsLoss, equivalente ao `class_weight='balanced'` do scikit-learn.

**Early stopping:** monitorado na `val_loss` de 10% do conjunto de treino (split estratificado).
O estado de melhor `val_loss` é restaurado ao final.

### Modelo Baseline: Regressão Logística (scikit-learn)

| Hiperparâmetro | Valor |
|---|---|
| Solver | lbfgs |
| Regularização C | 1.0 (L2) |
| class_weight | balanced |
| max_iter | 1000 |

---

## 4. Dados de Treinamento

| Campo | Valor |
|---|---|
| **Dataset** | Telco Customer Churn (IBM / Kaggle) |
| **Volume** | 7.043 clientes |
| **Features originais** | 20 variáveis + 1 target |
| **Balanceamento** | 26.5% churn (classe positiva), 73.5% não-churn |
| **Split** | 80% treino / 20% teste, estratificado por classe |
| **Pré-processamento** | StandardScaler (ajustado apenas no treino) |

**Variáveis do dataset:**
- Demográficas: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Serviços: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`,
  `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- Contratuais: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`,
  `MonthlyCharges`, `TotalCharges`
- Target: `Churn` (Yes/No → 1/0)

---

## 5. Engenharia de Features

### Encoding das Variáveis Categóricas

- **Variáveis binárias** (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`):
  LabelEncoder → valores 0/1
- **Variáveis multi-classe** (`MultipleLines`, `InternetService`, `Contract`, `PaymentMethod`, etc.):
  `pd.get_dummies` com `drop_first=True` (elimina multicolinearidade)

### Features Derivadas (6 novas features)

| Feature | Fórmula | Intuição |
|---|---|---|
| `charges_per_month` | `TotalCharges / tenure` (ou `MonthlyCharges` se `tenure==0`) | Valor médio gasto por mês de contrato |
| `is_monthly` | `1 - Contract_One year - Contract_Two year` | Flag de contrato mensal (maior risco de churn) |
| `service_count` | Soma dos 6 serviços de internet | Clientes com mais serviços tendem a ficar |
| `has_no_services` | `1` se `service_count == 0` | Sem serviços adicionais = maior risco |
| `is_senior_alone` | `SeniorCitizen × (1 - Partner)` | Sênior sem parceiro = perfil vulnerável |
| `charges_x_monthly` | `MonthlyCharges × is_monthly` | Interação: caro + mensal = risco alto |

### Seleção de Features (Random Forest)

Features ordenadas por importância decrescente via `RandomForestClassifier(n_estimators=100)`.
São selecionadas as N features que cobrem ≥ 90% da importância acumulada.
O contrato final da API é persistido em `models/feature_names.txt` a cada startup.

---

## 6. Avaliação

### Métricas Alvo

| Métrica | Meta | Justificativa |
|---|---|---|
| AUC-ROC | ≥ 0.80 | Capacidade geral de discriminação |
| Recall | ≥ 0.75 | Minimizar falsos negativos (cliente perdido) |
| PR-AUC | ≥ 0.65 | Foco na classe minoritária |
| F2-Score | ≥ 0.70 | Pondera recall 2× mais que precision |

### Resultados dos Modelos (conjunto de teste, 1.409 amostras)

| Métrica | Regressão Logística | MLP PyTorch | Meta | Status |
|---|---|---|---|---|
| AUC-ROC | 0.841 | ver `/api/v1/model/comparison` | ≥ 0.80 | ✅ LR aprovada |
| Recall | ver endpoint | ver endpoint | ≥ 0.75 | — |
| PR-AUC | ver endpoint | ver endpoint | ≥ 0.65 | — |
| F1-Score | ver endpoint | ver endpoint | — | — |
| Valor de Negócio | +R$ 61.500 | ver endpoint | positivo | ✅ LR aprovada |

> Resultados completos e atualizados disponíveis em `GET /api/v1/model/comparison`
> (gerado em cada startup com os dados mais recentes).

### Matriz de Custo-Benefício

| Decisão | Custo | Descrição |
|---|---|---|
| Verdadeiro Positivo (TP) | **+R$ 400** | Cliente retido com sucesso |
| Falso Positivo (FP) | **-R$ 50** | Ação de retenção desnecessária |
| Falso Negativo (FN) | **-R$ 500** | Cliente perdido sem identificação |
| Verdadeiro Negativo (TN) | **R$ 0** | Previsão correta de não-churn |

**Implicação estratégica:** FN custa 10× mais que FP → modelo prioriza Recall alto.

---

## 7. Como Usar (API)

### Predição Individual

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female",
    "isSeniorCitizen": 0,
    "hasPartner": 0,
    "hasDependents": 0,
    "tenureMonths": 2,
    "hasPhoneService": 1,
    "hasPaperlessBilling": 1,
    "monthlyCharges": 95.0,
    "totalCharges": 190.0,
    "multipleLinesNoPhone": 0,
    "multipleLinesActive": 0,
    "internetFiberOptic": 1,
    "internetNone": 0,
    "onlineSecurityNoInternet": 0,
    "onlineSecurityActive": 0,
    "onlineBackupNoInternet": 0,
    "onlineBackupActive": 0,
    "deviceProtectionNoInternet": 0,
    "deviceProtectionActive": 0,
    "techSupportNoInternet": 0,
    "techSupportActive": 0,
    "streamingTvNoInternet": 0,
    "streamingTvActive": 0,
    "streamingMoviesNoInternet": 0,
    "streamingMoviesActive": 0,
    "contractOneYear": 0,
    "contractTwoYear": 0,
    "paymentCreditCardAutomatic": 0,
    "paymentElectronicCheck": 1,
    "paymentMailedCheck": 0
  }'
```

**Resposta esperada:**
```json
{
  "prediction": 1,
  "predictionLabel": "Churn",
  "probabilityNoChurn": "23.0%",
  "probabilityChurn": "77.0%",
  "confidence": "77.0%"
}
```

---

## 8. Limitações

- **Dataset estático:** sem divisão temporal — o modelo pode não capturar sazonalidade
- **Domínio único:** treinado em dados de uma operadora específica; transferência para outros mercados requer retreinamento
- **Threshold fixo:** threshold padrão de 0.5 não foi otimizado para maximizar valor de negócio
- **Sem SHAP:** não há explicabilidade por instância implementada
- **Sem auditoria de fairness:** disparate impact por gênero e faixa etária não foi formalmente verificado
- **Treinamento no startup:** cada reinício do servidor re-treina os modelos, o que pode variar levemente por não-determinismo do PyTorch em GPU

---

## 9. Considerações Éticas

- **LGPD:** o modelo usa dados pessoais (`gender`, `SeniorCitizen`). O uso em produção requer conformidade com a LGPD, incluindo base legal para tratamento e direito de oposição
- **Viés demográfico:** `gender` e `SeniorCitizen` são features do modelo. Monitorar se o modelo trata grupos demograficamente distintos com taxas de falso positivo/negativo desproporcionais
- **Decisão humana:** o modelo deve apoiar, não substituir, a decisão da equipe de retenção. Clientes identificados como churn devem ser abordados por humanos
- **Transparência:** a equipe de retenção deve ser informada de que as indicações provêm de um modelo preditivo e não de certeza absoluta

---

**Documento criado por:** Giovanni de Aguirre Tamanini  
**Data:** Maio 2026  
**Versão:** 1.0
