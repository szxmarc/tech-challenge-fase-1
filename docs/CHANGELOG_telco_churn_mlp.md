# Documentação de Mudanças - `telco_churn_mlp.ipynb`

## 📝 Alterações Realizadas

Durante a reorganização do projeto em **30 de Abril de 2026**, foram realizadas alterações no notebook para melhorar a organização e consistência com a estrutura do projeto.

---

## 1. Ajuste do Caminho de Leitura de Dados

### Mudança:
```python
# ANTES
df = pd.read_csv("Telco-Customer-Churn.csv")

# DEPOIS
df = pd.read_csv("../data/raw/Telco-Customer-Churn.csv")
```

**Linha:** ~89

**Motivação:**
- O notebook foi movido de `docs/` para `notebooks/`
- Necessário ajustar caminho relativo para acessar `data/raw/`
- Alinhamento com estrutura padrão do projeto

**Impacto:**
- ✅ Notebook funciona corretamente da pasta `notebooks/`
- ✅ Consistente com `01_eda_e_baselines.ipynb`
- ✅ Segue convenções de organização de projetos ML

---

## 2. Configuração do MLflow Tracking URI

### Mudança:
```python
# ANTES
import mlflow
import mlflow.pytorch

mlflow.set_experiment("telco-churn")

# DEPOIS
import mlflow
import mlflow.pytorch
import os

# Configurar tracking URI para usar o mesmo diretório do outro notebook
mlruns_dir = "../mlruns"
os.makedirs(mlruns_dir, exist_ok=True)
mlflow.set_tracking_uri(f"file:{mlruns_dir}")

mlflow.set_experiment("telco-churn")
```

**Linhas:** ~1138-1147

**Motivação:**
- Centralizar tracking de experimentos em um único diretório
- Facilitar comparação entre experimentos de diferentes notebooks
- Garantir que todos os runs do MLflow sejam salvos em `mlruns/` na raiz

**Impacto:**
- ✅ Experimentos do MLP salvos em `../mlruns/`
- ✅ Possível visualizar todos os experimentos com: `mlflow ui --backend-store-uri mlruns/`
- ✅ Consistente com configuração do notebook `01_eda_e_baselines.ipynb`

---

## 3. Movimentação do Arquivo

### Mudança:
```
ANTES: docs/telco_churn_mlp.ipynb
DEPOIS: notebooks/telco_churn_mlp.ipynb
```

**Motivação:**
- Notebooks devem estar em `notebooks/`, não em `docs/`
- Pasta `docs/` é reservada para documentação (markdown, PDFs)
- Melhor organização e clareza da estrutura do projeto

**Impacto:**
- ✅ Estrutura de diretórios mais profissional
- ✅ Facilita localização de notebooks
- ✅ Segue padrões da comunidade Data Science

---

## 📋 Resumo das Alterações

| Aspecto | Antes | Depois |
|---------|-------|--------|
| **Localização** | `docs/` | `notebooks/` |
| **Leitura CSV** | `Telco-Customer-Churn.csv` | `../data/raw/Telco-Customer-Churn.csv` |
| **MLflow URI** | Não configurado | `../mlruns` |
| **Consistência** | Diferente do outro notebook | Alinhado com `01_eda_e_baselines.ipynb` |

---

## ✅ Verificação de Funcionalidade

Após as mudanças, o notebook:

- ✅ Carrega dados corretamente do CSV em `data/raw/`
- ✅ Salva experimentos MLflow em `../mlruns/`
- ✅ Mantém compatibilidade com estrutura do projeto
- ✅ Funciona de forma independente quando executado da pasta `notebooks/`

---

## 🔍 Antes vs. Depois

### Antes das mudanças:
- ❌ Notebook em local inadequado (`docs/`)
- ❌ Caminhos hardcoded
- ❌ MLflow tracking desorganizado
- ❌ Inconsistente com outros notebooks

### Depois das mudanças:
- ✅ Estrutura organizada e profissional
- ✅ Caminhos relativos corretos
- ✅ MLflow centralizado
- ✅ Consistente com todo o projeto
- ✅ Facilita manutenção e colaboração

---

## 📚 Arquivos Relacionados Atualizados

### `docs/ml_canvas.md`
- Seção 9 adicionada com evolução das métricas de negócio
- Documentação completa da diferença entre os dois notebooks
- Comparação técnica detalhada
- Justificativa para decisões de modelagem

---

**Documento criado:** 30 de Abril de 2026  
**Status:** ✅ Mudanças implementadas e validadas
