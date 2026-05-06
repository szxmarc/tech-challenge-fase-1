from src.prediction.feature_mapping import (
    API_FEATURE_NAMES,
    FEATURE_NAME_MAPPING,
    MODEL_FEATURE_NAMES,
    to_model_input,
)


def test_mapeamento_contem_30_features():
    assert len(FEATURE_NAME_MAPPING) == 30


def test_nomes_api_e_modelo_tem_mesmo_tamanho():
    assert len(API_FEATURE_NAMES) == len(MODEL_FEATURE_NAMES)


def test_nomes_api_nao_contem_espacos():
    for name in API_FEATURE_NAMES:
        assert " " not in name, f"'{name}' contém espaço"


def test_to_model_input_converte_tenure(sample_input):
    result = to_model_input(sample_input)
    assert "tenure" in result
    assert "tenureMonths" not in result


def test_to_model_input_converte_senior_citizen(sample_input):
    result = to_model_input(sample_input)
    assert "SeniorCitizen" in result
    assert "isSeniorCitizen" not in result


def test_to_model_input_preserva_valores(sample_input):
    result = to_model_input(sample_input)
    assert result["tenure"] == sample_input["tenureMonths"]
    assert result["MonthlyCharges"] == sample_input["monthlyCharges"]
    assert result["gender"] == sample_input["gender"]


def test_to_model_input_ignora_chaves_desconhecidas():
    result = to_model_input({"chaveDesconhecida": 99, "gender": 0})
    assert "chaveDesconhecida" not in result
    assert "gender" in result


def test_ordem_api_feature_names_preservada():
    assert API_FEATURE_NAMES[0] == "gender"
    assert API_FEATURE_NAMES[4] == "tenureMonths"
    assert API_FEATURE_NAMES[-1] == "paymentMailedCheck"


def test_ordem_model_feature_names_preservada():
    assert MODEL_FEATURE_NAMES[0] == "gender"
    assert MODEL_FEATURE_NAMES[4] == "tenure"


def test_contract_features_sem_espacos():
    assert "Contract_One year" in MODEL_FEATURE_NAMES
    assert "contractOneYear" in API_FEATURE_NAMES
    assert "contractOneYear" not in MODEL_FEATURE_NAMES
