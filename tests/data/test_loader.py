from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.loader import load_processed_data, load_raw_data


def test_load_raw_data_retorna_dataframe(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("col1,col2\n1,2\n3,4\n")
    result = load_raw_data(filepath=csv_file)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)


def test_load_raw_data_levanta_erro_arquivo_ausente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(filepath=tmp_path / "inexistente.csv")


def test_load_raw_data_usa_path_padrao_quando_none():
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    with patch("src.data.loader.RAW_DATASET", mock_path), \
         patch("pandas.read_csv", return_value=pd.DataFrame({"a": [1]})):
        result = load_raw_data()
    assert isinstance(result, pd.DataFrame)


def test_load_processed_data_levanta_erro_sem_x_train(tmp_path):
    missing = tmp_path / "X_train.csv"
    with patch("src.data.loader.X_TRAIN_FILE", missing):
        with pytest.raises(FileNotFoundError):
            load_processed_data()


def test_load_processed_data_retorna_5_elementos(tmp_path):
    import joblib
    from sklearn.preprocessing import StandardScaler

    X = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y = pd.Series([0, 1])
    scaler = StandardScaler()

    x_train = tmp_path / "X_train.csv"
    x_test = tmp_path / "X_test.csv"
    y_train = tmp_path / "y_train.csv"
    y_test = tmp_path / "y_test.csv"
    scaler_file = tmp_path / "scaler.joblib"

    X.to_csv(x_train, index=False)
    X.to_csv(x_test, index=False)
    y.to_csv(y_train, index=False)
    y.to_csv(y_test, index=False)
    joblib.dump(scaler, scaler_file)

    with patch("src.data.loader.X_TRAIN_FILE", x_train), \
         patch("src.data.loader.X_TEST_FILE", x_test), \
         patch("src.data.loader.Y_TRAIN_FILE", y_train), \
         patch("src.data.loader.Y_TEST_FILE", y_test), \
         patch("src.data.loader.SCALER_FILE", scaler_file):
        result = load_processed_data()

    assert len(result) == 5
