from typing import Literal

from pydantic import BaseModel


class CustomerFeatures(BaseModel):
    gender: Literal["male", "female"]
    isSeniorCitizen: bool | int
    hasPartner: bool | int
    hasDependents: bool | int
    tenureMonths: float
    hasPhoneService: bool | int
    hasPaperlessBilling: bool | int
    monthlyCharges: float
    totalCharges: float
    multipleLinesNoPhone: bool | int
    multipleLinesActive: bool | int
    internetFiberOptic: bool | int
    internetNone: bool | int
    onlineSecurityNoInternet: bool | int
    onlineSecurityActive: bool | int
    onlineBackupNoInternet: bool | int
    onlineBackupActive: bool | int
    deviceProtectionNoInternet: bool | int
    deviceProtectionActive: bool | int
    techSupportNoInternet: bool | int
    techSupportActive: bool | int
    streamingTvNoInternet: bool | int
    streamingTvActive: bool | int
    streamingMoviesNoInternet: bool | int
    streamingMoviesActive: bool | int
    contractOneYear: bool | int
    contractTwoYear: bool | int
    paymentCreditCardAutomatic: bool | int
    paymentElectronicCheck: bool | int
    paymentMailedCheck: bool | int


class BatchPredictRequest(BaseModel):
    customers: list[CustomerFeatures]
