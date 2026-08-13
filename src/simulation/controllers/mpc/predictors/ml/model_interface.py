
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar


class ModelLike(Protocol):
    """Minimal model interface used by predictors."""

    def predict(self, features):
        ...


TModel = TypeVar("TModel", bound=ModelLike)
TRegressor = TypeVar("TRegressor", bound=ModelLike)
TClassifier = TypeVar("TClassifier", bound=ModelLike)


class FoldModelBankLike(Protocol[TModel]):
    """Interface for fold-routed models keyed by player id."""

    def get_predictor_model(self, player_id: int) -> TModel:
        ...


class PredictorModelBankLike(Protocol[TRegressor, TClassifier]):
    """Interface for metric-specific fold model banks."""

    base_load_model_bank: FoldModelBankLike[TRegressor]
    pv_gen_model_bank: FoldModelBankLike[TRegressor]
    ev1_status_model_bank: FoldModelBankLike[TClassifier]
    ev2_status_model_bank: FoldModelBankLike[TClassifier]


@dataclass
class FoldModelBank(Generic[TModel]):
    models_by_fold: dict[str, TModel]
    id_to_fold: dict[int, str]

    def get_predictor_model(self, player_id: int) -> TModel:
        fold_id = self.id_to_fold[int(player_id)]
        return self.models_by_fold[fold_id]


@dataclass
class PredictorModelBank(Generic[TRegressor, TClassifier]):
    base_load_model_bank: FoldModelBank[TRegressor]
    pv_gen_model_bank: FoldModelBank[TRegressor]
    ev1_status_model_bank: FoldModelBank[TClassifier]
    ev2_status_model_bank: FoldModelBank[TClassifier]

    