
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar


class ModelLike(Protocol):
    """Minimal model interface used by predictors."""

    def predict(self, features):
        ...


TModel = TypeVar("TModel", bound=ModelLike, covariant=True)
TRegressor = TypeVar("TRegressor", bound=ModelLike, covariant=True)
TClassifier = TypeVar("TClassifier", bound=ModelLike, covariant=True)


class FoldModelBankLike(Protocol[TModel]):
    """Interface for fold-routed models keyed by player id."""

    def get_predictor_model(self, player_id: int) -> TModel:
        ...


class PredictorModelBankLike(Protocol[TRegressor, TClassifier]):
    """Interface for metric-specific fold model banks."""
    @property
    def base_load_model_bank(self) -> FoldModelBankLike[TRegressor]:
        ...

    @property
    def pv_gen_model_bank(self) -> FoldModelBankLike[TRegressor]:
        ...

    @property
    def ev1_status_model_bank(self) -> FoldModelBankLike[TClassifier]:
        ...

    @property
    def ev2_status_model_bank(self) -> FoldModelBankLike[TClassifier]:
        ...


@dataclass
class FoldModelBank(Generic[TModel]):
    """Model bank for one fold"""
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

    