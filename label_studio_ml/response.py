
from typing import Dict, Optional, List, Any, Union
from pydantic import BaseModel

try:
    from label_studio_sdk.label_interface.objects import PredictionValue
except Exception:  # pragma: no cover - fallback for SDK layout differences
    class PredictionValue(BaseModel):
        result: List[Dict[str, Any]]
        score: float = 0.0
        model_version: Optional[str] = None


# one or multiple predictions per task
SingleTaskPredictions = Union[List[PredictionValue], PredictionValue]


class ModelResponse(BaseModel):
    """ Model response with predictions for Label Studio, used in /predict API endpoint
    """
    class Config:
        protected_namespaces = ('__.*__', '_.*')  # Excludes 'model_'

    model_version: Optional[str] = None
    predictions: List[SingleTaskPredictions]

    def has_model_version(self) -> bool:
        return bool(self.model_version)

    def update_predictions_version(self) -> None:
        """
        """
        for prediction in self.predictions:
            if isinstance(prediction, PredictionValue):
                prediction = [prediction]
            for p in prediction:
                if not p.model_version:
                    p.model_version = self.model_version
    
    def set_version(self, version: str) -> None:
        """
        """
        self.model_version = version
        # Set the version for each prediction
        self.update_predictions_version()
        
