import kserve
import joblib
from http import HTTPStatus
from typing import Dict
from fastapi.exceptions import HTTPException, RequestValidationError
from process import TimeSeriesPostProcessor


class Model(kserve.Model):

    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model = None
        self.load()
        self.processor = TimeSeriesPostProcessor()

    def load(self):
        self.model = joblib.load("model_storage/model-new.sav")
        self.ready = True

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:

        error_msg = ""
        success_flag = False

        try:
            assert "range_start" in payload and "range_end" in payload, \
                "Provide inputs 'range_start: date' and 'range_end: date'."

            forecast = self.processor.get_forecast(self.model, payload["range_start"], payload["range_end"])
            success_flag = True

        except AssertionError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))

        except RequestValidationError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Error in predict(): " + str(e))

        return {"success": success_flag, "forecast": forecast, "error": error_msg}
