import os
import uvicorn
from pathlib import Path
import joblib
from typing import Dict
from http import HTTPStatus
from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException, RequestValidationError
from process import TimeSeriesPostProcessor
from train import TimeSeriesTrainer
PROJECT_ROOT_DIR = str(Path(__file__).parent.parent)


class Model:
    def __init__(self, name: str):
        self.name = name
        self.ready = False
        self.model = None
        self.load()
        self.processor = TimeSeriesPostProcessor()

    def load(self):
        try:
            self.model = joblib.load(os.path.join(PROJECT_ROOT_DIR, "model_storage/model-new.sav"))
        except EOFError as eof:
            print(str(eof))
            trainer = TimeSeriesTrainer()
            trainer()
            self.model = joblib.load(os.path.join(PROJECT_ROOT_DIR, "model_storage/model-new.sav"))

        self.ready = True

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:

        error_msg = ""
        success_flag = False

        try:
            assert "range_start" in payload and "range_end" in payload, \
                "Provide inputs 'range_start: date' and 'range_end: date'."

            forecast = self.processor.get_forecast(self.model, payload["range_start"], payload["range_end"])
            print(f"FORECAST: \n{forecast}")
            success_flag = True

        except AssertionError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(e))

        except RequestValidationError as e:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Error in predict(): " + str(e))

        return {"success": success_flag, "forecast": forecast, "error": error_msg}


server = FastAPI(debug=True)
model = Model(name="varma-nn")
# A sample GET endpoint with a path parameter

@server.post("/forecast")
def read_item(request: dict):
    return model.predict(request)


if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=int(os.getenv('PORT', "8080")))
