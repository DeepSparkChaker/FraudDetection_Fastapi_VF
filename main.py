# First, we will need to import the library and initialize the main application object:
import os
import joblib
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from pydantic import BaseModel
# import nest_asyncio
from utils.pipeline import *

# from typing import Any, Dict,List,Enum
# import numpy as np  

"""
1. Set up the FastAPI application
2. Load the model(s) into the application
3. Create required API endpoint(s) for users to submit data:
   - These could be CSV file(s), image(s), JSON object(s), etc.
   - Handle incoming data appropriately
4. Use the indended model to predict the result(s) on the data submitted
5. If successful, return the predictions, else raise an error  

 """

## API INSTANTIATION
## ----------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description="A simple API that use Ml model to predict fraud ",
    version="0.1",
)


# Creating the data model for data validation
class ClientData(BaseModel):
    step: List[int]
    type: List[str]
    amount: List[float]
    nameOrig: List[str]
    oldbalanceOrig: List[float]
    newbalanceOrig: List[float]
    nameDest: List[str]
    oldbalanceDest: List[float]
    newbalanceDest: List[float]


# Load  the model  a serialized .joblib file
#joblib_filename = "models/pipeline_model_lgbm_final.joblib"
#model = joblib.load(joblib_filename)
with open('models/pipeline_model_lgbm_final.joblib', 'rb') as joblib_filename:
    model = joblib.load(joblib_filename)
   

## API ENDPOINTS
## ----------------------------------------------------------------

##################
@app.get('/')
def index():
    """
  This is a first docstring.
  """
    return {'message': 'This is a Fraud  Classification API!'}


# Tester
@app.get('/ping')
def ping():
    '''
  This is a first docstring.
  '''
    return ('pong', 200)


# Defining the prediction endpoint without data validation
@app.post('/basic_predict_raw')
async def basic_predict(request: Request):
    '''
  This route don't validate data .
  '''
    # Getting the JSON from the body of the request
    input_data = await request.json()

    # Converting JSON to Pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Getting the prediction
    pred = model.predict(input_df)[0]

    return pred


# We now define the function that will be executed for each URL request and return the value:
@app.post("/predict-fraud")
async def predict_fraud(item: ClientData):
    """
  A simple function that receive a client data and predict Fraud.
  :param client_data:
  :return: prediction, probabilities
  
  """
    # perform prediction
    # df =pd.DataFrame([item])
    h = item.dict()
    df = pd.DataFrame.from_dict(h, orient="columns")
    prediction = model.predict(df)
    prediction_final = ["Fraud" if (x > 0.5) else "Not Fraud" for x in prediction]
    return prediction_final
	
# Create the POST endpoint with path '/predict'
@app.post("/predict_csv")
async def create_upload_file(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):
        # Create a temporary file with the same name as the uploaded 
        # CSV file to load the data into a pandas Dataframe
        with open(file.filename, "wb")as f:
            f.write(file.file.read())
        data = pd.read_csv(file.filename)
        os.remove(file.filename)
        # Return a JSON object containing the model predictions
        return {
            "predections": model.predict(data)
        }    
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request 
        # (you can learn more about HTTP response status codes here)
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
# uvicorn app:app --reload
