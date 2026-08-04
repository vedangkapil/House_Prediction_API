import io
import joblib
import pandas as pd 
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load('house_model.joblib')
features = joblib.load('house_features.joblib')

# input schema
class HouseFeatures(BaseModel):
    MedInc : float = Field(gt=0, description='Median income of Neighourhood')
    HouseAge : float = Field(gt=0, description='Average age of the houses in the block')
    AveRooms : float = Field(gt=0, description='Average number of rooms per house')
    AveBedrms : float = Field(gt=0, description='Average number of bedrooms per house')
    Population : float = Field(gt=0, description='Total population of the block')
    AveOccup : float = Field(gt=0, description='Average number of people per house')
    Latitude : float = Field(ge=32, le=42, description='Latitude')
    Longitude : float = Field(ge=-125, le=-114, description='Longitude')


@app.get('/')
def home():
    return{
        'meassage': 'California house prediction API',
        'status': 'Running',
        'endpoint': 'Send POST request to /predict'
    }

@app.get('/health')
def health():
    return{
        'status': 'Running',
        'Model': 'RandomForestRegressor',
        'Features': features,
        'avg_error': '$32,784' 
    }

# Prediction
@app.post('/predict')
def predict(House: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            'MedInc': House.MedInc,
            'HouseAge': House.HouseAge,
            'AveRooms': House.AveRooms,
            'AveBedrms': House.AveBedrms,
            'Population': House.Population,
            'AveOccup': House.AveOccup,
            'Latitude': House.Latitude,
            'Longitude': House.Longitude
        }])

        predicted = model.predict(input_data)[0]
        price_usd = predicted *100000

        return{
            'predicted_price': f'${price_usd:,.0f}',
            'predicted_price_short': f'${predicted:.2f} hundred thousands',
            'fidence_range': f'${price_usd - 32784:,.0f} to ${price_usd + 32784:,.0f}'
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail= f'prediction failed: {str(e)}'
        )

@app.post('/predict-file')
async def predict_file(file: UploadFile=File(...)):

    if not file.filename.endswith('csv'):
        raise HTTPException(
            status_code=400,
            detail= 'please upload  CSV file only'
        )

    contents = await file.read()
    df = pd.DataFrame(io.BytesIO(contents))

