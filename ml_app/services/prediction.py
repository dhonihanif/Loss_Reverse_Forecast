import os 
import pandas as pd 
import numpy as np 
from rest_framework import status 
import pickle, joblib
import json

base_path = os.getcwd() 
pickle_path=os.path.normpath(base_path+os.sep+'models')
log_path=os.path.normpath(base_path+os.sep+'log')

class Prediction:
    def predict(self,request):
        return_dict=dict()
        try:
            input_request=request.body 
            decode_input_request=input_request.decode('utf8').replace("'",'"')
            request_dict=json.loads(decode_input_request)
            df_pred=pd.json_normalize(request_dict)

            model=joblib.load(r"C:\Users\DHONI HANIF\OneDrive\Documents\AI_Collection_and_Loss_Reverse_Forecast_\models\nasabah\loss_reverse.joblib")
            prediction=model.predict(df_pred)
            print(prediction)

            request_dict['prediction']=prediction 
            return_dict['response']=request_dict
            return_dict['status']=status.HTTP_200_OK
            return return_dict

        except Exception as e: 
            return_dict['response']="Exception when prediction: "+str(e)
            return_dict['status']=status.HTTP_500_INTERNAL_SERVER_ERROR
            return return_dict 