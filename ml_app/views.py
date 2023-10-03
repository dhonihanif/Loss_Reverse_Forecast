from django.shortcuts import render
import pandas as pd
from rest_framework.response import Response 
from rest_framework.views import APIView
from ml_app.services.prediction import Prediction
import joblib

def predict(request):
    if request.method == 'POST':
        # Dapatkan input dari pengguna (misalnya: form input)
        input_data = [float(request.POST['net_income']), float(request.POST['loan_amount']),
                      float(request.POST["avg_amounts_previous_bills"]),
                      float(request.POST["avg_previous_payment"]),
                      float(request.POST["amount_of_late"]),
                      float(request.POST["late_payment_amount"]),
                      float(request.POST["credit_score"]),
                      float(request.POST["arrears_amounts"]),
                      float(request.POST["aging"]), float(request.POST["employment_type"]),
                      float(request.POST["loss_reverse"])
                      ]

        # Muat model ML
        model = joblib.load(r'C:\Users\DHONI HANIF\OneDrive\Documents\AI_Collection_and_Loss_Reverse_Forecast_\models\nasabah\loss_reverse.joblib')

        # Lakukan prediksi
        data = {
            "net_income": input_data[0],
            "loan_amount": input_data[1],                 
            "avg_amounts_previous_bills": input_data[2],  
            "avg_previous_payment": input_data[3],        
            "amount_of_late": input_data[4],              
            "late_payment_amount": input_data[5],         
            "credit_score": input_data[6],                
            "arrears_amounts": input_data[7],             
            "aging": input_data[8],                        
            "employment_type": input_data[9],              
            "loss_reverse": input_data[10],                
                    }
        data = pd.DataFrame({i: [j] for i, j in zip(data.index, data.values)})
        prediction = model.predict(data)

        # Tampilkan hasil prediksi
        return render(request, 'result.html', {'prediction': prediction[0]})

    return render(request, 'predict.html')

class PredChurnModel(APIView): 
    def post(self,request):
        pred_obj=Prediction()
        response_dict=pred_obj.predict(request)
        response=response_dict['response']
        status_value=response_dict['status']
        return Response(response,status_value)