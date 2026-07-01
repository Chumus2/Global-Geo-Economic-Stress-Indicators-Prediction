import requests
from django.shortcuts import render
from django.views import View


API_URL = "http://api:8001/predict"

class MainPageView(View):

    def get(self, request):
        return render(request, "main/main.html")

    def post(self, request):
        context = {}
        context["form_data"] = request.POST.dict()

        try:
            payload = {
                "year": int(request.POST.get("year")),
                "gdp_growth": float(request.POST.get("gdp_growth")),
                "inflation": float(request.POST.get("inflation")),
                "unemployment": float(request.POST.get("unemployment")),
                "gdp_per_capita": float(request.POST.get("gdp_per_capita")),
                "population": float(request.POST.get("population")),
                "food_production_index": float(request.POST.get("food_production_index")),
                "cereal_yield": float(request.POST.get("cereal_yield")),
                "cereal_production_tonnes": float(request.POST.get("cereal_production_tonnes")),
                "agricultural_land_pct": float(request.POST.get("agricultural_land_pct")),
                "dietary_energy_supply_adequacy": float(request.POST.get("dietary_energy_supply_adequacy")),
                "data_completeness_score": float(request.POST.get("data_completeness_score")),
                "latitude": float(request.POST.get("latitude")),
                "longitude": float(request.POST.get("longitude")),
                "food_pressure_score": float(request.POST.get("food_pressure_score")),
                "income_group": request.POST.get("income_group"),
            }
        except (TypeError, ValueError):
            context["error"] = "Please fill in all fields with valid numeric values."
            return render(request, "main/main.html", context)

        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            score = data.get("predicted_economic_stress_score")
            context["prediction"] = score

            if score < 33:
                context["prediction_label"] = "Low economic stress"
            elif score < 66:
                context["prediction_label"] = "Moderate economic stress"
            else:
                context["prediction_label"] = "High economic stress"

        except requests.exceptions.RequestException as e:
            context["error"] = f"Could not reach the prediction service: {e}"

        return render(request, "main/main.html", context)