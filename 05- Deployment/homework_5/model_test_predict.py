import pickle

datapoint = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}


with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


print(pipeline.predict_proba(datapoint)[0, 1])