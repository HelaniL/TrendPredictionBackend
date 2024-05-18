import os
import json
import glob
import pandas as pd
from joblib import load
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.multioutput import MultiOutputClassifier

# Load models and encoders
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

clf_colour = load(os.path.join(BASE_DIR, 'trendprediction', 'model2', 'clf_colour.hd'))
colour_encoder = load(os.path.join(BASE_DIR, 'trendprediction', 'model2', 'colour_encoder.hd'))
label_encoders = {
    'MONTH': load(os.path.join(BASE_DIR, 'trendprediction', 'model2', 'MONTH_label_encoder.hd')),
    'Category': load(os.path.join(BASE_DIR, 'trendprediction', 'model2', 'Category_label_encoder.hd')),
    'GENDER': load(os.path.join(BASE_DIR, 'trendprediction', 'model2', 'GENDER_label_encoder.hd')),
    'Age': load(os.path.join(BASE_DIR, 'trendprediction', 'model2', 'Age_label_encoder.hd')),
}
clf_pattern = load(os.path.join(BASE_DIR, 'trendprediction', 'model2', 'clf_pattern.hd'))
pattern_encoder = load(os.path.join(BASE_DIR, 'trendprediction', 'model2', 'pattern_encoder.hd'))


class DataStore():
    Prod = None
    Prod2 = None
    Prod3 = None


data = DataStore()


def home(request):
    return JsonResponse({'message': 'Welcome to the trend prediction API!'})


@csrf_exempt
def predict_colour(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            year = data.get('year')
            month = data.get('month')
            category = data.get('category')
            gender = data.get('gender')
            age = data.get('age')
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

        if None in [year, month, category, gender, age]:
            return JsonResponse({'error': 'Missing parameters'}, status=400)

        try:
            year = int(year)
        except ValueError:
            return JsonResponse({'error': 'Invalid year'}, status=400)

        # Encode the inputs using the label encoders
        month_encoded = label_encoders['MONTH'].transform([month])
        category_encoded = label_encoders['Category'].transform([category])
        gender_encoded = label_encoders['GENDER'].transform([gender])
        age_encoded = label_encoders['Age'].transform([age])

        # Create the feature array
        features = [[year, month_encoded[0], category_encoded[0], gender_encoded[0], age_encoded[0]]]

        # Predict the colour
        colour_pred = clf_colour.predict(features)

        # Decode the predictions to get the actual label
        colour = colour_encoder.inverse_transform(colour_pred)

        # Return the prediction
        return JsonResponse({'colour': colour[0]})
    else:
        return JsonResponse({'error': 'Please use a POST request with the correct parameters to get a prediction.'})


@csrf_exempt
def predict_pattern(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            year = data.get('year')
            month = data.get('month')
            category = data.get('category')
            gender = data.get('gender')
            age = data.get('age')
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

        if None in [year, month, category, gender, age]:
            return JsonResponse({'error': 'Missing parameters'}, status=400)

        try:
            year = int(year)
        except ValueError:
            return JsonResponse({'error': 'Invalid year'}, status=400)

        # Encode the inputs using the label encoders
        month_encoded = label_encoders['MONTH'].transform([month])
        category_encoded = label_encoders['Category'].transform([category])
        gender_encoded = label_encoders['GENDER'].transform([gender])
        age_encoded = label_encoders['Age'].transform([age])

        # Create the feature array
        features = [[year, month_encoded[0], category_encoded[0], gender_encoded[0], age_encoded[0]]]

        # Predict the pattern
        pattern_pred = clf_pattern.predict(features)

        # Decode the predictions to get the actual label
        pattern = pattern_encoder.inverse_transform(pattern_pred)

        # Return the prediction
        return JsonResponse({'pattern': pattern[0]})
    else:
        return JsonResponse({'error': 'Please use a POST request with the correct parameters to get a prediction.'})


@csrf_exempt
def predict_style(request):
    if request.method == 'POST':
        csv_path = os.path.join(BASE_DIR, 'trendprediction', 'csv', 'model_style.csv')
        df = pd.read_csv(csv_path)

        input_features = ['gender', 'season', 'subCategory']
        X = df[input_features]
        y = df[['sleeve_type', 'neck_type']]

        preprocessor = ColumnTransformer(
            transformers=[('cat', OneHotEncoder(), input_features)]
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
        ])

        model.fit(X_train, y_train)

        try:
            data = json.loads(request.body)
            month = data.get('month')
            category = data.get('category')
            gender = data.get('gender')
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

        input_data = pd.DataFrame({
            'gender': [gender],
            'season': [month],
            'subCategory': [category]
        })

        prediction = model.predict(input_data)
        sleeve_pred, neck_pred = prediction[0]
        return JsonResponse({'sleeve': sleeve_pred, 'neck': neck_pred})
    else:
        return JsonResponse({'error': 'Please use a POST request with the correct parameters to get a prediction.'})


def currenttrends(request):
    if 'tshirts' in request.GET:
        top1_name = []
        top1_description = []
        top1_score = []
        bottom1_name = []
        bottom1_description = []
        bottom1_score = []

        images_path = []
        images_folder = os.path.join(BASE_DIR, 'trendprediction', 'static', 'img', 'current_trends')
        imagestshirts = glob.glob(os.path.join(images_folder, 'shirt+' + '*.jpeg'))
        for img_path in imagestshirts:
            images_path.append(img_path)
        imagestshirts = glob.glob(os.path.join(images_folder, 'shirt-' + '*.jpeg'))
        for img_path in imagestshirts:
            images_path.append(img_path)

        colnames = ['sno', 'URL', 'id', 'desc', 'stars', 'num_ratings', 'num_reviews', 'reviews', 'vader_score', 'final_score']
        reqdcolnames = ['id', 'stars', 'desc', 'URL', 'final_score']
        csv_path = os.path.join(BASE_DIR, 'trendprediction', 'CurrentTrends', 'final_csv', 'tshirts', 'tshirts_csv_final.csv')
        dataset_csv = pd.read_csv(csv_path, names=colnames, delimiter=',', on_bad_lines='skip', header=None, usecols=reqdcolnames, na_values=" NaN")
        dataset_csv = dataset_csv.dropna()
        dataset_csv2 = dataset_csv.sort_values(by='final_score', ascending=False).reset_index()

        for i in range(1, 6):
            top1_name.append(dataset_csv2['desc'][i])
            top1_description.append(dataset_csv2['URL'][i])
            top1_score.append(dataset_csv2['final_score'][i])
        for i in range(len(dataset_csv2) - 5, len(dataset_csv2)):
            bottom1_name.append(dataset_csv2['desc'][i])
            bottom1_description.append(dataset_csv2['URL'][i])
            bottom1_score.append(dataset_csv2['final_score'][i])

        df = pd.read_csv(os.path.join(BASE_DIR, 'trendprediction', 'CurrentTrends', 'Leaderboard', 'tshirt_colour_top_bottom.csv'))
        df = df[["Bigram", "Rating", "Count"]]
        df1 = df.groupby(['Bigram', 'Rating'])['Count'].sum().reset_index()
        flare = {'name': 'flare', 'children': []}
        d = flare
        for line in df1.values:
            Bigram, Rating, Count = line
            keys_list = [item['name'] for item in d['children']]
            if not Bigram in keys_list:
                d['children'].append({'name': Bigram, 'children': [{'name': Rating, 'size': Count}]})
            else:
                d['children'][keys_list.index(Bigram)]['children'].append({'name': Rating, 'size': Count})
        data.Prod = json.loads(json.dumps(d))
        Prod = data.Prod

        df = pd.read_csv(os.path.join(BASE_DIR, 'trendprediction', 'CurrentTrends', 'Leaderboard', 'tshirt_neck_top_bottom.csv'))
        df = df[["Bigram", "Rating", "Count"]]
        df1 = df.groupby(['Bigram', 'Rating'])['Count'].sum().reset_index()
        flare = {'name': 'flare', 'children': []}
        d = flare
        for line in df1.values:
            Bigram, Rating, Count = line
            keys_list = [item['name'] for item in d['children']]
            if not Bigram in keys_list:
                d['children'].append({'name': Bigram, 'children': [{'name': Rating, 'size': Count}]})
            else:
                d['children'][keys_list.index(Bigram)]['children'].append({'name': Rating, 'size': Count})
        data.Prod2 = json.loads(json.dumps(d))
        Prod2 = data.Prod2

        df = pd.read_csv(os.path.join(BASE_DIR, 'trendprediction', 'CurrentTrends', 'Leaderboard', 'tshirt_print_top_bottom.csv'))
        df = df[["Bigram", "Rating", "Count"]]
        df1 = df.groupby(['Bigram', 'Rating'])['Count'].sum().reset_index()
        flare = {'name': 'flare', 'children': []}
        d = flare
        for line in df1.values:
            Bigram, Rating, Count = line
            keys_list = [item['name'] for item in d['children']]
            if not Bigram in keys_list:
                d['children'].append({'name': Bigram, 'children': [{'name': Rating, 'size': Count}]})
            else:
                d['children'][keys_list.index(Bigram)]['children'].append({'name': Rating, 'size': Count})
        data.Prod3 = json.loads(json.dumps(d))
        Prod3 = data.Prod3

        context = {
            'images_path': images_path,
            'Prod': Prod,
            'Prod2': Prod2,
            'Prod3': Prod3,
            'top1_name': top1_name,
            'top1_description': top1_description,
            'top1_score': top1_score,
            'bottom1_name': bottom1_name,
            'bottom1_description': bottom1_description,
            'bottom1_score': bottom1_score
        }

        return JsonResponse(context)
