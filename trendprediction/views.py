import json
import glob
import random
import pandas as pd
import numpy as np
from joblib import load
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier

clf_colour = load('model2\clf_colour.hd')
colour_encoder = load('model2\colour_encoder.hd')
label_encoders = {
    'MONTH': load('model2\MONTH_label_encoder.hd'),
    'Category': load('model2\Category_label_encoder.hd'),
    'GENDER': load('model2\GENDER_label_encoder.hd'),
    'Age': load('model2\Age_label_encoder.hd'),
}
clf_pattern = load('model2\clf_pattern.hd')
pattern_encoder = load('model2\pattern_encoder.hd')


class DataStore():
    Prod = None
    Prod2 = None
    Prod3 = None


data = DataStore()


def home(request):
    return render(request, 'website.html')


@csrf_exempt
def predict_colour(request):
    if request.method == 'POST':
        # Extract data from the request
        try:
            data = json.loads(request.body)
            year = data.get('year')
            month = data.get('month')
            category = data.get('category')
            gender = data.get('gender')
            age = data.get('age')
        except json.JSONDecodeError:
            year = request.POST.get('year')
            month = request.POST.get('month')
            category = request.POST.get('category')
            gender = request.POST.get('gender')
            age = request.POST.get('age')

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
        # For GET requests, you might want to return instructions or a form for input
        return JsonResponse({'error': 'Please use a POST request with the correct parameters to get a prediction.'})


@csrf_exempt
def predict_pattern(request):
    if request.method == 'POST':
        # Extract data from the request
        # data = request.json
        data = json.loads(request.body)
        year = data.get('year')
        month = data.get('month')
        category = data.get('category')
        gender = data.get('gender')
        age = data.get('age')

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
        df = pd.read_csv('csv/model_style.csv')

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

        # year = request.form.get('year')
        # month = request.form.get('month')
        # gender = request.form.get('gender')
        # category = request.form.get('category')

        data = json.loads(request.body)
        month = data.get('month')
        category = data.get('category')
        gender = data.get('gender')

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
    if 'tshirts' in request.args:
        #        Insert code here/ information to retrieve
        top1_name = []
        top1_description = []
        top1_score = []
        bottom1_name = []
        bottom1_description = []
        bottom1_score = []

        images_path = []
        imagestshirts = glob.glob('static/img/current_trends/shirt+' + '*.jpeg')
        for i in range(0, len(imagestshirts)):
            images_path.append(imagestshirts[i])
        imagestshirts = glob.glob('static/img/current_trends/shirt-' + '*.jpeg')
        for i in range(0, len(imagestshirts)):
            images_path.append(imagestshirts[i])

        colnames = ['sno', 'URL', 'id', 'desc', 'stars', 'num_ratings', 'num_reviews', 'reviews', 'vader_score',
                    'final_score']
        reqdcolnames = ['id', 'stars', 'desc', 'URL', 'final_score']
        dataset_csv = pd.read_csv('CurrentTrends/final_csv/tshirts/tshirts_csv_final.csv', names=colnames,
                                  delimiter=',', on_bad_lines='skip',
                                  header=None, usecols=reqdcolnames, na_values=" NaN")
        dataset_csv = dataset_csv.dropna()
        dataset_csv2 = dataset_csv.sort_values(by='final_score', ascending=False)
        dataset_csv2 = dataset_csv2.reset_index()

        # print(dataset_csv2.head())
        for i in range(1, 6):
            top1_name.append(dataset_csv2['desc'][i])
            top1_description.append(dataset_csv2['URL'][i])
            top1_score.append(dataset_csv2['final_score'][i])
        for i in range((len(dataset_csv2) - 5), len(dataset_csv2)):
            bottom1_name.append(dataset_csv2['desc'][i])
            bottom1_description.append(dataset_csv2['URL'][i])
            bottom1_score.append(dataset_csv2['final_score'][i])

        df = pd.read_csv('CurrentTrends/Leaderboard/tshirt_colour_top_bottom.csv')
        # print(df.head())
        df = df[["Bigram", "Rating", "Count"]]

        # order in the groupby here matters, it determines the json nesting the groupby call makes a pandas series by
        # grouping 'the_parent' and 'the_child', while summing the numerical column 'child_size'
        df1 = df.groupby(['Bigram', 'Rating'])['Count'].sum()
        df1 = df1.reset_index()

        # start a new flare.json document
        flare = dict()
        d = {"name": "flare", "children": []}
        i = 0
        for line in df1.values:
            Bigram = line[0]
            Rating = line[1]
            Count = line[2]
            # make a list of keys
            keys_list = []
            for item in d['children']:
                keys_list.append(item['name'])

            # if 'the_parent' is NOT a key in the flare.json yet, append it
            if not Bigram in keys_list:
                d['children'].append({"name": Bigram, "children": [{"name": Rating, "size": Count}]})

            # if 'the_parent' IS a key in the flare.json, add a new child to it
            else:
                d['children'][keys_list.index(Bigram)]['children'].append({"name": Rating, "size": Count})

        flare = d
        e = json.dumps(flare)
        data.Prod = json.loads(e)
        Prod = data.Prod

        df = pd.read_csv('CurrentTrends/Leaderboard/tshirt_neck_top_bottom.csv')
        # print(df.head())
        df = df[["Bigram", "Rating", "Count"]]

        # order in the groupby here matters, it determines the json nesting the groupby call makes a pandas series by
        # grouping 'the_parent' and 'the_child', while summing the numerical column 'child_size'
        df1 = df.groupby(['Bigram', 'Rating'])['Count'].sum()
        df1 = df1.reset_index()

        # start a new flare.json document
        flare = dict()
        d = {"name": "flare", "children": []}

        for line in df1.values:
            Bigram = line[0]
            Rating = line[1]
            Count = line[2]

            # make a list of keys
            keys_list = []
            for item in d['children']:
                keys_list.append(item['name'])

            # if 'the_parent' is NOT a key in the flare.json yet, append it
            if not Bigram in keys_list:
                d['children'].append({"name": Bigram, "children": [{"name": Rating, "size": Count}]})

            # if 'the_parent' IS a key in the flare.json, add a new child to it
            else:
                d['children'][keys_list.index(Bigram)]['children'].append({"name": Rating, "size": Count})

        flare = d
        e = json.dumps(flare)
        data.Prod2 = json.loads(e)
        Prod2 = data.Prod2

        df = pd.read_csv('CurrentTrends/Leaderboard/tshirt_print_top_bottom.csv')
        # print(df.head())
        df = df[["Bigram", "Rating", "Count"]]

        # order in the groupby here matters, it determines the json nesting the groupby call makes a pandas series by
        # grouping 'the_parent' and 'the_child', while summing the numerical column 'child_size'
        df1 = df.groupby(['Bigram', 'Rating'])['Count'].sum()
        df1 = df1.reset_index()

        # start a new flare.json document
        flare = dict()
        d = {"name": "flare", "children": []}

        for line in df1.values:
            Bigram = line[0]
            Rating = line[1]
            Count = line[2]

            # make a list of keys
            keys_list = []
            for item in d['children']:
                keys_list.append(item['name'])

            # if 'the_parent' is NOT a key in the flare.json yet, append it
            if not Bigram in keys_list:
                d['children'].append({"name": Bigram, "children": [{"name": Rating, "size": Count}]})

            # if 'the_parent' IS a key in the flare.json, add a new child to it
            else:
                d['children'][keys_list.index(Bigram)]['children'].append({"name": Rating, "size": Count})

        flare = d
        e = json.dumps(flare)
        data.Prod3 = json.loads(e)
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

        return render("website_currenttrends_tshirts.html", context)
