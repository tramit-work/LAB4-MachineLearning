from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from .decisionTree import DecisionTreeClass, accuracy  
from .randomForest import RandomForest 

def result_lab(request):
    try:

        data = pd.read_csv("data/drug200.csv") 
    except FileNotFoundError:
        return JsonResponse({'error': 'Data file not found.'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    X = data.drop(columns=['Drug'])
    y = data['Drug']


    X['Sex'] = X['Sex'].replace({'M': 0, 'F': 1})
    X['BP'] = X['BP'].replace({'HIGH': 2, 'NORMAL': 1, 'LOW': 0})
    X['Cholesterol'] = X['Cholesterol'].replace({'HIGH': 1, 'NORMAL': 0})
    y = y.replace({'drugA': 0, 'drugB': 1, 'drugC': 2, 'drugX': 3, 'DrugY': 4})


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    decision_tree = DecisionTreeClass(min_samples_split=2, max_depth=10)
    decision_tree.fit(X_train, y_train)
    y_pred_tree = decision_tree.predict(X_test)


    random_forest = RandomForest(n_trees=3, n_features=4)
    random_forest.fit(X_train, y_train)
    y_pred_forest = random_forest.predict(X_test)


    accuracy_tree = accuracy(y_test.values, y_pred_tree)
    accuracy_forest = accuracy(y_test.values, y_pred_forest)

    context = {
        'y_test': y_test.values.tolist(),
        'y_pred_tree': y_pred_tree.tolist(),
        'y_pred_forest': y_pred_forest.tolist(),
        'accuracy_tree': accuracy_tree,
        'accuracy_forest': accuracy_forest,
    }
    return render(request, 'index.html', context)
