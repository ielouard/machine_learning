from sklearn import tree

clf = tree.DecisionTreeClassifier()


# [height, weight]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']
     """

X = [[168, 50], [152, 50], [184, 60], [191, 70], [176, 80], [150, 45],[188, 90], [156, 75],[198,110],[198, 120], [152, 80],[170,110]]
Y= ['Not obese', 'Not obese', 'Not obese', 'Not obese', 'Not obese', 'Not obese', 'over weight','over weight','over weight','obese', 'obese', 'obese']
"""
# train  data
clf = clf.fit(X, Y)

prediction = clf.predict([[174, 180]])

tree.export_graphviz(clf, out_file='arbre.dot') 
# CHALLENGE compare their reusults and print the best one!

print(prediction)