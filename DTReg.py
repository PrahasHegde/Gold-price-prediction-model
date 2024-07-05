#USING DECISION TREE REGRESSOR

#imports
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv('gld_price_data.csv')

print(df.head())
print(df.shape)
print(df.info())

#dropping Date column
df = df.drop(columns=['Date'])

#correlation Heatmap of features and label
fig = px.imshow(df.corr(), text_auto=True)
fig.show()


#Train-Test Split
X = df.drop(columns=['GLD'])
y = df['GLD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=345)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape )


#Building and Evaluating the Model

dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
dtr_prediction = dtr.predict(X_test)


#matrix
print(mean_absolute_error(y_test, dtr_prediction)) #1.5539302336244543
print(mean_squared_error(y_test, dtr_prediction)) #12.746302031895944
print(r2_score(y_test, dtr_prediction)) #0.9756342936129743

#plot actual vs predicted
actual = y_test
predicted = dtr_prediction

plt.figure(figsize=(15, 10))

# Plot the actual values as a scatter plot
plt.scatter(range(len(actual)), actual, color='blue', label='Actual')

# Plot the predicted values as a line
plt.scatter(range(len(actual)), predicted, color='red', label='Predicted')

# A line between the actual point and predicted point
for i in range(len(actual)):
    plt.plot([i, i], [actual.iloc[i], predicted[i]], color='green', linestyle='--')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values (Gold price prediction)')
plt.legend()
plt.show()


