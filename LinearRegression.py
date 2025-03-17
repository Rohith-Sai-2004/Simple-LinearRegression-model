import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('data_for_lr.csv')
data = data.dropna(inplace=False)


X= np.array(data['x']).reshape(-1,1)
y= np.array(data['y']).reshape(-1,1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

st.header("Linear Regression Model")
with st.expander("About the Model"):
    st.write(" ### This is Simple Linear Regression Type Model.")
    st.write("This Model takes just one input(Independent Variable) and gives output(Dependent Variable).")
    st.write("### Y=mX+c")

st.sidebar.header("User Marks")

with st.sidebar.expander("ℹ️ About Simple Linear Regression"):
    st.write("### Definition")
    st.write("Simple Linear Regression models the relationship between two variables **X** (input) and **Y** (output).")
    st.latex(r"Y = mX + c")
    
    st.write("### Key Steps")
    st.write("1. Collect data for **X** and **Y**.")
    st.write("2. Fit a regression line minimizing error.")
    st.write("3. Use **Mean Squared Error (MSE)**:")
    st.latex(r"MSE = \frac{1}{n} \sum (Y_{\text{actual}} - Y_{\text{predicted}})^2")
    st.write("4. Optimize parameters using **Gradient Descent**.")
    st.write("5. Predict new values.")
user = st.sidebar.number_input("Enter Total Marks :",0.0,100.0,step=0.1)

grade_pred = model.predict([[user]])[0][0]

status = "PASS" if grade_pred > 40 else "FAIL"

st.write(f"### Predicted : {grade_pred:.2f} ")
st.write(f"### status :{status}")

