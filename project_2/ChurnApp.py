import dash
import dash_core_components as dcc
import dash_html_components as html
from sklearn.model_selection import train_test_split
from dash.dependencies import Input, Output, State
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

pd.set_option('display.max_columns', None)

# Trained machine learning models
model = joblib.load('./artifacts/rf_model.pk2')
model_lr = joblib.load('./artifacts/model_lr_V2.pk1')
data = pd.read_csv('./data/prepped_df.csv')

# Create a Dash app
app = dash.Dash(__name__)
server=app.server

def ft_importance(model_dt):
    
    test_df = pd.read_csv('./data/prepped_df.csv', index_col=False)

    X = test_df.drop(columns=['Churn'])
    Y = test_df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model_dt.fit(X_train,y_train)
    
    # Extract features and their coefficients
    coef = model_dt.named_steps["logisticregression"].coef_[0]
    ft = X_train.columns
    
    # Convert to Pandas Series
    ft_importance = pd.Series(
        np.exp(coef), index=ft
        ).sort_values(ascending=False)
    
    #Plot ft importance
    # Create horizontal bar chart of feature importances
    fig = px.bar(
        data_frame=ft_importance, 
        x=ft_importance[:16].values, 
        y=ft_importance[:16].index, 
        title="Feature Importance"
    )

    fig.update_layout(xaxis_title='Gini Importance', yaxis_title='')
    return fig

# Define the layout
app.layout = html.Div([
    dcc.Graph(figure=ft_importance(model_lr), style={'width': '80%'}),
    html.H1("Customer Churn Prediction"),

    # Main-container
    html.Div([

        # Categorical Data
        html.Div([

            # Gender {gender_Male	gender_Female}
            html.H4("Gender: *"),
            dcc.Dropdown(
                id='gender-dropdown',
                options=[
                    {'label': 'Male', 'value': 0},
                    {'label': 'Female', 'value': 1}
                ],
                value='',
                placeholder="Select Gender"
            ),

            # SeniorCitizen {Done}
            html.H4("Senior Citizen: *"),
            dcc.Dropdown(
                id='citizen-dropdown',
                options=[
                    {'label': 'I am older than 65', 'value': 1},
                    {'label': 'I am younger than 65', 'value': 0}
                ],
                value='',
                placeholder="Indication of Senior Citizen"
            ),

            # PhoneService {PhoneService_Yes	PhoneService_No}
            html.H4("Phone Service: *"),
            dcc.Dropdown(
                id='phone-dropdown',
                options=[
                    {'label': 'I have phone service', 'value': 0},
                    {'label': 'I do not have phone service', 'value': 1}
                ],
                value='',
                placeholder="Indication of phone service"
            ),

            html.H4("Multiple Phone Lines: *"),
            # MultipleLines {MultipleLines_No	MultipleLines_No phone service	MultipleLines_Yes}
            dcc.Dropdown(
                id='multiLines-dropdown',
                options=[
                    {'label': 'I have multiple lines', 'value': 2},
                    {'label': 'I do not have multiple lines', 'value': 0}
                ],
                value='',
                placeholder="Indication of multiple lines"
            ),

            html.H4("Internet Service: *"),
            # InternetService {InternetService_DSL	InternetService_Fiber optic	InternetService_No}
            dcc.Dropdown(
                id='internet-dropdown',
                options=[
                    {'label': 'I have DSL', 'value': 0},
                    {'label': 'I have Fiber optic', 'value': 1},
                    {'label': 'I do not have internet service', 'value': 2}
                ],
                value='',
                placeholder="Indication of internet service"
            ),

        ], style={'width':'25%'}),

        # Categorical Data
        html.Div([
            
            html.H4("Online Security: *"),
            # OnlineSecurity {DeviceProtection_Yes	DeviceProtection_No	DeviceProtection_No internet service}
            dcc.Dropdown(
                id='security-dropdown',
                options=[
                    {'label': 'I have online security', 'value': 0},
                    {'label': 'I do not have online security', 'value': 1}
                ],
                value='',
                placeholder="Indication of online security"
            ),

            html.H4("Tech Support: *"),
            # TechSupport {TechSupport_No	TechSupport_Yes	TechSupport_No internet service}
            dcc.Dropdown(
                id='support-dropdown',
                options=[
                    {'label': 'I make use of tech support', 'value': 1},
                    {'label': 'I do not make use of tech support', 'value': 0}
                ],
                value='',
                placeholder="Indication of tech support"
            ),

            html.H4("Contract: *"),
            # Contract {Contract_One year	Contract_Month-to-month	Contract_Two year}
            dcc.Dropdown(
                id='contract-dropdown',
                options=[
                    {'label': 'I make a month-to-month contract', 'value': 1},
                    {'label': 'I make a one year contract', 'value': 0},
                    {'label': 'I make a two year contract', 'value': 2}
                ],
                value='',
                placeholder="Indication of contract"
            ),

            html.H4("Paperless Billing: *"),
            # PaperlessBilling {PaperlessBilling_No	PaperlessBilling_Yes}
            dcc.Dropdown(
                id='billing-dropdown',
                options=[
                    {'label': 'I have paperless bills', 'value': 1},
                    {'label': 'I have on paper bills', 'value': 0}
                ],
                value='',
                placeholder="Indication of paper billing"
            ),

            html.H4("Payment Method: *"),
            # PaymentMethod {PaymentMethod_Mailed check,	PaymentMethod_Bank transfer (automatic),	PaymentMethod_Electronic check, PaymentMethod_Credit card (automatic)}
            dcc.Dropdown(
                id='payment-dropdown',
                options=[
                    {'label': 'I make use of bank transfer (automatic)', 'value': 1},
                    {'label': 'I make use of credit card (automatic)', 'value': 3},
                    {'label': 'I make use of electronic check', 'value': 2},
                    {'label': 'I make use of mailed check', 'value': 0},
                ],
                value='',
                placeholder="Indication of payment method"
            ),

        ], style={'width':'25%'}),

        # Sliders
        html.Div([

            html.H4("Customer Duration: *"),
            # tenure {tenure_cat}
            dcc.Dropdown(
                id='tenure-input',
                options=[
                    {'label': 'Long term customer', 'value': 2},
                    {'label': 'Recurring customer', 'value': 1},
                    {'label': 'New customer', 'value': 0}
                ],
                value='',
                placeholder="Tenure of customer"
            ),

            html.H4("Family: *"),
            # Family {family_0.0 family_1.0}
            dcc.Dropdown(
                id='family-dropdown',
                options=[
                    {'label': 'I have a family', 'value': 1},
                    {'label': 'I am currently living alone', 'value': 0}
                ],
                value='',
                placeholder="Indication of family"
            ),

            html.H4("Streaming: *"),
            # Streaming {Streaming_No	Streaming_both	Streaming_TV	Streaming_M}
            dcc.Dropdown(
                id='stream-dropdown',
                options=[
                    {'label': 'I only stream movies', 'value': 3},
                    {'label': 'I only stream on my TV', 'value': 2},
                    {'label': 'I stream on my TV and stream movies', 'value': 1},
                    {'label': 'I do not stream', 'value': 0},
                ],
                value='',
                placeholder="Indication of streaming"
            ),

            html.H4("Monthly Charges:"),
            # MonthlyCharges {Done}
            dcc.Slider(
                id='monthCharges-input',
                min=0,
                max=1000,
                value=0,
                marks={
                    0: '0',
                    100: '100',
                    200: '200',
                    300: '300',
                    400: '400',
                    500: '500',
                    600: '600',
                    700: '700',
                    800: '800',
                    900: '900',
                    1000: '1000'
                }
            ),

            html.H4("Total Charges:"),
            # TotalCharges {Done}
            dcc.Slider(
                id='totalCharges-input',
                min=0,
                max=10000,
                value=0,
                marks={
                    0: '0',
                    1000: '1000',
                    2000: '2000',
                    3000: '3000',
                    4000: '4000',
                    5000: '5000',
                    6000: '6000',
                    7000: '7000',
                    8000: '8000',
                    9000: '9000',
                    10000: '10000'
                }
            ),

            
        ], style={'width':'40%'}),

        
    ], style={'display':'flex', 'gap': '2vw', 'justify-content': 'center', 'align-content': 'center', 'background-color': 'rgb(82, 212, 255)', 'width': '90%', 'padding': '20px', 'border': '3px black solid'}),

    
    html.Div([

        html.Button('Predict Loan Approval', id='predict-button', style={'width': '25%', 'font-size': '2vw', 'padding': '10px 0px 10px 0px', 'border-radius': '10px'}),
        html.H3("Output: ", style={'line-height': '6.5vh'}),
        html.Div(id='prediction-output')

    ], style={'display':'flex', 'gap': '2.5vw', 'width': '90%', 'justify-content': 'center', 'align-content': 'center', 'background-color': 'rgb(244, 244, 244)', 'padding': '10px 20px 10px 20px', 'border': '3px black solid'})
    
], style={'display':'grid', 'place-items': 'center'})

# Define callback to update prediction output
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('gender-dropdown', 'value'),
    State('citizen-dropdown', 'value'),
    State('phone-dropdown', 'value'),
    State('multiLines-dropdown', 'value'),
    State('internet-dropdown', 'value'),
    State('security-dropdown', 'value'),
    State('support-dropdown', 'value'),
    State('contract-dropdown', 'value'),
    State('billing-dropdown', 'value'),
    State('payment-dropdown', 'value'),
    State('monthCharges-input', 'value'),
    State('totalCharges-input', 'value'),
    State('tenure-input', 'value'),
    State('family-dropdown', 'value'),
    State('stream-dropdown', 'value'),
    
    # Add other input components as additional Input arguments
)

def update_prediction(n_clicks, gender, citizen, phone, multiLines, 
                      internet, security, support, contract, billing,
                      payment, monthCharges, totalCharge, tenure, family,
                      stream):
    
    if (n_clicks == None or gender == None or citizen == None or phone == None or multiLines == None or internet == None or security == None or support == None or contract == None or billing == None or payment == None or tenure == None or family == None or stream == None):
        return html.H1("Please enter all the required information")
    
    if phone == 1:
        multiLines = 1

    if internet == 2:
        security = 2
        support = 2
    
    cat_len_list = [2, 2, 3, 3, 3, 3, 3, 2, 4, 2, 4]
    cat_list = [gender, phone, multiLines, internet, security, support, contract, billing, payment, family, stream]
    val_list = []
        
    for i, col in enumerate(cat_list):
        for x in range(0, cat_len_list[i]):
            if (x == col):
                val_list.append(1)
            else:
                val_list.append(0)
        
    # Create a DataFrame with user inputs
    user_data = pd.DataFrame({

        # Required columns:			

        # Gender {gender_Male	gender_Female}
        'gender_Male': [val_list[0]],
        'gender_Female': [val_list[1]],

        # SeniorCitizen {Done}
        'SeniorCitizen': [citizen],

        # PhoneService {PhoneService_Yes	PhoneService_No}
        'PhoneService_Yes': [val_list[2]],
        'PhoneService_No': [val_list[3]],

        # MultipleLines {MultipleLines_No	MultipleLines_No phone service	MultipleLines_Yes}
        'MultipleLines_No': [val_list[4]],
        'MultipleLines_No phone service': [val_list[5]],
        'MultipleLines_Yes': [val_list[6]],

        # InternetService {InternetService_DSL	InternetService_Fiber optic	InternetService_No}
        'InternetService_DSL': [val_list[7]],
        'InternetService_Fiber optic': [val_list[8]],
        'InternetService_No': [val_list[9]],

        # OnlineSecurity {DeviceProtection_Yes	DeviceProtection_No	DeviceProtection_No internet service}
        'DeviceProtection_Yes': [val_list[10]],
        'DeviceProtection_No': [val_list[11]],
        'DeviceProtection_No internet service': [val_list[12]],


        # TechSupport {TechSupport_No	TechSupport_Yes	TechSupport_No internet service}
        'TechSupport_No': [val_list[13]],
        'TechSupport_Yes': [val_list[14]],
        'TechSupport_No internet service': [val_list[15]],

        # Contract {Contract_One year	Contract_Month-to-month	Contract_Two year}
        'Contract_One year': [val_list[16]],
        'Contract_Month-to-month': [val_list[17]],
        'Contract_Two year': [val_list[18]],

        # PaperlessBilling {PaperlessBilling_No	PaperlessBilling_Yes}
        'PaperlessBilling_No': [val_list[19]],
        'PaperlessBilling_Yes': [val_list[20]],

        # PaymentMethod {PaymentMethod_Mailed check,	PaymentMethod_Bank transfer (automatic),	PaymentMethod_Electronic check, PaymentMethod_Credit card (automatic)}
        'PaymentMethod_Mailed check': [val_list[21]],
        'PaymentMethod_Bank transfer (automatic)': [val_list[22]],
        'PaymentMethod_Electronic check': [val_list[23]],
        'PaymentMethod_Credit card (automatic)': [val_list[24]],

        # MonthlyCharges {Done}
        'MonthlyCharges': [monthCharges],

        # TotalCharges {Done}
        'TotalCharges': [totalCharge],

        # tenure {tenure_cat}
        'tenure_cat': [tenure],

        # Family {family_0.0 family_1.0}
        'family_0.0': [val_list[25]],
        'family_1.0': [val_list[26]],

        # Streaming {Streaming_No	Streaming_both	Streaming_TV	Streaming_M}
        'Streaming_No': [val_list[27]],
        'Streaming_both': [val_list[28]],
        'Streaming_TV': [val_list[29]],
        'Streaming_M': [val_list[30]],
        
    })


    # Make predictions using the trained model
    prediction = model.predict(user_data)[0]

    # print(user_data)

    txt_color = 'black'

    if prediction == 1:
        result_text = "The customer is likely to leave the business."
        txt_color = 'red'
    else:
        result_text = "The customer is not likely to leave the business."
        txt_color = 'green'

    return html.H1(result_text, style={'color': txt_color})

if __name__ == '__main__':
    app.run_server(debug=True)
