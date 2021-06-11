import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'C:\\Users\\home\\Desktop\\MLFLOW_FORECAST_PROJECT\\DS_Project_2020\\Dental_model_GBoost_Clf_model2.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('C:\\Users\\home\\Desktop\\MLFLOW_FORECAST_PROJECT\\main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        Gender = flask.request.form['Gender']
        Appointment_Day = flask.request.form['Appointment_Day']
        Employment_status = flask.request.form['Employment_status']
        Type_of_Treatment = flask.request.form['Type_of_Treatment']
        Cost_of_Treatment = flask.request.form['Cost_of_Treatment']
        Insurance = flask.request.form['Insurance']
        Age = flask.request.form['Age']
        Distance = flask.request.form['Distance']

        # Make DataFrame for model
        input_variables = pd.DataFrame([[Gender,Appointment_Day,Employment_status,Type_of_Treatment,Cost_of_Treatment,Insurance,Age,Distance]],
                                       columns=['Gender','Appointment_Day','Employment_status','Type_of_Treatment','Cost_of_Treatment','Insurance','Age','Distance'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('C:\\Users\\home\\Desktop\\MLFLOW_FORECAST_PROJECT\\main.html',
                                     original_input={'Gender':Gender,
                                                     'Appointment_Day':Appointment_Day,
                                                     'Employment_status':Employment_status,
                                                     'Type_of_Treatment':Type_of_Treatment,
                                                     'Cost_of_Treatment':Cost_of_Treatment,
                                                     'Insurance':Insurance,
                                                     'Age':Age,
                                                     'Distance':Distance},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()