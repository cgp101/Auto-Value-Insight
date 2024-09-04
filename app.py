from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load your model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

# Define mappings for categorical variables
fuel_mapping = {
    'Petrol': 0,
    'Diesel': 1,
    'CNG': -1,          
    'LPG': -1,          
    'Electric': -1      
}

seller_type_mapping = {
    'Individual': 0,
    'Dealer': 1,
    'Trustmark Dealer': 2
}

transmission_mapping = {
    'Manual': 1,
    'Automatic': 0
}

owner_mapping = {
    'Test Drive Car': 0,
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        year = int(request.form['year'])
        km_driven = int(request.form['km_driven'])
        fuel = fuel_mapping[request.form['fuel']]
        seller_type = seller_type_mapping[request.form['seller_type']]
        transmission = transmission_mapping[request.form['transmission']]
        owner = owner_mapping[request.form['owner']]
        mileage = float(request.form['mileage'])
        engine = int(request.form['engine'])
        max_power = float(request.form['max_power'])
        seats = int(request.form['seats'])
        torque_nm = int(request.form['torque_nm'])
        torque_rpm = int(request.form['torque_rpm'])

        # Handle unsupported fuel types (CNG, LPG, Electric)
        if fuel == -1:
            return render_template('index.html', prediction_text='Fuel type not supported.')

        # Create a feature array for prediction
        features = np.array([[year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats, torque_nm, torque_rpm]])

        # Predict using the model
        prediction = model.predict(features)

        return render_template('index.html', prediction_text=f'The predicted selling price of the car is Rs. {prediction[0]:.2f}')
    
    except KeyError as e:
        # Handle any missing form fields or incorrect mappings
        return render_template('index.html', prediction_text=f'Invalid input: {str(e)}')
    
    except ValueError as e:
        # Handle invalid number conversion
        return render_template('index.html', prediction_text='Invalid input: Please enter valid numbers.')

if __name__ == "__main__":
    app.run(debug=True)
