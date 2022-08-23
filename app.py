from flask import Flask,request, render_template
import tensorflow as tf
import datetime
from tensorflow import keras
import os
from flask_bootstrap import Bootstrap

# Initalise the Flask app
app = Flask(__name__, template_folder='Templates')
bootstrap = Bootstrap(app)
def formatDate(date_string):
    if (not date_string):
        date_object = datetime.datetime.today()
    else:
        date_object = datetime.datetime.strptime(date_string, "%Y-%m-%d")    
    date_string = datetime.datetime.strftime(date_object, "%d/%B")
    return date_string

@app.route('/')
def home():    
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'Models/baggage-predictor')
    #loaded_model = tf.saved_model.load(model_path)
    loaded_model = tf.keras.models.load_model(model_path)
    inference_fn = loaded_model.signatures['serving_rest']
    departure = request.form.get("departure")
    arrival = request.form.get("arrival")    
    departure = formatDate(departure)
    arrival = formatDate(arrival)

    adults = int(request.form.get("adults"))
    children = int(request.form.get("children"))
    infants = int(request.form.get("infants"))
    trip_type = request.form.get("trip_type")
    train = request.form.get("train")
    sms = request.form.get("sms")
    train = "TRUE" if train else "FALSE"   
    sms = "TRUE" if sms else "FALSE"   

    gds = int(request.form.get("gds"))
    no_gds = int(request.form.get("no_gds"))
    haul_type = request.form.get("haul_type")
    website = request.form.get("website")
    product = request.form.get("product")
    distance = float(request.form.get("distance"))

    departure_t = tf.constant(departure, dtype=tf.string, shape=(1,1))
    arrival_t = tf.constant(request.form.get("arrival"), dtype=tf.string, shape=(1,1))
    adults_t = tf.constant(adults, dtype=tf.int64, shape=(1,1))
    children_t = tf.constant(children, dtype=tf.int64, shape=(1,1))
    infants_t = tf.constant(infants, dtype=tf.int64, shape=(1,1))
    trip_type_t = tf.constant(trip_type, dtype=tf.string, shape=(1,1))
    train_t = tf.constant(train, dtype=tf.string, shape=(1,1))
    gds_t = tf.constant(gds, dtype=tf.int64, shape=(1,1))
    haul_type_t = tf.constant(haul_type, dtype=tf.string, shape=(1,1))
    no_gds_t = tf.constant(no_gds, dtype=tf.int64, shape=(1,1))
    website_t = tf.constant(website, dtype=tf.string, shape=(1,1))
    product_t = tf.constant(product, dtype=tf.string, shape=(1,1))
    sms_t = tf.constant(sms, dtype=tf.string, shape=(1,1))
    distance_t = tf.constant(distance, dtype=tf.float64, shape=(1,1))

    result = inference_fn(departure=departure_t, 
                        arrival=arrival_t,
                        adults=adults_t,
                        children=children_t,
                        infants=infants_t,
                        trip_type=trip_type_t,
                        train=train_t,
                        gds=gds_t,
                        haul_type=haul_type_t,
                        no_gds=no_gds_t,
                        website=website_t,
                        product=product_t,
                        sms=sms_t,
                        distance=distance_t)

    

    prediction = result['output_0'].numpy()[0][0]
    prediction = format(prediction, '.2f')
    
    
    return render_template('home.html',pred=prediction)

if __name__ == '__main__':
    app.run(debug=True)
