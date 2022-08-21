from ast import Mod
from flask import Flask,request, url_for, redirect, render_template, jsonify
import tensorflow as tf
import tensorflow as tf
import json
import datetime

# Initalise the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():    
    #model_path = '/MyApp/Models/baggage-predictor/1'
    #loaded_model = tf.saved_model.load(model_path)
    #inference_fn = loaded_model.signatures['serving_rest']
    departure = request.form.get("departure")
    if (not departure):
        today = datetime.datetime.strptime
    departure_t = tf.constant(request.form.get("departure"), dtype=tf.string, shape=(1,1))
    arrival_t = tf.constant(request.form.get("arrival"), dtype=tf.string, shape=(1,1))
    adults_t = tf.constant(request.form.get("adults"), dtype=tf.int64, shape=(1,1))
    children_t = tf.constant(request.form.get("children"), dtype=tf.int64, shape=(1,1))
    infants_t = tf.constant(request.form.get("infants"), dtype=tf.int64, shape=(1,1))
    trip_type_t = tf.constant(request.form.get("trip_type"), dtype=tf.string, shape=(1,1))
    train_t = tf.constant(request.form.get("train"), dtype=tf.string, shape=(1,1))
    gds_t = tf.constant(request.form.get("gds"), dtype=tf.int64, shape=(1,1))
    haul_type_t = tf.constant(request.form.get("haul_type"), dtype=tf.string, shape=(1,1))
    no_gds_t = tf.constant(request.form.get("no_gds"), dtype=tf.int64, shape=(1,1))
    website_t = tf.constant(request.form.get("website"), dtype=tf.string, shape=(1,1))
    product_t = tf.constant(request.form.get("product"), dtype=tf.string, shape=(1,1))
    sms_t = tf.constant(request.form.get("sms"), dtype=tf.string, shape=(1,1))
    distance_t = tf.constant(request.form.get("distance"), dtype=tf.float64, shape=(1,1))

    #result = inference_fn(departure=departure_t, 
    #                    arrival=arrival_t,
    #                    adults=adults_t,
    #                    children=children_t,
    #                    infants=infants_t,
    #                    trip_type=trip_type_t,
    #                    train=train_t,
    #                    gds=gds_t,
    #                    haul_type=haul_type_t,
    #                    no_gds=no_gds_t,
    #                    website=website_t,
    #                    product=product_t,
    #                    sms=sms_t,
    #                    distance=distance_t)

    

    #prediction = result['output_0'].numpy()

    prediction = 0
    
    return render_template('home.html',pred='Expected number of extra bags will be {}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)
