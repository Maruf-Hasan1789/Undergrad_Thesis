from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np


app = Flask(__name__)

Final_model = tf.keras.models.load_model('Thesis_Model_Fine_Tuning_Final_91.model')

def predict_class(img_path,model_name):
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    img = np.array(img, dtype='float32')
    img /= 255.0

    # Expand the image dimensions to (1, 100, 100, 3)
    img = np.expand_dims(img, axis=0)

    # Get the prediction
    return_string='none'
    pred = Final_model.predict(img)
    pred_class = np.argmax(pred, axis=1)

    # Map the class index to the class label
    if pred_class[0] == 0:
        return_string= "Normal"
    else:
        return_string= "Pneumonia"

    return return_string





app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    result='None'
    if request.method == 'POST':
        image = request.files['image']
        imgpath='1.jpeg'
        image.save(imgpath)
        model_name=request.form.get('Model_name')
        #print(type(model_name))
        predicted_class=predict_class(imgpath,model_name)
        print(predicted_class)
        result=predicted_class
        # Do something with the image
        ...

    return render_template("index_checking.html", output_string=result)

if __name__ == '__main__':
    app.run(debug=True)




