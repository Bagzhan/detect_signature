from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from detect import detect_signature

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # get the uploaded file
        file = request.files['image']

        # read the image data from the file object using OpenCV
        img_data = file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        result = detect_signature(img)
        # save the file to disk
        # file.save('uploaded_image.jpg')

        # display the uploaded image on the page
        return jsonify({'result': len(result)})
    else:
        # show the upload form
        return render_template('upload_image.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # retrieve image from request
    img_str = request.data
    nparr = np.fromstring(img_str, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # process image
    # ...

    # convert result to string
    result = detect_signature(img)

    # return result as JSON
    return jsonify({'result': len(result)})

if __name__ == '__main__':
    app.run(debug=True)
