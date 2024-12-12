from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
import numpy as np
from imageio.v2 import imread, imwrite
from scipy.ndimage.filters import convolve

app = Flask(__name__)

def calc_energy(img):
    du_filter = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    du_filter = np.stack([du_filter] * 3, axis=2)

    dv_filter = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    dv_filter = np.stack([dv_filter] * 3, axis=2)

    img = img.astype('float32')

    convolved = np.absolute(convolve(img, du_filter)) + np.absolute(convolve(img, dv_filter))

    mapForEnergy = convolved.sum(axis=2)

    return mapForEnergy

def crop_c(img, c_scale):
    r, c, _ = img.shape
    new_c = int(c_scale * c)

    for i in range(c - new_c):
        img = carve_column(img)

    return img

def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def carve_column(img):
    r, c, _ = img.shape
    M, backtrack = minimum_seam(img)

    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def minimum_seam(img):
    r, c, _ = img.shape
    mapForEnergy = calc_energy(img)
    M = mapForEnergy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    axis = request.form['axis']
    scale = float(request.form['scale'])

    img = imread(file)

    if axis == 'r':
        processed_img = crop_r(img, scale)
    elif axis == 'c':
        processed_img = crop_c(img, scale)
    else:
        return "Invalid axis", 400

    output = BytesIO()
    imwrite(output, processed_img, format='png')
    output.seek(0)

    return send_file(output, mimetype='image/png')

if __name__ == '__main__':
    app.run()
