import base64

import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from flask import Flask, request, send_file, jsonify, render_template_string

app = Flask(__name__)

img_size = 400
tf.random.set_seed(272)

# Load VGG19 model
vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='imagenet')
vgg.trainable = False

STYLE_LAYERS = [
    ('block1_conv1', 0.2),
    ('block2_conv1', 0.2),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.2)]

content_layer = [('block5_conv4', 1)]

# Content cost function
def compute_content_cost(content_output, generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.transpose(tf.reshape(a_C, shape=[m, -1, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, shape=[m, -1, n_C]))
    J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    return J_content

# Gram matrix function
def gram_matrix(A):
    GA = tf.matmul(A, A, transpose_b=True)
    return GA

# Style cost function
def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]))
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = (1 / (4 * n_C ** 2 * (n_H * n_W) ** 2)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    return J_style_layer

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    J_style = 0
    a_S = style_image_output[:-1]
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weight[1] * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha=10, beta=40):
    J = alpha * J_content + beta * J_style
    return J

# Function to get outputs of selected layers
def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function()
def train_step(generated_image, vgg_model_outputs, a_C, a_S):
    with tf.GradientTape() as tape:
        a_G = vgg_model_outputs(generated_image)
        J_style = compute_style_cost(a_S, a_G)
        J_content = compute_content_cost(a_C, a_G)
        J = total_cost(J_content, J_style)
    grad = tape.gradient(J, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    return J

vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Upload Images</title>
    </head>
    <body>
        <h1>Upload Content and Style Images</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <label for="content_image">Content Image:</label>
            <input type="file" name="content_image" id="content_image" required><br><br>
            <label for="style_image">Style Image:</label>
            <input type="file" name="style_image" id="style_image" required><br><br>
            <label for="epochs">Number of Epochs:</label>
            <input type="number" name="epochs" id="epochs" value="50" min="1" required><br><br>
            <button type="submit">Upload</button>
        </form>
        <div id="status"></div>
        <script>
            const form = document.querySelector('form');
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = new FormData(form);
                const statusDiv = document.getElementById('status');
                statusDiv.innerText = 'Processing...';
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.result) {
                    const img = document.createElement('img');
                    img.src = 'data:image/png;base64,' + data.result;
                    statusDiv.innerText = '';
                    statusDiv.appendChild(img);

                    // Create download link
                    const downloadLink = document.createElement('a');
                    downloadLink.href = 'data:image/png;base64,' + data.result;
                    downloadLink.download = 'stylized_image.png';
                    downloadLink.innerText = 'Download Image';
                    statusDiv.appendChild(downloadLink);
                } else {
                    statusDiv.innerText = 'An error occurred.';
                }
            });
        </script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload():
    content_image = request.files['content_image']
    style_image = request.files['style_image']
    epochs = int(request.form['epochs'])

    content_image = Image.open(content_image).resize((img_size, img_size))
    content_image = np.array(content_image)
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

    style_image = Image.open(style_image).resize((img_size, img_size))
    style_image = np.array(style_image)
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

    generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
    generated_image = tf.add(generated_image, noise)
    generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

    content_target = vgg_model_outputs(content_image)
    style_targets = vgg_model_outputs(style_image)

    preprocessed_content = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)

    preprocessed_style = tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S = vgg_model_outputs(preprocessed_style)

    generated_image = tf.Variable(generated_image)

    for i in range(epochs):
        train_step(generated_image, vgg_model_outputs, a_C, a_S)
        print(f'Epoch {i + 1}/{epochs}')

    result_image = tensor_to_image(generated_image)

    img_io = BytesIO()
    result_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Encode the image to base64 for inclusion in JSON response
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    return jsonify({'result': img_base64})

if __name__ == '__main__':
    app.run(debug=True)
