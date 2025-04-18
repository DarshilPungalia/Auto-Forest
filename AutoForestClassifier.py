import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def cross_similarity(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return bce * 0.6 + (1 - ssim) * 0.4

class AutoencoderClassifier:
    def __init__(self):
        self.predictor = tf.keras.models.load_model(r"C:\Users\LENOVO\Desktop\ThresholdPredictor.keras")

        self.unet_1_encoder = tf.keras.models.load_model(r"C:\Users\LENOVO\Desktop\Auto-Forest\models\UnetInspired.keras", custom_objects={'cross_similarity': cross_similarity})
        self.unet_2_encoder = tf.keras.models.load_model(r"C:\Users\LENOVO\Desktop\Auto-Forest\models\Unet2.keras", custom_objects={'cross_similarity': cross_similarity})
        self.residual_skip_encoder = tf.keras.models.load_model(r"C:\Users\LENOVO\Desktop\Auto-Forest\models\UResNetEncoder.keras", custom_objects={'cross_similarity': cross_similarity})
        self.attention_encoder = tf.keras.models.load_model(r"C:\Users\LENOVO\Desktop\Auto-Forest\models\AttentionEncoder.keras", custom_objects={'cross_similarity': cross_similarity})

        self.models = [self.unet_1_encoder, self.unet_2_encoder, self.residual_skip_encoder, self.attention_encoder]
        self.latent_layer_names = ['conv2d_11', 'conv2d_284', 'conv2d_59', 'batch_normalization_7']

    def build_latent_models(self):
        return [tf.keras.Model(inputs=model.input, outputs=model.get_layer(latent).output)
                for model, latent in zip(self.models, self.latent_layer_names)]

    def latent_outputs(self, input_data):
        input_data = input_data.prefetch(tf.data.AUTOTUNE)
        latent_models = self.build_latent_models()
        latent_matrix = []

        @tf.function
        def compute_latents(x_batch):
            latents = [tf.reshape(latent_model(x_batch, training=False), [tf.shape(x_batch)[0], -1])
                       for latent_model in latent_models]
            return tf.concat(latents, axis=1)

        for batch in input_data:
            if isinstance(batch, tuple):
                x_batch, _ = batch
            else:
                x_batch = batch

            concat_latents = compute_latents(x_batch)
            latent_matrix.append(concat_latents)

        return tf.concat(latent_matrix, axis=0)

    def evaluate(self, input_data):
        input_data = input_data.prefetch(tf.data.AUTOTUNE)
        all_losses = []

        @tf.function
        def compute_losses(x_batch, y_batch):
            batch_size = tf.shape(x_batch)[0]
            model_losses = []

            for model in self.models:
                predictions = model(x_batch, training=False)
                loss = model.compute_loss(x_batch, y_batch, predictions)
                if len(tf.shape(loss)) == 0:
                    loss = tf.repeat(loss, batch_size)
                model_losses.append(loss)

            return tf.stack(model_losses, axis=0)

        for batch in input_data:
            if isinstance(batch, tuple):
                x_batch, y_batch = batch
            else:
                x_batch = y_batch = batch

            batch_losses = compute_losses(x_batch, y_batch)
            all_losses.append(batch_losses)

        if not all_losses:
            return tf.zeros([0, len(self.models)])

        concatenated = tf.concat(all_losses, axis=1)
        return tf.transpose(concatenated)

    def inference(self, data, threshold = 0.33978017):
        if isinstance(data, tf.data.Dataset):
            loss = self.evaluate(data)
            latent = self.latent_outputs(data)
        else:
            x = data
            if not isinstance(x, tf.Tensor):
                x = tf.convert_to_tensor(x, dtype=tf.float32)
            if len(x.shape) == 3:
                x = tf.expand_dims(x, axis=0)
            x = tf.data.Dataset.from_tensor_slices(x).batch(32).prefetch(tf.data.AUTOTUNE)
            loss = self.evaluate(x)
            latent = self.latent_outputs(x)

        concat_data = tf.concat([loss, latent], axis=1)
        scores = self.predictor(concat_data, training=False)
        predictions = np.array('MBM' if scores >= threshold else 'Not MBM')
            
        return predictions

    def ensemble_prediction(self, input_data, use_ensemble=True, method='average', weights=None, selected_model='residual_skip'):
        selected_model_obj = {
            'unet_1': self.unet_1_encoder,
            'unet_2': self.unet_2_encoder,
            'residual_skip': self.residual_skip_encoder,
            'attention': self.attention_encoder
        }.get(selected_model, self.residual_skip_encoder)

        if isinstance(input_data, tf.data.Dataset):
            all_outputs = []
            for batch in input_data:
                if isinstance(batch, tuple):
                    batch = batch[0]

                if use_ensemble:
                    batch_outputs = [model.predict(batch, verbose=0) for model in self.models]
                    batch_outputs = tf.stack(batch_outputs, axis=0)

                    if method == 'average':
                        ensemble_batch = tf.reduce_mean(batch_outputs, axis=0)
                    elif method == 'weighted':
                        weights_tensor = tf.reshape(tf.convert_to_tensor(weights, dtype=tf.float32), (-1, 1, 1, 1, 1))
                        ensemble_batch = tf.reduce_sum(batch_outputs * weights_tensor, axis=0)
                    else:
                        raise ValueError(f"Unknown ensemble method: {method}")

                    all_outputs.append(ensemble_batch)
                else:
                    all_outputs.append(selected_model_obj.predict(batch, verbose=0))

            return tf.concat(all_outputs, axis=0)

        else:
            if use_ensemble:
                outputs = [model.predict(input_data, verbose=0) for model in self.models]
                outputs = tf.stack(outputs, axis=0)

                if method == 'average':
                    return tf.reduce_mean(outputs, axis=0)
                elif method == 'weighted':
                    weights_tensor = tf.reshape(tf.convert_to_tensor(weights, dtype=tf.float32), (-1, 1, 1, 1, 1))
                    return tf.reduce_sum(outputs * weights_tensor, axis=0)
                else:
                    raise ValueError(f"Unknown ensemble method: {method}")
                
            return selected_model_obj.predict(input_data, verbose=0)

    def auto_forest_pipeline(self, data, use_ensemble=True, method='average', weights=None, selected_model='residual_skip'):
        classification = self.inference(data)
        reconstruction = self.ensemble_prediction(data, use_ensemble, method, weights, selected_model)
        return classification, reconstruction

def load_image_from_path(image_path, image_size=(120, 120)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size, color_mode='grayscale')
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)
