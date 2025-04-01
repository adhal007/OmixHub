
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import ot
from sklearn.preprocessing import StandardScaler
from scipy.stats import nbinom
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class AutoencoderSimulator:
    def __init__(self, data_from_bq):
        self.data_from_bq = data_from_bq
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.latent_dim = 64
        self.input_dim = None
        self.gene_expr = None
        
    def identify_similar_samples(self):
        normal_samples = self.data_from_bq[self.data_from_bq['tissue_type'] == 'Normal']
        tumor_samples = self.data_from_bq[self.data_from_bq['tissue_type'] == 'Tumor']
        
        expr_col = 'expr_unstr_count'
        normal_expr = np.log1p(np.vstack(normal_samples[expr_col].apply(pd.Series).values))
        tumor_expr = np.log1p(np.vstack(tumor_samples[expr_col].apply(pd.Series).values))
        
        normal_expr_scaled = self.scaler.fit_transform(normal_expr)
        tumor_expr_scaled = self.scaler.transform(tumor_expr)
        
        a, b = ot.unif(normal_expr_scaled.shape[0]), ot.unif(tumor_expr_scaled.shape[0])
        M = ot.dist(normal_expr_scaled, tumor_expr_scaled)
        G = ot.emd(a, b, M)
        
        similar_indices = G.argmax(axis=1)
        unique_similar_indices = np.unique(similar_indices)[:len(normal_samples)]
        
        return tumor_samples.iloc[unique_similar_indices]

    def preprocess_data(self):

        similar_tumor_samples = self.identify_similar_samples()        
        # Use only the similar tumor samples
        expr_col = 'expr_unstr_count'
        self.gene_expr = np.vstack(similar_tumor_samples[expr_col].apply(pd.Series).values)
        gene_expr_log = np.log1p(self.gene_expr)
        return self.scaler.fit_transform(gene_expr_log)


    def custom_loss(self, y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        return mse + 0.1 * mae

    def build_autoencoder(self, input_dim):
        self.input_dim = input_dim
        
        # Encoder
        encoder_input = keras.layers.Input(shape=(input_dim,))
        x = keras.layers.Dense(512, activation='relu')(encoder_input)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        encoder_output = keras.layers.Dense(self.latent_dim, activation='relu')(x)
        self.encoder = keras.Model(encoder_input, encoder_output)

        # Decoder
        decoder_input = keras.layers.Input(shape=(self.latent_dim,))
        x = keras.layers.Dense(128, activation='relu')(decoder_input)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(512, activation='relu')(x)
        decoder_output = keras.layers.Dense(input_dim, activation='linear')(x)
        self.decoder = keras.Model(decoder_input, decoder_output)

        # Autoencoder
        autoencoder_input = keras.layers.Input(shape=(input_dim,))
        encoded = self.encoder(autoencoder_input)
        decoded = self.decoder(encoded)
        scaling_factor = keras.layers.Dense(1, activation='sigmoid')(encoded)
        scaled_output = keras.layers.Multiply()([decoded, scaling_factor])
        self.autoencoder = keras.Model(autoencoder_input, scaled_output)
        
        self.autoencoder.compile(optimizer='adam', loss=self.custom_loss)
        
        return self.autoencoder

    def train_autoencoder(self, data, epochs=200, batch_size=32):
        input_dim = data.shape[1]
        self.build_autoencoder(input_dim)
        
        early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=5)
        
        self.autoencoder.fit(data, data, epochs=epochs, batch_size=batch_size, 
                             shuffle=True, verbose=1, 
                             callbacks=[early_stopping, reduce_lr],
                             validation_split=0.1)

    def simulate_samples(self, num_samples):
        # Generate random noise in the latent space
        random_latent_points = np.random.normal(size=(num_samples, self.latent_dim))
        
        # Decode the random latent points
        generated_samples = self.decoder.predict(random_latent_points)
        
        # Apply scaling factor
        scaling_factor = keras.models.Model(self.autoencoder.input, self.autoencoder.layers[-2].output).predict(generated_samples)
        generated_samples *= scaling_factor
        
        # Inverse transform the generated samples
        generated_samples = self.scaler.inverse_transform(generated_samples)
        generated_samples = np.expm1(generated_samples)
        
        # Post-processing: adjust the overall distribution
        for i in range(generated_samples.shape[1]):
            orig_mean = np.mean(self.gene_expr[:, i])
            sim_mean = np.mean(generated_samples[:, i])
            generated_samples[:, i] *= (orig_mean / sim_mean) if sim_mean > 0 else 1
        
        # Ensure non-negative integer values
        generated_samples = np.round(np.maximum(generated_samples, 0)).astype(int)
        
        # Cap the maximum value to the maximum observed in the original normal samples
        max_original = np.max(self.gene_expr)
        generated_samples = np.minimum(generated_samples, max_original)
        
        # Create a DataFrame with the simulated samples
        simulated_df = pd.DataFrame(generated_samples, columns=[f'gene_{i}' for i in range(self.input_dim)])
        simulated_df['tissue_type'] = 'Simulated Normal'
        simulated_df['case_id'] = [f'simulated_normal_{i}' for i in range(num_samples)]
        
        expr_col = 'expr_unstr_count'
        simulated_df[expr_col] = simulated_df.apply(lambda row: row.drop(['tissue_type', 'case_id']).tolist(), axis=1)
        simulated_df = simulated_df[['case_id', 'tissue_type', expr_col]] 
        return simulated_df

    def plot_similarity_heatmap(self):
        normal_samples = self.data_from_bq[self.data_from_bq['tissue_type'] == 'Normal']
        similar_tumor_samples = self.identify_similar_samples()
        
        expr_col = 'expr_unstr_count'
        normal_expr = np.log1p(np.vstack(normal_samples[expr_col].apply(pd.Series).values))
        similar_tumor_expr = np.log1p(np.vstack(similar_tumor_samples[expr_col].apply(pd.Series).values))
        
        # Combine normal and similar tumor samples
        combined_expr = np.vstack([normal_expr, similar_tumor_expr])
        
        # Create labels for the y-axis
        y_labels = ['Normal'] * len(normal_expr) + ['Similar Tumor'] * len(similar_tumor_expr)
        
        # Create a DataFrame for the heatmap
        heatmap_df = pd.DataFrame(combined_expr, index=y_labels)
        
        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Create the heatmap and store the returned mappable
        heatmap = sns.heatmap(heatmap_df, cmap='viridis', yticklabels=False, cbar=False, vmin=0, vmax=20, ax=ax)
        
        # Customize the plot with larger font sizes
        ax.set_title('Comparison of Normal Samples to Most Similar Tumor Samples', fontsize=20)
        ax.set_xlabel('Genes', fontsize=16)
        ax.set_ylabel('Samples', fontsize=16)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        
        # Add a color bar legend with larger font
        cbar = fig.colorbar(heatmap.collections[0], ax=ax)
        cbar.set_label('Log1p Expression Value', fontsize=16)
        cbar.ax.tick_params(labelsize=12)
        
        # Add a horizontal line to separate Normal and Similar Tumor samples
        ax.axhline(y=len(normal_expr), color='red', linestyle='--', linewidth=2)
        
        # Add text labels for Normal and Similar Tumor sections with larger font
        ax.text(-30, len(normal_expr) / 2, 'Normal', rotation=90, verticalalignment='center', fontsize=16, fontweight='bold')
        ax.text(-30, len(normal_expr) + len(similar_tumor_expr) / 2, 'Similar Tumor', rotation=90, verticalalignment='center', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig

########### VAE Section
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

def vae_loss(encoder):
    def loss(inputs, reconstruction):
        z_mean, z_log_var, _ = encoder(inputs)
        
        # Clip input values to prevent extremely large numbers
        inputs_clipped = tf.clip_by_value(inputs, 1e-8, 1e8)
        reconstruction_clipped = tf.clip_by_value(reconstruction, 1e-8, 1e8)
        
        # Use mean squared error for reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.square(tf.math.log1p(inputs_clipped) - tf.math.log1p(reconstruction_clipped))
        )
        
        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        
        total_loss = reconstruction_loss + kl_loss
        return total_loss
    return loss

class VariationalAutoencoderSimulator:
    def __init__(self, data_from_bq):
        self.data_from_bq = data_from_bq
        self.scaler = StandardScaler()
        self.vae = None
        self.encoder = None
        self.decoder = None
        self.latent_dim = 32
        self.input_dim = None
        self.gene_expr = None
        self.normal_samples = None
        self.data_max = None
        self.data_min = None
        
    def identify_similar_samples(self):
        normal_samples = self.data_from_bq[self.data_from_bq['tissue_type'] == 'Normal']
        tumor_samples = self.data_from_bq[self.data_from_bq['tissue_type'] == 'Tumor']
        
        expr_col = 'expr_unstr_count'
        normal_expr = np.vstack(normal_samples[expr_col].apply(pd.Series).values)
        tumor_expr = np.vstack(tumor_samples[expr_col].apply(pd.Series).values)
        
        normal_expr_scaled = self.scaler.fit_transform(normal_expr)
        tumor_expr_scaled = self.scaler.transform(tumor_expr)
        
        a, b = ot.unif(normal_expr_scaled.shape[0]), ot.unif(tumor_expr_scaled.shape[0])
        M = ot.dist(normal_expr_scaled, tumor_expr_scaled)
        G = ot.emd(a, b, M)
        
        similar_indices = G.argmax(axis=1)
        unique_similar_indices = np.unique(similar_indices)[:len(normal_samples)]
        
        return tumor_samples.iloc[unique_similar_indices]

    def preprocess_data(self):
        similar_tumor_samples = self.identify_similar_samples()
        self.normal_samples = self.data_from_bq[self.data_from_bq['tissue_type'] == 'Normal']
        expr_col = 'expr_unstr_count'
        self.gene_expr = np.vstack(similar_tumor_samples[expr_col].apply(pd.Series).values)
        
        # Apply log1p transformation and scale the data
        self.gene_expr = np.log1p(self.gene_expr)
        self.data_max = np.max(self.gene_expr)
        self.data_min = np.min(self.gene_expr)
        self.gene_expr = (self.gene_expr - self.data_min) / (self.data_max - self.data_min)
        
        return self.gene_expr

    def build_vae(self, input_dim):
        self.input_dim = input_dim
        
        # Encoder
        encoder_inputs = keras.layers.Input(shape=(input_dim,))
        x = keras.layers.Dense(256, activation='relu')(encoder_inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        z_mean = keras.layers.Dense(self.latent_dim)(x)
        z_log_var = keras.layers.Dense(self.latent_dim)(x)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        decoder_inputs = keras.layers.Input(shape=(self.latent_dim,))
        x = keras.layers.Dense(128, activation='relu')(decoder_inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        decoder_outputs = keras.layers.Dense(input_dim, activation='softplus')(x)
        self.decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

        # VAE
        self.vae = VAE(self.encoder, self.decoder)
        
        initial_learning_rate = 1e-4
        optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
        
        self.vae.compile(optimizer=optimizer, loss=vae_loss(self.encoder))
        
        return self.vae

    def train_autoencoder(self, data, epochs=200, batch_size=32):
        input_dim = data.shape[1]
        self.build_vae(input_dim)
        
        early_stopping = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=10, min_lr=1e-6)
        
        self.vae.fit(data, data, epochs=epochs, batch_size=batch_size, 
                     shuffle=True, verbose=1, 
                     callbacks=[early_stopping, reduce_lr],
                     validation_split=0.1)

    def simulate_samples(self, num_samples):
        # Generate random noise in the latent space
        random_latent_points = np.random.normal(size=(num_samples, self.latent_dim))
        
        # Decode the random latent points
        generated_samples = self.decoder.predict(random_latent_points)
        
        # Inverse transform the generated samples
        generated_samples = generated_samples * (self.data_max - self.data_min) + self.data_min
        generated_samples = np.expm1(generated_samples)
        
        # Post-processing: adjust the overall distribution and variance
        expr_col = 'expr_unstr_count'
        normal_expr = np.vstack(self.normal_samples[expr_col].apply(pd.Series).values)
        
        for i in range(generated_samples.shape[1]):
            orig_mean = np.mean(normal_expr[:, i])
            orig_std = np.std(normal_expr[:, i])
            sim_mean = np.mean(generated_samples[:, i])
            sim_std = np.std(generated_samples[:, i])
            
            # Adjust mean and standard deviation
            generated_samples[:, i] = (generated_samples[:, i] - sim_mean) / sim_std * orig_std + orig_mean
        
        # Ensure non-negative integer values
        generated_samples = np.round(np.maximum(generated_samples, 0)).astype(int)
        
        # Cap the maximum value to the maximum observed in the original normal samples
        max_original = np.max(normal_expr)
        generated_samples = np.minimum(generated_samples, max_original)
        
        # Create a DataFrame with the simulated samples
        simulated_df = pd.DataFrame(generated_samples, columns=[f'gene_{i}' for i in range(self.input_dim)])
        simulated_df['tissue_type'] = 'Simulated Normal'
        simulated_df['case_id'] = [f'simulated_normal_{i}' for i in range(num_samples)]
        
        simulated_df[expr_col] = simulated_df.apply(lambda row: row.drop(['tissue_type', 'case_id']).tolist(), axis=1)
        simulated_df = simulated_df[['case_id', 'tissue_type', expr_col]] 
        return simulated_df

