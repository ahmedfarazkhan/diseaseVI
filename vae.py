#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vae: variational autoencoders to disentangle co-existing patterns in controls and patients
    Definitions for autoencoder variants with and without contrastive loss 
    Includes
        - Regular autoencoder (autoencoder())
        - Variational autoencoder (standard_vae())
        - Contrastive VAE (contrastive_vae())
            - Use two pools of latent variables to separate target vs. background patterns
            - "Salient" latent variables contribute only to target samples
            - "Irrelevant" latent variables contribute to all (target + background)
            
@author: ahmedkhan
August 10, 2023
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras import backend as K

def autoencoder(input_dim=4, intermediate_dim=12, latent_dim=2):
    '''
    Regular autoencoder

    Parameters
    ----------
    input_dim : int, optional
        Features in input data
    intermediate_dim : int, optional
        Size of intermediate layer(s)
    latent_dim : int, optional
        Dimensions of latent features

    Returns
    -------
    ae : model
        Full AE model
    encoder : model
        Encoder model
    decoder : model
        Decoder model

    '''
    input_shape = (input_dim, )

    ### Encoder
    inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
    if isinstance(intermediate_dim, int):
        x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(inputs)
    else:
        x = inputs
        for dim in intermediate_dim:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)
            
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

    # Reparametrization trick
    z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Encoder model
    encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    ### Decoder definition
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')
    if isinstance(intermediate_dim, int):
        x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    else:
        x = latent_inputs
        for dim in intermediate_dim:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)        
    outputs = tf.keras.layers.Dense(input_dim)(x)

    # Decoder model
    decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')

    ### AE model
    outputs = decoder(encoder(inputs)[2])
    ae = tf.keras.models.Model(inputs, outputs, name='ae_mlp')

    # Loss function (just reconstruction loss)
    reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
    reconstruction_loss *= input_dim
    
    ae_loss = tf.keras.backend.mean(reconstruction_loss)
    ae.add_loss(ae_loss)
    ae.compile(optimizer='adam')
    return ae, encoder, decoder    

def standard_vae(input_dim=4, intermediate_dim=12, latent_dim=2, beta=1):
    '''
    Regular variational autoencoder

    Parameters
    ----------
    input_dim : int, optional
        Features in input data
    intermediate_dim : int, optional
        Size of intermediate layer(s)
    latent_dim : int, optional
        Dimensions of latent features
    beta : float, optional
        Weights the loss function components. The default is 1.

    Returns
    -------
    vae : model
        Full VAE model
    encoder : model
        Encoder model
    decoder : model
        Decoder model

    '''
    input_shape = (input_dim, )

    ### Encoder definition
    inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_input')
    if isinstance(intermediate_dim, int):
        x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(inputs)
    else:
        x = inputs
        for dim in intermediate_dim:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

    # Use reparameterization trick to push the sampling out as input
    z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Encoder model 
    encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z], name='encoder') 

    ### Decoder definition
    latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')
    if isinstance(intermediate_dim, int):
        x = tf.keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    else:
        x = latent_inputs
        for dim in intermediate_dim:
            x = tf.keras.layers.Dense(dim, activation='relu')(x)        
    outputs = tf.keras.layers.Dense(input_dim)(x)

    # Decoder mdoel
    decoder = tf.keras.models.Model(latent_inputs, outputs, name='decoder')

    ### VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = tf.keras.models.Model(inputs, outputs, name='vae_mlp')

    # Loss function (reconstruction loss & KL loss)
    reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
    reconstruction_loss *= input_dim

    kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5  
    
    vae_loss = tf.keras.backend.mean(reconstruction_loss + beta*kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder, decoder

def contrastive_vae(input_dim=4, intermediate_dim=12, latent_dim=2, batch_size=100, beta=1, disentangle=False, gamma=0):
    '''
    Contrastive VAE 
    Use two populations of latent units:
        "Salient" s : contributes to target and background samples 
        "Irrelevant" z : contributes only to background samples

    Parameters 
    ----------
    input_dim : int, optional
        Features in input data
    intermediate_dim : int, optional
        Size of intermediate layer(s)
    latent_dim : int, optional
        Dimensions of latent features
    batch_size : int, optional
        Batch size
    beta : float, optional
        Weights the loss function components. The default is 1.
    disentangle : bool, optional
       Weights the loss function components. The default is 1.
    gamma : float, optional
        Weights the loss function components. The default is 0.

    Returns
    -------
    vae : model
        Full VAE model
    encoder : model
        Encoder model
    decoder : model
        Decoder model

    '''
    input_shape = (input_dim, )
    
    # Target and background inputs
    tg_inputs = tf.keras.layers.Input(shape=input_shape, name='tg_input')
    bg_inputs = tf.keras.layers.Input(shape=input_shape, name='bg_input')
    
    if isinstance(intermediate_dim, int):
        intermediate_dim = [intermediate_dim]
    
    z_h_layers = []
    
    # He initialization 
    for i in range(len(intermediate_dim)):
        dim = intermediate_dim[i]
        
        if i==0:
            curr_std = 2/input_dim
        else:
            curr_std = 2/intermediate_dim[i-1]
                
        z_h_layers.append(tf.keras.layers.Dense(dim, activation='relu',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=curr_std)))
        
    z_mean_layer = tf.keras.layers.Dense(latent_dim, name='z_mean')
    z_log_var_layer = tf.keras.layers.Dense(latent_dim, name='z_log_var')
    z_layer = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')

    def z_encoder_func(inputs):
        z_h = inputs
        for z_h_layer in z_h_layers:
            z_h = z_h_layer(z_h)
        z_mean = z_mean_layer(z_h)
        z_log_var = z_log_var_layer(z_h)
        z = z_layer([z_mean, z_log_var])
        return z_mean, z_log_var, z

    s_h_layers = []
    for i in range(len(intermediate_dim)):
        dim = intermediate_dim[i]
        
        if i==0:
            curr_std = 2/input_dim
        else:
            curr_std = 2/intermediate_dim[i-1]
            
        s_h_layers.append(tf.keras.layers.Dense(dim, activation='relu',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=curr_std)))
        
    s_mean_layer = tf.keras.layers.Dense(latent_dim, name='s_mean')
    s_log_var_layer = tf.keras.layers.Dense(latent_dim, name='s_log_var')
    s_layer = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='s')

    def s_encoder_func(inputs):
        s_h = inputs
        for s_h_layer in s_h_layers:
            s_h = s_h_layer(s_h)
        s_mean = s_mean_layer(s_h)
        s_log_var = s_log_var_layer(s_h)
        s = s_layer([s_mean, s_log_var])
        return s_mean, s_log_var, s

    tg_z_mean, tg_z_log_var, tg_z = z_encoder_func(tg_inputs)
    tg_s_mean, tg_s_log_var, tg_s = s_encoder_func(tg_inputs)
    bg_s_mean, bg_s_log_var, bg_s = s_encoder_func(bg_inputs)

    # Salient s, irrelevant z
    z_encoder = tf.keras.models.Model(tg_inputs, [tg_z_mean, tg_z_log_var, tg_z], name='z_encoder')
    s_encoder = tf.keras.models.Model(tg_inputs, [tg_s_mean, tg_s_log_var, tg_s], name='s_encoder')

    # Build decoder model
    cvae_latent_inputs = tf.keras.layers.Input(shape=(2 * latent_dim,), name='sampled')
    cvae_h = cvae_latent_inputs
    for i in range(len(intermediate_dim)):
        dim = intermediate_dim[i]
        
        if i==0:
            curr_std = 2/input_dim
        else:
            curr_std = 2/intermediate_dim[i-1]
            
        cvae_h = tf.keras.layers.Dense(dim, activation='relu', 
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=curr_std))(cvae_h)
    cvae_outputs = tf.keras.layers.Dense(input_dim)(cvae_h)

    cvae_decoder = tf.keras.models.Model(inputs=cvae_latent_inputs, outputs=cvae_outputs, name='decoder')


    def zeros_like(x):
        return tf.zeros_like(x)

    tg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, tg_s], -1))
    zeros = tf.keras.layers.Lambda(zeros_like)(tg_z)
    bg_outputs = cvae_decoder(tf.keras.layers.concatenate([zeros, bg_s], -1))
    fg_outputs = cvae_decoder(tf.keras.layers.concatenate([tg_z, zeros], -1))

    cvae = tf.keras.models.Model(inputs=[tg_inputs, bg_inputs], 
                                 outputs=[tg_outputs, bg_outputs], 
                                 name='contrastive_vae')

    cvae_fg = tf.keras.models.Model(inputs=tg_inputs, 
                                    outputs=fg_outputs, 
                                    name='contrastive_vae_fg')

    
    if disentangle:
        discriminator = Dense(1, activation='sigmoid')
        
        z1 = Lambda(lambda x: x[:int(batch_size/2),:])(tg_z)
        z2 = Lambda(lambda x: x[int(batch_size/2):,:])(tg_z)
        s1 = Lambda(lambda x: x[:int(batch_size/2),:])(tg_s)
        s2 = Lambda(lambda x: x[int(batch_size/2):,:])(tg_s)
        q_bar = tf.keras.layers.concatenate(
            [tf.keras.layers.concatenate([s1, z2], axis=1),
            tf.keras.layers.concatenate([s2, z1], axis=1)],
            axis=0)
        q = tf.keras.layers.concatenate(
            [tf.keras.layers.concatenate([s1, z1], axis=1),
            tf.keras.layers.concatenate([s2, z2], axis=1)],
            axis=0)
        q_bar_score = discriminator(q_bar)
        q_score = discriminator(q)        
        tc_loss = K.log(q_score / (1 - q_score)) 
        
        discriminator_loss = - K.log(q_score) - K.log(1 - q_bar_score)

    reconstruction_loss = tf.keras.losses.mse(tg_inputs, tg_outputs)
    reconstruction_loss += tf.keras.losses.mse(bg_inputs, bg_outputs)
    reconstruction_loss *= input_dim

    kl_loss = 1 + tg_z_log_var - tf.keras.backend.square(tg_z_mean) - tf.keras.backend.exp(tg_z_log_var)
    kl_loss += 1 + tg_s_log_var - tf.keras.backend.square(tg_s_mean) - tf.keras.backend.exp(tg_s_log_var)
    kl_loss += 1 + bg_s_log_var - tf.keras.backend.square(bg_s_mean) - tf.keras.backend.exp(bg_s_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    if disentangle:
        cvae_loss = K.mean(reconstruction_loss) + beta*K.mean(kl_loss) + gamma * K.mean(tc_loss) + K.mean(discriminator_loss)
    else:
        cvae_loss = K.mean(reconstruction_loss) + K.mean(kl_loss)

    cvae.add_loss(cvae_loss)
    cvae.compile(optimizer='adam')
    
    return cvae, cvae_fg, z_encoder, s_encoder, cvae_decoder

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon