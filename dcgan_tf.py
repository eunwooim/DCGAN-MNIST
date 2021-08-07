import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow.keras import layers

from IPython import display

def trainset():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = np.expand_dims(train_images,axis=-1).astype(np.float32)
    print(np.shape(train_images))
    train_images = train_images / 127.5 - 1
    dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(256)
    return dataset

def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7,7,256)))
    model.add(layers.Conv2DTranspose(128,(5,5), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64,(5,5), strides=(2,2), 
                                     padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1,(5,5), strides=(2,2), padding='same',
                                     use_bias=False, activation='tanh'))
    return model

def discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def ckpt(g_optimizer, d_optimizer, args):
    checkpoint_dir = args.save_path
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=g_optimizer,
                                    discriminator_optimizer=d_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    return checkpoint_prefix, checkpoint

@tf.function
def fit(images, args):
    noise = tf.random.normal([args.batch_size, 100])
    
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        # Forward Propagation
        generated = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated, training=True)

        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)
    
    # Backward Propagation
    g_grad = g_tape.gradient(g_loss, generator.trainable_variables)
    d_grad = d_tape.gradient(d_loss, discriminator.trainable_variables)

    g_optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))
    d_optimizer.apply_tradients(zip(d_grad, discriminator.trainable_variables))
    
def save_img(model, epoch, noise):
    predictions = model(noise, training=False)
    fig = plt.figures(figsize=(4,4))
    
    for i in range(np.shape(predictions)[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(np.squeeze(predictions)[i]*127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.savefig('epoch_{:04d}.png'.format(epoch))
    plt.show()

def display_img(epoch_num):
    return PIL.Image.open('epoch_{:04d}.png'.format(epoch_num))

def train(dataset, args):
    for epoch in range(args.epochs):
        for batch in dataset:
            fit(batch)
        display.clear_output(wait=True)
        save_img(generator,epoch+1,seed)
        args.checkpoint.save(file_prefix = args.checkpoint_prefix)
    display.clear_output(wait=True)
    save_img(generator,epoch+1,seed)
    display_img(epoch+1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_visualize', type=int, default=16)
    args = parser.parse_args()

    dataset = trainset()
    generator = generator_model()
    discriminator = discriminator_model()
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    g_optimizer = tf.keras.optimizers.Adam(1e-4)
    d_optimizer = tf.keras.optimizers.Adam(1e-4)

    args.checkpoint_prefix, args.checkpoint = ckpt(g_optimizer, 
                                                    d_optimizer, args)

    seed = tf.random.normal([args.num_visualize, 100])
    train(dataset, args.epochs)
