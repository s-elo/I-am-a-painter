import tensorflow as tf
from model.generator import get_generator
from model.discriminator import get_discriminator

# define the optimizers
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


def get_model():
    # original photo -> fake monet
    generator_g = get_generator()
    # fake monet -> fake original photo
    generator_f = get_generator()

    # distinguish original photo and fake original photo
    discriminator_x = get_discriminator()
    # distinguish monet and fake monet
    discriminator_y = get_discriminator()

    print('call')
