import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
from model.generator import get_generator
from model.discriminator import get_discriminator
from model.loss_def import discriminator_loss, generator_fool_loss, cycle_loss, identity_loss
from model.diff_aug import aug_fn
from model.utils import load_models

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def get_model(from_saved=False, learning_schedule=False, lr=2e-4, lambda_cycle=tf.Variable(8.0), lambda_id=tf.Variable(1.)):
    if (os.path.exists('./checkpoints') == True):
        # load from the checkpoints
        generator_g, generator_f, discriminator_y, discriminator_x = load_models(
            load_path='./checkpoints', model='all')
    elif (from_saved == False and os.path.exists('./checkpoints') == False):
        # original photo -> fake monet
        generator_g = get_generator()
        # fake monet -> fake original photo
        generator_f = get_generator()

        # distinguish original photo and fake original photo
        discriminator_x = get_discriminator()
        # distinguish monet and fake monet
        discriminator_y = get_discriminator()
    elif (from_saved == True and os.path.exists('./checkpoints') == False):
        # load from the saved_models
        generator_g, generator_f, discriminator_y, discriminator_x = load_models(
            load_path='./saved_models', model='all')

    # define the optimizers
    if (learning_schedule == False):
        generator_g_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        generator_f_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

        discriminator_x_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        discriminator_y_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
    else:
        initial_learning_rate = 1e-4
        decay_steps = 28000
        lr_schedule_g = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate, decay_steps, end_learning_rate=2e-5, power=1,
            cycle=False, name=None
        )
        lr_schedule_d = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate, decay_steps, end_learning_rate=2e-5, power=1,
            cycle=False, name=None
        )

        generator_g_optimizer = tf.keras.optimizers.Adam(
            lr_schedule_g, beta_1=0.5)
        generator_f_optimizer = tf.keras.optimizers.Adam(
            lr_schedule_g, beta_1=0.5)

        discriminator_y_optimizer = tf.keras.optimizers.Adam(
            lr_schedule_d, beta_1=0.5)
        discriminator_x_optimizer = tf.keras.optimizers.Adam(
            lr_schedule_d, beta_1=0.5)

    model = CycleGan(generator_g, generator_f,
                     discriminator_y, discriminator_x, lambda_cycle, lambda_id)
    model.compile(m_gen_optimizer=generator_g_optimizer,
                  p_gen_optimizer=generator_f_optimizer,
                  m_disc_optimizer=discriminator_y_optimizer,
                  p_disc_optimizer=discriminator_x_optimizer,
                  gen_loss_fn=generator_fool_loss,
                  disc_loss_fn=discriminator_loss,
                  cycle_loss_fn=cycle_loss,
                  identity_loss_fn=identity_loss)
    return model


class CycleGan(tf.keras.Model):
    def __init__(
        self,
        monet_generator,
        photo_generator,
        monet_discriminator,
        photo_discriminator,
        lambda_cycle=tf.Variable(8.0),
        # lambda_cycle=tf.Variable(0.11073),
        lambda_id=tf.Variable(1.)
        # lambda_id=tf.Variable(0.01384)
    ):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle
        self.lambda_id = lambda_id

    def compile(
        self,
        m_gen_optimizer,
        p_gen_optimizer,
        m_disc_optimizer,
        p_disc_optimizer,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        super(CycleGan, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn

        self.save_checkpoints_path = './checkpoints'

    def train_step(self, batch_data):
        real_monet, real_photo = batch_data
        batch_size = tf.shape(real_monet)[0]

        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # Diffaugment monet and photo
            both_monet = tf.concat([real_monet, fake_monet], axis=0)
            aug_monet = aug_fn(both_monet)
            aug_real_monet = aug_monet[:batch_size]
            aug_fake_monet = aug_monet[batch_size:]

            both_photo = tf.concat([real_photo, fake_photo], axis=0)
            aug_photo = aug_fn(both_photo)
            aug_real_photo = aug_photo[:batch_size]
            aug_fake_photo = aug_photo[batch_size:]

            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(aug_real_monet, training=True)
            disc_real_photo = self.p_disc(aug_real_photo, training=True)

            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(aug_fake_monet, training=True)
            disc_fake_photo = self.p_disc(aug_fake_photo, training=True)

            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(
                real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + \
                self.identity_loss_fn(
                    real_monet, same_monet, self.lambda_id)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + \
                self.identity_loss_fn(
                    real_photo, same_photo, self.lambda_id)

            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(
                disc_real_monet, disc_fake_monet)
            photo_disc_loss = self.disc_loss_fn(
                disc_real_photo, disc_fake_photo)

        # Calculate the gradients for generator and discriminator
        monet_generator_gradients = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        monet_discriminator_gradients = tape.gradient(monet_disc_loss,
                                                      self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,
                                                 self.m_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,
                                                  self.m_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))

        return {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss
        }

    def save(self, save_path='./saved_models'):
        # if it is not saving checkpoints
        if (save_path != self.save_checkpoints_path):
            # remove the checkpoints
            shutil.rmtree(self.save_checkpoints_path)

        # store the model for later use
        if os.path.exists(save_path) == True:
            shutil.rmtree(save_path)
        os.mkdir(save_path)

        self.m_gen.save(f'{save_path}/m_gen')
        self.p_gen.save(f'{save_path}/p_gen')
        self.m_disc.save(f'{save_path}/m_disc')
        self.p_disc.save(f'{save_path}/p_disc')


class CustomCallback(tf.keras.callbacks.Callback):
    # decay the lambda_cycle and lambda_id after each epoch
    def on_epoch_begin(self, epoch, logs=None):
        new_val = tf.math.scalar_mul(0.7, self.model.lambda_cycle)
        # minimum 0.0005
        val = tf.math.maximum(tf.Variable(0.0005), new_val)
        self.model.lambda_cycle.assign(val)

        new_id = tf.math.scalar_mul(0.7, self.model.lambda_id)
        # minimum 0.005
        new_id = tf.math.maximum(new_id, tf.Variable(0.005))
        self.model.lambda_id.assign(new_id)

    # just for showing the result after each epoch and save the checkpoints
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(save_path=self.model.save_checkpoints_path)
        # image = self.model.m_gen(self.model.e_photo)
        # plt.figure(figsize=(6, 6))
        # plt.title("Monet-esque Photo")
        # plt.imshow(image[0] * 0.5 + 0.5)
        # plt.show()
