import tensorflow as tf
from model.generator import get_generator
from model.discriminator import get_discriminator
from model.loss_def import discriminator_loss, generator_fool_loss, cycle_loss, identity_loss
from model.diff_aug import aug_fn

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

    model = CycleGan(generator_g, generator_f,
                     discriminator_y, discriminator_x, 10)
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
        lambda_cycle=10,
    ):
        super(CycleGan, self).__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
        self.lambda_cycle = lambda_cycle

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
                    real_monet, same_monet, self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + \
                self.identity_loss_fn(
                    real_photo, same_photo, self.lambda_cycle)

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


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        new_val = tf.math.scalar_mul(0.7, self.model.lambda_cycle)
        val = tf.math.maximum(tf.Variable(0.0005), new_val)
        self.model.lambda_cycle.assign(val)

        new_id = tf.math.scalar_mul(0.7, self.model.lambda_id)
        new_id = tf.math.maximum(new_id, tf.Variable(0.005))
        self.model.lambda_id.assign(new_id)

    # def on_epoch_end(self, epoch, logs=None):
    #     image = self.model.m_gen(self.model.e_photo)
    #     plt.figure(figsize=(6,6))
    #     plt.title("Monet-esque Photo")
    #     plt.imshow(image[0] * 0.5 + 0.5)
    #     plt.show()
