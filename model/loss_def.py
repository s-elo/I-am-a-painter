import tensorflow as tf

# used to determine how important the cycle loss is
LAMBDA = 10

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real, generated):
    # blur the expected label 1 to 0.9-1
    real_like_ones = tf.ones_like(real)
    smooth = tf.random.uniform([1], 0.9, 1)
    real_like_blurred = tf.math.scalar_mul(smooth[0], real_like_ones)

    # minimize the loss between real img and label 0.9-1
    real_loss = loss_obj(real_like_blurred, real)
    # minimize the loss between fake generated img and label 0
    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def generator_fool_loss(generated):
    # minimize the loss between fake generated img and label 1
    # to fool the  discriminator
    return loss_obj(tf.ones_like(generated), generated)


def cycle_loss(real_image, cycled_image, lambda_cycle=LAMBDA):
    # minimize the loss between cycled img and original img
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    # make it more important
    return lambda_cycle * loss1


def identity_loss(real_image, same_image, lambda_cycle=LAMBDA):
    # same_image == generator(real_image)
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return lambda_cycle * 0.5 * loss
