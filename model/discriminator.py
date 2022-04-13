from tensorflow_examples.models.pix2pix import pix2pix

OUTPUT_CHANNELS = 3


def get_discriminator():
    # can customize later 
    return pix2pix.discriminator(norm_type='instancenorm', target=False)
