from tensorflow_examples.models.pix2pix import pix2pix

OUTPUT_CHANNELS = 3


def get_generator():
    # can customize later 
    return pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
