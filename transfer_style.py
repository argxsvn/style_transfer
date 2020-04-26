"""
Transfers style from the specified style image onto the specified content image.
"""

import argparse
import os
import pickle
import tensorflow as tf

from gram import GramLayer


def main(vgg_model_filename, content_image, style_image, output_image, preserve_color, content_weight,
         variation_weight, color_mix, max_iters, init_random):

    with open(vgg_model_filename, "rb") as f_in:
        model_json = pickle.load(f_in)

    model = tf.keras.models.model_from_json(model_json, custom_objects={"GramLayer": GramLayer})

    print(model.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("vgg_model_filename", help="Path to the file storing the saved VGG16 model to use")
    parser.add_argument("content_image", help="Filename of the image to use for content features")
    parser.add_argument("style_image", help="Filename of the image to use for style features")
    parser.add_argument("output_image", help="Filename of the style-transferred image to output")

    parser.add_argument("-c", action="store_true", help="Specify to preserve content image colors")
    parser.add_argument("--rand", action="store_true", help="Specify to initialize from random noise instead of content")

    parser.add_argument("--content_weight", type=float, default=0.99, help="Weight of content loss between 0-1 (Default: 0.99)")
    parser.add_argument("--variation_weight", type=float, default=0.01, help="Weight of image variation loss (Default: 0.01)")

    parser.add_argument("--color_mix", type=float, default=0.6, help="Weight of content color if color preserved, between 0-1 (Default: 0.6)")

    parser.add_argument("--max_iters", type=int, default=100, help="Maximum number of L-BFGS iterations (Default: 100)")

    args = parser.parse_args()
    main(os.path.abspath(args.vgg_model_filename), os.path.abspath(args.content_image), os.path.abspath(args.style_image),
         os.path.abspath(args.output_image), args.c, args.content_weight, args.variation_weight, args.color_mix,
         args.max_iters, args.rand)

