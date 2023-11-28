# Jude Tear
# Friday November 17th 2023
# CISC 471: UNet

'''
About; UNet

Unet is an architecture for a fully convolutional neural network that specializes in image segmentation
also called semantic segmentation. It can predict where an object is on the picture, but also isable to 
create a mask that shows where on the image that specifc object is located

- fully convolutional
- does not contain any dense layers 
- fully connected layers
- single class segmentation (can be changed for multiple)
    - requires ground truths


encoder
- responsible for the what part
- image gets smaller and smaller but number of channels gets bigger and bigger

decoder
- responsible for the where part
- upsample the data, the data gets bigger and bigger but the number of channels half


Variation of classifcation: every pixel in an image gets assigned to a class that it belongs to 
which then kind of forms the mask
'''



import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input
from keras import layers

def conv_block(input_tensor, num_filters):
    """A convolutional block consisting of 2 convolution layers followed by a max-pooling layer"""
    x = Conv2D(num_filters, (3, 3), padding="same", activation="relu")(input_tensor)
    x = Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
    return x

def encoder_block(input_tensor, num_filters):
    """An encoder block (convolutions + max-pooling)"""
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, concat_tensor, num_filters):
    """A decoder block (upsampling, concatenation, convolutions)"""
    x = UpSampling2D((2, 2))(input_tensor)
    x = Concatenate()([x, concat_tensor])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, num_filters, num_classes):
    inputs = layers.Input(input_shape)

    # Encoder
    x1, p1 = encoder_block(inputs, num_filters)    # 128 -> 64
    x2, p2 = encoder_block(p1, num_filters * 2)    # 64 -> 32
    x3, p3 = encoder_block(p2, num_filters * 4)    # 32 -> 16

    # Bottleneck
    b = conv_block(p3, num_filters * 8)            # 16

    # Decoder
    x3 = decoder_block(b, x3, num_filters * 4)     # 16 -> 32
    x2 = decoder_block(x3, x2, num_filters * 2)    # 32 -> 64
    x1 = decoder_block(x2, x1, num_filters)        # 64 -> 128

   
    # Output layer 
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(x1)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Define model parameters
input_shape = (128, 128, 1)  # Example input shape, modify as needed
num_filters = 64             # Base number of convolutional filters
num_classes = 1              # Number of classes (1 for binary segmentation)

# Build and compile the model
unet_model = build_unet(input_shape, num_filters, num_classes)
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# unet_model.summary()  # Uncomment to see the model architecture




