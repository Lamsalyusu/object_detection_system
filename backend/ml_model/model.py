"""
This file builds our object detection model.
It looks at a photo and finds 10 objects: sunglasses, knife, water bottle,
pen, chair, human face, mobile phone, helmet, fire, can.
  Step 1 - BACKBONE : Reads the image, shrinks it, finds patterns (edges, shapes)
  Step 2 - NECK(FPN): Combines what it found at different zoom levels
  Step 3 - HEAD     : Says "there is a knife at this location with 95% confidence"
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 640      # every image is resized to 640x640 before entering the model
NUM_CLASSES = 10    # we detect 10 types of objects


# FUNCTION 1: conv_block
# The basic building block used everywhere in this model.
# Does 3 things: scan for patterns → stabilize → remove negatives
def conv_block(x, filters, kernel=3, strides=1, name=None):

    # Conv2D: scans the image with a small filter to find patterns like edges and shapes
    # filters = how many patterns to look for
    # kernel  = size of the scanning window (3 means 3x3 area)
    # strides = how many pixels to jump each step (strides=2 makes image half the size)
    # padding='same' keeps the output the same size as input
    # use_bias=False because BatchNorm below already handles this
    x = layers.Conv2D(filters, kernel, strides=strides, padding='same',
                      use_bias=False, name=f"{name}_conv" if name else None)(x)

    # BatchNormalization: keeps all values in a stable range so training doesn't go crazy
    x = layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)

    # ReLU: turns all negative values to 0, keeps positive ones
    # without this the model can only learn straight-line relationships
    x = layers.ReLU(name=f"{name}_relu" if name else None)(x)

    return x

# FUNCTION 2: csp_block
# Splits data into 2 paths, processes separately, then merges.
# Prevents the "vanishing gradient" problem in deep networks
# (gradients = the learning signal that travels backward during training)

def csp_block(x, filters, name):

    split_filters = filters // 2    # each path gets half the filters

    # PATH 1 (shortcut): just compresses channels, no deep processing
    # this gives gradients a fast highway to travel backward = better learning
    path1 = layers.Conv2D(split_filters, 1, padding='same', use_bias=False,
                          name=f"{name}_path1")(x)
    path1 = layers.BatchNormalization(name=f"{name}_bn1")(path1)
    path1 = layers.ReLU(name=f"{name}_relu1")(path1)

    # PATH 2 (processing): does the actual deep learning
    # first conv: compress channels
    path2 = layers.Conv2D(split_filters, 1, padding='same', use_bias=False,
                          name=f"{name}_path2_conv1")(x)
    path2 = layers.BatchNormalization(name=f"{name}_bn2")(path2)
    path2 = layers.ReLU(name=f"{name}_relu2")(path2)

    # second conv: learn spatial patterns by looking at 3x3 neighborhood
    path2 = layers.Conv2D(split_filters, 3, padding='same', use_bias=False,
                          name=f"{name}_path2_conv2")(path2)
    path2 = layers.BatchNormalization(name=f"{name}_bn3")(path2)
    path2 = layers.ReLU(name=f"{name}_relu3")(path2)

    # Concatenate: stack path1 and path2 together side by side
    out = layers.Concatenate(name=f"{name}_concat")([path1, path2])

    # final conv: blend both paths into one clean output
    out = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                        name=f"{name}_transition")(out)
    out = layers.BatchNormalization(name=f"{name}_bn_out")(out)
    out = layers.ReLU(name=f"{name}_relu_out")(out)

    return out

# FUNCTION 3: sppf_block
# Applies max pooling 3 times in a row, then stacks all results.
# Lets the model see both tiny details and wide context at the same time.
# (MaxPooling2D: looks at a region and keeps only the strongest value)
def sppf_block(x, filters, name):

    # compress channels before pooling
    conv1 = layers.Conv2D(filters // 2, 1, padding='same', name=f"{name}_conv1")(x)
    conv1 = layers.BatchNormalization(name=f"{name}_bn1")(conv1)
    conv1 = layers.ReLU(name=f"{name}_relu1")(conv1)

    # pool1: looks at 5x5 area → small context
    pool1 = layers.MaxPooling2D(5, strides=1, padding='same', name=f"{name}_pool1")(conv1)
    # pool2: pools on top of pool1 → medium context (effectively 9x9 area)
    pool2 = layers.MaxPooling2D(5, strides=1, padding='same', name=f"{name}_pool2")(pool1)
    # pool3: pools on top of pool2 → wide context (effectively 13x13 area)
    pool3 = layers.MaxPooling2D(5, strides=1, padding='same', name=f"{name}_pool3")(pool2)

    # stack original + all 3 pooled versions together
    concat = layers.Concatenate(name=f"{name}_concat")([conv1, pool1, pool2, pool3])

    # compress the stacked result back to the right number of filters
    out = layers.Conv2D(filters, 1, padding='same', name=f"{name}_conv2")(concat)
    out = layers.BatchNormalization(name=f"{name}_bn2")(out)
    out = layers.ReLU(name=f"{name}_relu2")(out)

    return out

# MAIN FUNCTION: build_model
# Puts everything together into the full detection network
def build_model():

    # define the input shape: one image, 640x640 pixels, 3 color channels (RGB)
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")


    # BACKBONE: shrink the image step by step while finding patterns
    # 640 → 320 → 160 → 80 → 40 → 20
    # smaller size = each "cell" sees a bigger area of the original image
    # stem: quickly shrink 640x640 → 320x320 → 160x160
    x = conv_block(inputs, 32, kernel=3, strides=2, name="stem1")   # 640 → 320
    x = conv_block(x, 64, kernel=3, strides=2, name="stem2")        # 320 → 160

    # learn patterns at 160x160 scale
    x = csp_block(x, 64, name="stage1")

    # shrink to 80x80
    x = conv_block(x, 128, kernel=3, strides=2, name="down1")       # 160 → 80

    # learn patterns at 80x80 — this scale is best for SMALL objects (pen, knife)
    x = csp_block(x, 128, name="stage2a")
    x = csp_block(x, 128, name="stage2b")
    small_features = x      # save 80x80 features for FPN later

    # shrink to 40x40
    x = conv_block(x, 256, kernel=3, strides=2, name="down2")       # 80 → 40

    # learn patterns at 40x40 — this scale is best for MEDIUM objects (bottle, phone)
    x = csp_block(x, 256, name="stage3a")
    x = csp_block(x, 256, name="stage3b")
    medium_features = x     # save 40x40 features for FPN later

    # shrink to 20x20
    x = conv_block(x, 512, kernel=3, strides=2, name="down3")       # 40 → 20

    # learn patterns at 20x20 — this scale is best for LARGE objects (chair, face)
    x = csp_block(x, 512, name="stage4a")
    x = csp_block(x, 512, name="stage4b")

    # SPPF: give the 20x20 map awareness of both small and large areas at once
    x = sppf_block(x, 512, name="sppf")
    large_features = x      # save 20x20 features for FPN later

    # NECK (FPN): merge the 3 saved feature maps top-down
    # deep features (20x20) know WHAT the object is
    # shallow features (80x80) know WHERE the object is
    # FPN combines them so every scale knows both WHAT and WHERE
   
    # P5: compress large features (20x20), then upsample to 40x40
    p5 = conv_block(large_features, 256, kernel=1, name="neck_p5")
    p5_up = layers.UpSampling2D(2, name="p5_upsample")(p5)   # 20x20 → 40x40
    # UpSampling2D: makes the feature map bigger by repeating each value

    # P4: merge medium features with upsampled P5
    # now P4 knows both medium-scale location AND high-level semantics
    p4 = conv_block(medium_features, 256, kernel=1, name="neck_p4_reduce")
    p4 = layers.Concatenate(name="p4_concat")([p4, p5_up])   # stack them together
    p4 = csp_block(p4, 256, name="neck_p4")                  # blend them together
    p4_up = layers.UpSampling2D(2, name="p4_upsample")(p4)   # 40x40 → 80x80

    # P3: merge small features with upsampled P4
    # now P3 knows fine-scale location AND high-level semantics
    p3 = conv_block(small_features, 256, kernel=1, name="neck_p3_reduce")
    p3 = layers.Concatenate(name="p3_concat")([p3, p4_up])   # stack them together
    p3 = csp_block(p3, 256, name="neck_p3")                  # blend them together


    # HEAD: make the final predictions at each scale
    # each grid cell predicts 45 values:
    #   3 anchors x (4 box coords + 1 confidence + 10 class scores) = 45
    #   anchor = a pre-defined box shape the model adjusts from

    num_anchors = 3
    out_channels = num_anchors * (5 + NUM_CLASSES)  # 3 x 15 = 45

    # small object head: uses P3 (80x80 grid)
    # 80x80 = 6400 cells → fine grid → catches small objects
    head_s = conv_block(p3, 256, kernel=3, name="head_s")
    head_s = layers.Dropout(0.2, name="head_s_dropout")(head_s)  # randomly turn off 20% of neurons to prevent overfitting
    head_s = layers.Conv2D(out_channels, 1, name="detect_s")(head_s)  # output 45 values per cell

    # medium object head: uses P4 (40x40 grid)
    # 40x40 = 1600 cells  catches medium objects
    head_m = conv_block(p4, 256, kernel=3, name="head_m")
    head_m = layers.Dropout(0.2, name="head_m_dropout")(head_m)
    head_m = layers.Conv2D(out_channels, 1, name="detect_m")(head_m)

    # large object head: uses P5 (20x20 grid)
    # 20x20 = 400 cells → coarse grid → catches large objects
    head_l = conv_block(p5, 256, kernel=3, name="head_l")
    head_l = layers.Dropout(0.2, name="head_l_dropout")(head_l)
    head_l = layers.Conv2D(out_channels, 1, name="detect_l")(head_l)

    # return the complete model
    # input  → one 640x640 image
    # output → 3 grids: [80x80x45, 40x40x45, 20x20x45]
    #          total 25,200 box predictions → filtered to best ones
    return keras.Model(inputs, [head_s, head_m, head_l], name="SmartVision_YOLO")


if __name__ == "__main__":
    m = build_model()
    m.summary()