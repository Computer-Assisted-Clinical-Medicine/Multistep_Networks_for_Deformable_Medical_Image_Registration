import tensorflow.keras.layers as KL
from tensorflow.keras import layers
import voxelmorph as vxm
from voxelmorph import layers
# TODO: change full module imports as opposed to specific function imports
from voxelmorph.tf.modelio import LoadableModel, store_config_args

from NetworkBasis.layers import *

# make ModelCheckpointParallel directly available from vxm
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel

class Network(LoadableModel):
    """
    Neural Network for (unsupervised) registration between two images.
    MultistepA: 4 Steps/Resolutions, 1/8 1/4 1/2 1
    MultistepB: 3 Steps/Resolutions, 1/4 1/2 1
    MultistepC: 2 Steps/Resolutions, 1/2 1
    benchmark: 1 Step/Resolution, 1
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_features, architecture, variant):
        """
        Parameters:
            inshape: Input shape. e.g. (256,245,64)
            nb_features: encoder and decoder
            variant: steps/networks used
        """
        # configure unet input shape (concatenation of moving and fixed images)
        src_feats = 1
        trg_feats = 1

        if architecture == "MultistepA":
            down_factor=[8,4,2,1]
        elif architecture == "MultistepB":
            down_factor=[4,2,1]
        elif architecture == "MultistepC":
            down_factor=[2,1]
        elif architecture == "Benchmark":
            down_factor=[1]

        nb_steps=len(variant)

        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        moved_images = []
        disp_fields = []

        source_down=ne.layers.Resize((1/down_factor[0]))(source)
        target_down=ne.layers.Resize((1/down_factor[0]))(target)

        output = vxm.networks.VxmDense(inshape=source_down.shape[1:-1], nb_unet_features=nb_features, int_downsize=1)([source_down, target_down])
        moved_images.append(tf.cast(output[0], 'float32'))
        disp_fields.append(tf.cast(output[1], 'float32'))
        disp_sum = tf.cast(layers.RescaleTransform(down_factor[0])(output[1]), 'float32')

        if nb_steps > 1:
            target_down = ne.layers.Resize((1 / down_factor[1]))(target)
            moved_image = vxm.layers.SpatialTransformer(name='transformer1')([source, disp_sum])
            moved_image_down = ne.layers.Resize((1 / down_factor[1]))(moved_image)

            output = vxm.networks.VxmDense(inshape=moved_image_down.shape[1:-1], nb_unet_features=nb_features, int_downsize=1, name = '1')([moved_image_down, target_down])
            moved_images.append(tf.cast(output[0], 'float32'))
            disp_fields.append(tf.cast(output[1], 'float32'))
            disp_sum += tf.cast(layers.RescaleTransform(down_factor[1])(output[1]), 'float32')

        if nb_steps > 2:
            target_down = ne.layers.Resize((1 / down_factor[2]))(target)
            moved_image = vxm.layers.SpatialTransformer(name='transformer2')([source, disp_sum])
            moved_image_down = ne.layers.Resize((1 / down_factor[2]))(moved_image)

            output = vxm.networks.VxmDense(inshape=moved_image_down.shape[1:-1], nb_unet_features=nb_features, int_downsize=1, name = '2')([moved_image_down, target_down])
            moved_images.append(tf.cast(output[0], 'float32'))
            disp_fields.append(tf.cast(output[1], 'float32'))
            disp_sum += tf.cast(layers.RescaleTransform(down_factor[2])(output[1]), 'float32')

        if nb_steps > 3:
            moved_image = vxm.layers.SpatialTransformer(name='transformer3')([source, disp_sum])

            output = vxm.networks.VxmDense(inshape=moved_image.shape[1:-1], nb_unet_features=nb_features, int_downsize=1, name = '3')([moved_image, target])
            moved_images.append(tf.cast(output[0], 'float32'))
            disp_fields.append(tf.cast(output[1], 'float32'))
            disp_sum += tf.cast(output[1], 'float32')

        moved_image = vxm.layers.SpatialTransformer(name='transformer_final')([source, disp_sum])

        outputs = [moved_image, disp_sum]

        super().__init__(name='Network', inputs=input_model.outputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        # self.references.net_model = net
        self.references.y_source = moved_image
        self.references.pos_flow = disp_sum

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])#