class model_config:
    def __init__(self):
        self.model_name = None
        self.input_channels = 3
        self.num_classes = 1
        self.image_height = None
        self.image_width = None
        self.shift_bbox = 10
        self.device = None

class segformer_config(model_config):
    def __init__(self):
        super().__init__()
        self.model_name = "segformer"
        self.image_height = 256
        self.image_width = 256

class unet_config(model_config):
    def __init__(self):
        super().__init__()
        self.model_name = "unet"
        self.image_height = 256
        self.image_width = 256

class swinunet_config(model_config):
    def __init__(self):
        super().__init__()
        self.model_name = "swinunet"
        self.image_height = 224
        self.image_width = 224

class transunet_config(model_config):
    def __init__(self):
        super().__init__()
        self.model_name = "transunet"
        self.image_height = 256
        self.image_width = 256

