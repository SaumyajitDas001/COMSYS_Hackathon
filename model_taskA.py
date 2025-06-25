
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

def create_model(model_type="cnn"):
    if model_type == "cnn":
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
    elif model_type == "mobilenetv2":
        base = MobileNetV2(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
        base.trainable = False
        x = GlobalAveragePooling2D()(base.output)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base.input, outputs=output)
    elif model_type == "both":
        # Default to MobileNetV2 with fallback to CNN if no internet
        try:
            return create_model("mobilenetv2")
        except:
            return create_model("cnn")
    else:
        raise ValueError("Unknown model_type")
    return model
