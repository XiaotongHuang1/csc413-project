import tensorflow as tf
from tensorflow.keras import layers

def create_transformer_model(input_dim, num_heads, ff_dim, num_layers):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(ff_dim, activation="relu")(inputs)
    for _ in range(num_layers):
        # Normalization
        x = layers.LayerNormalization()(x)
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        # Feed-forward network
        ff_output = layers.Dense(ff_dim, activation="relu")(x)
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

input_dim = 6440  # 25 * 256 for headlines + 30 for prices

num_heads = 2  # Adjust based on your preference
ff_dim = 256  # Adjust the feed-forward layer dimension if needed
num_layers = 2  # Number of transformer layers

model = create_transformer_model(input_dim, num_heads, ff_dim, num_layers)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10)


test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_acc}")
