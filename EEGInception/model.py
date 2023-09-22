import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from matplotlib import pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions





def TimeSeriesTransformer(input_length,
                          embed_dim, num_heads=2, ff_dim=32,
                          classes=1, dense_units=20, dropout_rate=0.5):
    """
    TimeSeries Transformer Classifier Instance.
    :param embed_dim: Embedding size for each token.
    :param num_heads: Number of attention heads.
    :param ff_dim: Hidden layer size in feed forward network inside transformer.
    """

    inputs = layers.Input(shape=(input_length, embed_dim))
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    # transformer_block2 = TransformerBlock(64, num_heads, ff_dim)

    x = transformer_block(inputs)
    # x = transformer_block(x)
    # x = transformer_block(x)
    # x = transformer_block(x)

    # x = transformer_block2(tf.transpose(x, perm=[0, 2, 1]))
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    if classes <= 1:
        outputs = layers.Dense(classes, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(classes, activation="softmax")(x)


    model = keras.Model(inputs=inputs, outputs=outputs)
    return model




# k-fold cross volidation

# data, labels = load_data()
#
# # define 10-fold cross validation test harness
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
# cvscores = []
# i = 1
#
# for train, test in kfold.split(data, labels):
#
#     x_train, x_test = data[train], data[test]
#     y_train, y_test = labels[train], labels[test]
#

#
#     K.clear_session()
#     reset_default_graph()
#
#     model = TimeSeriesTransformer(input_length=1250,
#                                   embed_dim=19, num_heads=1, ff_dim=32,
#                                   classes=1, dense_units=20, dropout_rate=0.5)
#     model.summary()
#
#     redce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1000, verbose=1,
#                                  mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.00001)
#
#     model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
#
#     checkpointer = ModelCheckpoint(filepath='./Model/Transformer{}.h5'.format(i), monitor='val_accuracy', verbose=1,
#                                    save_best_only=True)
#     callbacks = [redce_lr, EarlyStopping(patience=10, monitor='val_accuracy', restore_best_weights=True), checkpointer]
#
#     history = model.fit(
#         x_train, y_train, batch_size=64, epochs=1, validation_data=(x_test, y_test), callbacks=callbacks
#     )
#
#     model.load_weights('./Model/Transformer.h5')
#
#
#     # 输出最高预测准确率
#     print("The best val_accuracy:")
#     scores = model.evaluate(x_test, y_test, verbose=1)
#
#     print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
#     cvscores.append(scores[1] * 100)
#
#     # probs = model.predict(x_test)
#     preds = np.around(model.predict(x_test))
#
#     # 输出预测报告
#     print("\n模型在测试集上的预测报告：")
#     print(classification_report(y_test, preds, digits=4))
#
#     # 绘制混淆矩阵
#     print("\n真实标签与预测值的混淆矩阵：")
#     cm = confusion_matrix(y_test, preds)
#     print(cm)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot()
#     plt.show()
#     plt.savefig('confusion_matrix_fold{}'.format(i))
#
#     # plot the accuracy and loss graph
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('acc & loss')
#     plt.xlabel('epoch')
#     plt.legend(['tra_acc', 'val_acc', 'tra_loss', 'val_loss'], loc='upper right')
#     plt.show()
#     plt.savefig('acc&loss{}'.format(++i))
#
#
#
#
#
# print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))






