# LSTM testing

## Notes:

**[LSTM Layer](https://keras.io/api/layers/recurrent_layers/lstm/) in Keras**

```python
tf.keras.layers.LSTM(
    units,
    activation="tanh",
    recurrent_activation="sigmoid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    implementation=2,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    time_major=False,
    unroll=False,
    **kwargs
)
```

- `units` : Dimensionality of the output space. Refers to the the size of the internal state of the LSTM layer. Kind of like the unit in a hidden layer 
- `return_sequences` : If true the LSTM layer will return an output vector with `units` elements. If stacking LSTM layers usually the first and inners ones return the full sequence and the last returns only the last element.
- `stateful` : 

The `input_shape` of the LSTM is (Tx, nx) where TX is the lengths of the input sequence and nx is the number of features. If we only predict from a single sequence the nx would be 1 (also called univariant). If e.g. we measrue two quantities over time we would use nx = 2.

## Reference:

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Nice images
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### Some interesting stackoverflow discussions:

- [Understanding Keras LSTMs](https://stackoverflow.com/questions/38714959/understanding-keras-lstms/50235563#50235563)
- [Followup to above](https://stackoverflow.com/questions/53955093/doubts-regarding-understanding-keras-lstms)
