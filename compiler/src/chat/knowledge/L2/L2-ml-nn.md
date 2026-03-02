# nn (L2)
ml/nn — Neural network primitives

## Functions
dense_forward(input: array, weights: array, bias: array) → array — FC layer
relu_forward(x: array) → array — ReLU activation
sigmoid_forward(x: array) → array — Sigmoid activation
tanh_forward(x: array) → array — Tanh activation
softmax(x: array) → array — Softmax over logits
cross_entropy_loss(pred: array, target: array) → float — CE loss
mse_loss(pred: array, target: array) → float — MSE loss
init_weights(rows: int, cols: int) → array — Random weight init
sgd_update(params: array, grads: array, lr: float) → array — SGD step
dropout(x: array, rate: float) → array — Random dropout
batch_norm(x: array, n_features: int) → array — Batch normalization
