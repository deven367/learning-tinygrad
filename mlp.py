from tinygrad import Tensor, nn
from tinygrad.nn.datasets import mnist

X_train, Y_train, X_test, Y_test =  mnist()

print(X_train.shape)

TYPE = 1

class Model:
    def __init__(self):
        if TYPE == 1:
            self.mlp1 = nn.Linear(784, 100)
            self.mlp2 = nn.Linear(100, 100)
            self.mlp3 = nn.Linear(100, 10)
        else:
            self.mlp1 = nn.Linear(784, 400)
            self.mlp2 = nn.Linear(400, 10)

    def __call__(self, x: Tensor):
        if TYPE == 1:
            x = x.reshape(x.shape[0], -1)
            y = self.mlp1(x).relu()
            y = self.mlp2(y).relu()
            y = self.mlp3(y)
            return y
        else:
            x = x.reshape(x.shape[0], -1)
            y = self.mlp1(x).relu()
            y = self.mlp2(y).relu()
            return y

m = Model()
optim = nn.optim.Adam(nn.state.get_parameters(m), lr=1e-3)

# print(nn.state.get_parameters(m))

STEPS = 100
BATCH_SIZE = 256

Tensor.training = True

for i in range(STEPS):
    samp = Tensor.randint(BATCH_SIZE, high=X_train.shape[0])
    x = X_train[samp]
    y = Y_train[samp]

    logits = m(x)
    loss = logits.cross_entropy(y)
    # ce = logits.sparse_categorical_crossentropy

    optim.zero_grad()
    loss.backward()
    optim.step()

    if i % 10 == 0:
        s2 = Tensor.randint(BATCH_SIZE, high=X_test.shape[0])
        acc = (m(X_test[s2]).argmax(axis=1) == Y_test[s2]).mean().item()
        print(f"Epoch {i}: loss {loss.item():.4f} : Acc {acc:.4f}")
