# Task 6_2: Pytorch regression
Uzdevuma jēga: Iepazīt Pytorch, Softmax

Embedded matrix:

Konceptuāli liekas ka sapratu, detaļas implementācijā nav īsti skaidras.

~~~
elf.embs = torch.nn.ModuleList()
        for i in range(4): # brand, fuel, transmission, dealership
            self.embs.append(
                torch.nn.Embedding(
                    num_embeddings=len(dataset_full.labels[i]),
                    embedding_dim=3
                )
            )

    def forward(self, x, x_classes):
        x_emb_list = []
        for i, emb in enumerate(self.embs):
            x_emb_list.append(
                emb.forward(x_classes[:, i])
            )
        x_emb = torch.cat(x_emb_list, dim=-1)
        x_cat = torch.cat([x, x_emb], dim=-1)
        y_prim = self.layers.forward(x_cat)
        return y_prim
~~~

Loss Huber:

~~~
class LossHuber(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, y_prim, y):
        return torch.mean(self.delta**2 * (torch.sqrt(1 + ((y - y_prim)/self.delta) ** 2) - 1))
~~~

Modeļa rezultāts:

![6 uzd model result](media/6-uzd-model-result.PNG))


# Task 6_3: Pytorch classification

Veidojot softmax funkciju, nesanāca to patstāvīgi izdarīt.

Forward: Kāpēc vajag iegūt np.max no ievaddatiem? Mans risinājums arī iegūst summā 1 katram no 16 ierakstiem.

Backward: Nesapratu else daļu - liekas ka a[:,row] * a[:, column] īsti neizpilda formulēto rezultātu?
*Baigi labs pieraksts, nebutu pats izdomājis, jo vel nedomāju tik labi par for loopiem iekš matricām.

Softmax:
~~~
class LayerSoftmax():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x):
        self.x = x

        exp_array = np.exp(x.value)

        self.output = Variable(
            np.exp(x.value) / np.sum(exp_array, axis=-1, keepdims=True)
        )
        return self.output

    def backward(self):
        size = self.x.value.shape[-1]
        J = np.zeros((BATCH_SIZE, size, size))
        a = self.output.value

        result = np.zeros((BATCH_SIZE, size))
        
        for row in range(size):
            for column in range(size):
                if row == column:
                    J[:, row, column] = a[:,row] * (1 - a[:, column])
                else: 
                    J[:, row, column] = a[:,row] * a[:, column]

        self.x.grad += np.squeeze(J @ result[:,:,np.newaxis], axis=-1)
~~~