# Task 5.1: Apmācības modelis mašīnu cenu pareģošanai

Interesants uzdevums, ilgu laiku nočakarēju, līdz sapratu, ka man atsķirās datu kopas input features un output data (man bija 6 nevis 7 inputi un 2 nevis 1 output - t.i netikai vērtība tiek izvadīta.) Bet tas, man lika izrakņāties vairāk cauri kodam, kas likās noderīgi, lai labāk saprastu algoritmu.

Novēroju, ka dažādiem modeļu tipiem tiek izveidoti citādi pielāgošanās grafiki. Bet īsti nesaprotu kā interperetēt šos grafikus.

Eksperimentēju ar slāņu daudzumiem un neironu skaitiem:

Mazāk neironi un slāņi:
![model 3 layers less neurons](./media/model_3_layers_less_neurons_0.PNG)


![model 3 layers less neurons](./media/model_3_layers_less_neurons.PNG)

Vairāk neironu un slāņi:
![model 4 layers more neurons](./media/model_4_layers_more_neurons_0.PNG)


![model 4 layers more neurons](./media/model_4_layers_more_neurons.PNG)

Kāpēc pēdējā bildē ir tik dīvaina train loss līkne?

### List of implemented functions

1. MSE/L2 loss function

~~
class LossMSE():
    def __init__(self):
        self.y = None
        self.y_prim  = None

    def forward(self, y: Variable, y_prim: Variable):
        self.y = y
        self.y_prim = y_prim
        loss = np.mean(np.sum((y.value - y_prim.value)**2))
        return loss

    def backward(self):
        self.y_prim.grad += -2*(self.y.value - self.y_prim.value)
~~

Izmantojot šo kļūdas funkciju, tiek iegūts dīvains rezultāts, es pieņemu, ka tas ir ātra nokļūšana pie lokālā minimuma un nespēj no tā izkāpt?

![model mse local min](./media/model_mse_local_min.PNG)

2. ReLU implemetation

Nēesmu drošs vai ir parezi, it īpaši par backwards algoritmu.

~~~
class LayerRelu():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        temp = self.x.value
        temp[temp<0]=0
        self.output = Variable( temp )
        return self.output

    def backward(self):
        temp = self.output.value
        temp[temp<0]=0
        temp[temp>0]=1
        self.x.grad += temp * self.output.grad
~~~

Rezultāti izmantojot Relu:

Ar MSE:
![relu mse model](./media/relu_mse_model.PNG

Ar MAE:
![relu mae model](./media/relu_mae_model.PNG))


3. NRMSE

Nestrādā - nekas naparādās un grafika.

~~~
def calculateNRMSE(y, y_prim):
    rmse = np.sqrt(np.mean(np.sum((y_prim - y)**2)))
    result = rmse/np.std(y)
    return result
~~~

4. Swish funkcija
Pēc rezultāti, liekas ka nav pareizi implementēts.
~~~
class LayerSwish():
    def __init__(self):
        self.x = None
        self.output = None

    def forward(self, x: Variable):
        self.x = x
        self.output = Variable(x.value / (1.0 + np.exp(-x.value)))
        return self.output

    def backward(self):
        self.x.grad += self.output.value + np.std(x) * (1.0 - self.output.value) 
~~~

Rezultāts:

![swish model](./media/swish_model.PNG))