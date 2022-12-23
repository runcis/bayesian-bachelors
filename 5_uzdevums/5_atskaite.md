# Task 5.1: Apmācības modelis mašīnu cenu pareģošanai

Interesants uzdevums, ilgu laiku nočakarēju, līdz sapratu, ka man atsķirās datu kopas input features un output data (man bija 6 nevis 7 inputi un 2 nevis 1 output - t.i netikai vērtība tiek izvadīta.) Bet tas, man lika izrakņāties vairāk cauri kodam, kas likās noderīgi, lai labāk saprastu algoritmu.

Novēroju, ka dažādiem modeļu tipiem tiek izveidoti citādi pielāgošanās grafiki. Bet īsti nesaprotu kā interperetēt šos grafikus.

Eksperimentēju ar slāņu daudzumiem un neironu skaitiem:

Mazāk neironi un slāņi:
![model 3 layers less neurons](media/model_3_layers_less_neurons_0.PNG)


![model 3 layers less neurons](media/model_3_layers_less_neurons.PNG)

Vairāk neironu un slāņi:
![model 4 layers more neurons](media/model_4_layers_more_neurons_0.PNG)


![model 4 layers more neurons](media/model_4_layers_more_neurons.PNG)

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

![model mse local min](media/model_mse_local_min.PNG)

2. ReLU