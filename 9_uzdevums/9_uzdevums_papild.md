### 9.uzdevuma MD:

Veidojot pārbaudi anomāliju detektēšanai, sanāca interesanti rezultāti.

1. Vai nevajadzētu kkā izmainīt kļūdas funkciju?

Sākumā uztaisiju daudz lielāku encoding - decoding algoritmu ar daudziem slāņiem:

~~~
class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),
            torch.nn.Mish(),
            
            torch.nn.Upsample(size=80),

            torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=8, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=12, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=60),

            torch.nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=16, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=16, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=20, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=40),

            torch.nn.Conv2d(in_channels=20, out_channels=24, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=24, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=32, num_groups=2),
            torch.nn.Mish(),
            
            torch.nn.Upsample(size=20),

            torch.nn.Conv2d(in_channels=32, out_channels=40, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=40, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=40, out_channels=48, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=48, num_groups=2),
            torch.nn.Mish(),
            
            torch.nn.Upsample(size=10),

            torch.nn.Conv2d(in_channels=48, out_channels=56, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=56, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=56, out_channels=60, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=60, num_groups=2),
            torch.nn.Mish(),
            
            torch.nn.Upsample(size=3),

            torch.nn.Conv2d(in_channels=60, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=64, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=1),

            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=64, num_groups=2),

            torch.nn.Tanh()
        )

        self.decoder = torch.nn.Sequential(
            
            torch.nn.Conv2d(in_channels=64, out_channels=60, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=60, num_groups=2),
            torch.nn.Mish(),
            
            torch.nn.Upsample(size=2),

            torch.nn.Conv2d(in_channels=60, out_channels=56, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=56, num_groups=2),
            torch.nn.Mish(),
            
            torch.nn.Upsample(size=5),

            torch.nn.Conv2d(in_channels=56, out_channels=48, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=48, num_groups=2),
            torch.nn.Mish(),
            
            torch.nn.Conv2d(in_channels=48, out_channels=40, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=40, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=40, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=32, num_groups=2),
            torch.nn.Mish(),
            
            torch.nn.Upsample(size=10),

            torch.nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=24, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=24, out_channels=20, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=20, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=20),

            torch.nn.Conv2d(in_channels=20, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=16, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=12, num_groups=2),
            torch.nn.Mish(),


            torch.nn.Upsample(size=40),

            torch.nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=8, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=6, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=80),

            torch.nn.Conv2d(in_channels=6, out_channels=4, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Upsample(size=100),  # 10x10>100

            torch.nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=4, num_groups=2),
            torch.nn.Mish(),

            torch.nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=1, padding=1),
            torch.nn.GroupNorm(num_channels=3, num_groups=1),
            torch.nn.Tanh()

        )

    def forward(self, x):
        z = self.encoder.forward(x)
        z = z.view(-1, 64)
        y_prim = self.decoder.forward(z.view(-1, 64, 1, 1))
        return y_prim, z
~~~

Tikai testa datusetam pievienoju 5 kaķu bildes:

![cat_1.jpg](..%2Fdata%2F9_5_anomalies%2Fcat_1.jpg)
![cat_2.jpg](..%2Fdata%2F9_5_anomalies%2Fcat_2.jpg)
![cat_3.jpg](..%2Fdata%2F9_5_anomalies%2Fcat_3.jpg)
![cat_4.jpg](..%2Fdata%2F9_5_anomalies%2Fcat_4.jpg)
![cat_5.jpg](..%2Fdata%2F9_5_anomalies%2Fcat_5.jpg)

Man nesanāca izdomāt elegantu veidu, kā pievienot bildes jaunam datusetam, negribēju arī laiku veltīt lai satīrītu kodu, 
līdzko sakā strādāt:

~~~
class DatasetAnomalies(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        img1 = np.array([])
        img2 = np.array([])
        img3 = np.array([])
        img4 = np.array([])
        img5 = np.array([])
        with Image.open("../data/9_5_anomalies/cat_1.jpg") as im:
            img1= np.array(im)
        with Image.open("../data/9_5_anomalies/cat_2.jpg") as im:
            img2= np.array(im)
        with Image.open("../data/9_5_anomalies/cat_3.jpg") as im:
            img3= np.array(im)
        with Image.open("../data/9_5_anomalies/cat_4.jpg") as im:
            img4= np.array(im)
        with Image.open("../data/9_5_anomalies/cat_5.jpg") as im:
            img5= np.array(im)

        arrayOfImages = np.concatenate((img1[np.newaxis,:], img2[np.newaxis,:], img3[np.newaxis,:], img4[np.newaxis,:], img5[np.newaxis,:]),0)

        X = torch.from_numpy(np.array(arrayOfImages))
        self.X = X.permute(0, 3, 1, 2)
        Y = torch.LongTensor([5, 5, 5, 5, 5])
        self.Y = Y.unsqueeze(dim=-1)
        self.labels = ['orange']
        self.input_size = 100

    def __len__(self):
        if MAX_LEN:
            return MAX_LEN
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx] / 255
        y_label = self.Y[idx]

        y_target = x

        return x, y_target, y_label


dataset_apples = DatasetApples()
dataset_anomalies = DatasetAnomalies()

train_test_split = int(len(dataset_apples) * TRAIN_TEST_SPLIT)
dataset_train, dataset_test = torch.utils.data.random_split(
    dataset_apples,
    [train_test_split, len(dataset_apples) - train_test_split],
    generator=torch.Generator().manual_seed(0)
)

dataset_test = torch.utils.data.ConcatDataset([dataset_test, dataset_anomalies])
~~~

## Rezultātu analīze (bez noise):

Sākumā modelis atšķīra kaķu bildes no āboliem:
![anomalies-1.PNG](..%2Fmedia%2Fanomalies-1.PNG)

Taču turpinot trenēties, modelis šķiet identificēja mazāk atšķību no citām bildēm:
![anomalies-2.PNG](..%2Fmedia%2Fanomalies-2.PNG)


## Noise papildinājumi:

Lai implementētu noise tikai treniņu datiem, pārveidoju stage par class variable un pievienoju pārbaudi noise funkcijai:
~~~
   if stage == 'train':
       self.applyNoise(x)
~~~

1. Gaussian (classroom) noise:
~~~
def applyNoise(self, x):
    noise = torch.randn(x.size())
    x[noise < 0.5] = 0
~~~
![cats_with_noise_1.PNG](..%2Fmedia%2Fcats_with_noise_1.PNG)

2. Poisson noise:
~~~
def applyPoissonNoise(self, x):
    noise = torch.poisson(torch.rand(x.size()))
    x[noise == 0] = 0
~~~
![poissont-noise.PNG](..%2Fmedia%2Fpoissont-noise.PNG)

3. Gamma noise:
~~~
def applyGammaNoise(self, x):
    alpha = torch.rand(x.size())
    alpha[alpha <= 0.0] = 0.001 # lai izvairitos no constraints.positive
    beta = torch.ones(x.size())
    gamma = torch.distributions.gamma.Gamma(alpha, beta)
    x[gamma.mean < 0.5] = 0
~~~
![gamma-1.PNG](..%2Fmedia%2Fgamma-1.PNG)
Viens interesants ģenerets kaķis:
![gamma2.PNG](..%2Fmedia%2Fgamma2.PNG)

4. Beta noise:
~~~
def applyBetaNoise(self, x):
    alpha = 2
    beta = 2
    beta = torch.distributions.beta.Beta(alpha, beta)
    x[beta.sample(x.size()) < 0.5] = 0
~~~
![beta05.PNG](..%2Fmedia%2Fbeta05.PNG)

Iedomājos ka jāsamazina troksnis, nevis pusei no pikseliem, bet apmeram 30% (x[beta.sample(x.size()) < 0.5] = 0):
![beta03.PNG](..%2Fmedia%2Fbeta03.PNG)
šķiet ka atdalīšanās kaķu bildēm ir lielāka

5. Laplace noise:
~~~
def applyLaplaceNoise(self, x):
    location = 0
    scale = 2
    laplace = torch.distributions.laplace.Laplace(location, scale)
    x[laplace.sample(x.size()) < 0] = 0
~~~
![laplace_noise.PNG](..%2Fmedia%2Flaplace_noise.PNG)


Mēģināju izmantot InstanceNorm2d, bet sapratu ka nevar izmantot instanceNorm2d ar 1x1 izmēra matricām:
![9_5_reoccuring_error.PNG](..%2Fmedia%2F9_5_reoccuring_error.PNG)

Finally, man ik pa laikam, rādijās ļoti mazs datu diagrammā un nesaprotu kapēc:
![9_5_batch_bug.PNG](..%2Fmedia%2F9_5_batch_bug.PNG)
![9_5_batch_bug_2.PNG](..%2Fmedia%2F9_5_batch_bug_2.PNG)