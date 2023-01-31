### 9.uzdevuma MD:

Veidojot pārbaudi anomāliju detektēšanai, sanāca interesanti rezultāti.

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

## Rezultātu analīze:

Sākumā modelis atšķīra kaķu bildes no āboliem:
![anomalies-1.PNG](..%2Fmedia%2Fanomalies-1.PNG)

Taču turpinot trenēties, modelis šķiet identificēja mazāk atšķību no citām bildēm:
![anomalies-2.PNG](..%2Fmedia%2Fanomalies-2.PNG)

Nesapratu kāpēc tā...



Mēģināju izmantot InstanceNorm2d, bet sapratu ka nevar izmantot instanceNorm2d ar 1x1 izmēra matricām:
![9_5_reoccuring_error.PNG](..%2Fmedia%2F9_5_reoccuring_error.PNG)

Finally, man ik pa laikam, rādijās ļoti mazs datu diagrammā un nesaprotu kapēc:
![9_5_batch_bug.PNG](..%2Fmedia%2F9_5_batch_bug.PNG)
![9_5_batch_bug_2.PNG](..%2Fmedia%2F9_5_batch_bug_2.PNG)