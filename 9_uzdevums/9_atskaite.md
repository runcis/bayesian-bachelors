# Task 9


Ļoti interesants. Saskāros ar vairākām problēmām bet visas ātrāk vai vēlāk atrisisināju(colab negāja, decoding slānis nedeva correct shape u.c. )

1. Pamaniju ka tīrīšanas slānis neuzlabo rezultātus.

Ar tirisanas slāni:
![ar tirisanas slani](../media/9_ar_tirizanas_slani.PNG)

Bez tīrīšanas slāņa:
![bez tirisanas slani](../media/9_bez_tirizanas_slana.PNG)


Pirmais mēģinajums denoising:

~~~
    def __getitem__(self, idx):
        x = self.X[idx] / 255
        y_label = self.Y[idx]
        dims = x.shape
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    if random.randint(1, 10) < 6:
                        x[i][j][k] = 0

        return x, y_target, y_label
~~~

algoritms bija ļoti lēns.
Norakstīts no lekcijas. Nesparotu kapēc tas strādā - vai tad mēs neuzģenērjam jaun umatricu ar pilnīgi random vērībām no x paņemot tikai size?

~~~
    def applyNoise(self, x):
        if np.random.random() < 0.5:
            noise = torch.randn(x.size())
            x[noise < 0.5] = 0
~~~