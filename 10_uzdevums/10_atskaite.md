### Fruits dataset

Sākumā izmēģināju veidot decoderi ar groupNorm, šķiet ka rezultāi bija ļoti nevienmērīgi, vienā epochā sanāk skaisti 
kā bildē, bet nākamajā visi augļi var atkal konverģēties.

~~~
self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.GroupNorm(num_channels=out_channels, num_groups=4),
            torch.nn.Mish(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.GroupNorm(num_channels=out_channels, num_groups=2),
            torch.nn.Mish(),
            torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.GroupNorm(num_channels=out_channels, num_groups=4),
            torch.nn.Mish()
        )
~~~
![10_groupNorm.PNG](..%2Fmedia%2F10_groupNorm.PNG)

BatchNorm:
1. Pēc ~100 epochām:
![10_batchNorm.PNG](..%2Fmedia%2F10_batchNorm.PNG)
2. Palaidu un aizgaju uz treniņu - pēc 400 epochām izskatās šādi:
![10_batchNorm2.PNG](..%2Fmedia%2F10_batchNorm2.PNG)
