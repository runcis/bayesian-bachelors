### 10 mnists


Atlasīju vairākas bildes ar 3 un ar 0, saskaitiju z vektorus ar torch.add:

~~~
# ENCODING 3:
x_to_encode = []
for idx in INDEXES_TO_ENCODE_THREE:
    x_to_encode.append(dataset[idx][0])

x_tensor = torch.stack(x_to_encode)
zs = model.encode_z(x_tensor)

# ENCODING 0:
x2_to_encode = []
for idx in INDEXES_TO_ENCODE_ZERO:
    x2_to_encode.append(dataset[idx][0])

x2_tensor = torch.stack(x2_to_encode)
zs2 = model.encode_z(x2_tensor)

z_comb = torch.add(torch.mean(zs, dim=0), torch.mean(zs2, dim=0))

z_mu = torch.mean(z_comb, dim=0)
z_sigma = torch.std(z_comb, dim=0)
~~~
![10_0_selected.PNG](..%2Fmedia%2F10_0_selected.PNG)
+
![10_3_selected.PNG](..%2Fmedia%2F10_3_selected.PNG)
=
![10_0_3_torch_add.PNG](..%2Fmedia%2F10_0_3_torch_add.PNG)

### 10 balls
Mēģināju dažādos veidos apvienot features, beigās sanāca ar summu/2, domāju ka 32 atlasīti rezultāti ir parak maz, lai iznestu latent features.
~~~
# ENCODING big red:
x_to_encode = []
for idx in INDEXES_TO_ENCODE_BIG_RED:
    x_to_encode.append(dataset[idx])

x_tensor = torch.stack(x_to_encode)
zs = model.encode_z(x_tensor.permute(0, 3, 2, 1))

# ENCODING green small:
x2_to_encode = []
for idx in INDEXES_TO_ENCODE_SMALL_GREEN:
    x2_to_encode.append(dataset[idx])

x2_tensor = torch.stack(x2_to_encode)
zs2 = model.encode_z(x2_tensor.permute(0, 3, 2, 1))

z_comb = (zs + zs2) * 0.5

z_mu = torch.mean(z_comb, dim=0)
z_sigma = torch.std(z_comb, dim=0)
~~~
mazas zaļas:
![10_small_green.PNG](..%2Fmedia%2F10_small_green.PNG)
+ lielas sarkanas:
![big_10_red.PNG](..%2Fmedia%2Fbig_10_red.PNG)
=
![10_small_red.PNG](..%2Fmedia%2F10_small_red.PNG)

mēģināju arī:
mazas zaļas:
![10_small_green.PNG](..%2Fmedia%2F10_small_green.PNG)
+ labais augšējais stūris:
![10_corner.PNG](..%2Fmedia%2F10_corner.PNG)
=
![10_small_corner_fail.PNG](..%2Fmedia%2F10_small_corner_fail.PNG)


