def get_size_conv(
        w_in,
        p,
        s,
        k
):
    return (w_in + 2 * p - k) / s + 1

#w_out = w_in/4:
out = get_size_conv(
    w_in=100,
    k=8,
    s=4,
    p=2
)

#w_out = w_in/2:
out2 = get_size_conv(
    w_in=100,
    k=4,
    s=2,
    p=1
)

#w_out = w_in:
out3 = get_size_conv(
    w_in=100,
    k=3,
    s=1,
    p=1
)
#print(out3)



#1 > 4> 16>32> 64>100
def get_size_transposed_conv(
        w_in,
        p,
        s,
        k
):
    return (w_in - 1) * s - 2 * p * k

out4 = get_size_transposed_conv(w_in=100, k = 6, s=1, p= 1 )
print(out4)
exit()