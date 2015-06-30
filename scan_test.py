import theano
from theano import tensor as T
from ipdb import set_trace as pause


k = T.iscalar("k")
A = T.vector("A")
b = T.vector("b")
def product(prior_result,A,b):
    return prior_result*A, 2*b

# Symbolic description of the result
result, updates = theano.scan(fn=product,
                              outputs_info=T.ones_like(A),
                              non_sequences=A,
                              sequences=b,
                              n_steps=k)

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.
final_result = result[-1]

# compiled function that returns A**k
power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)

print power(range(10),2)
print power(range(10),4)
pause()