# Lazy aggregation in Python

One day at work I was making report that wanted to show percentage of total sum for each item in the table.

This made me thinking, how would you model this relationship where total depends on items and item share of total depends on total.

So I made this library that allows defining lazily calculated values and propagating calculation to their dependencies.

Simplest example is [lazy_aggregate](https://github.com/MetalRain/python-lazy-value/blob/main/main.py#L4)

I was thinking what else behaves like that, spreadsheets, neural networks and tried to build simple fully connected neural network with this, example for that in [backprop](https://github.com/MetalRain/python-lazy-value/blob/main/main.py#L26)

I didn't quite get it working, I think I'm missing the derivates and gradient descent part, while error gets calculated my solution doesn't attribute it correctly to neurons and model doesn't learn at all.

## References

How to Code a Neural Network with Backpropagation In Python (from scratch) by Jason Brownlee
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/