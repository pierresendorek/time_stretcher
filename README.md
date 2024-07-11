# time_stretcher

We use quantile regression to draw according to a given probability distribution, say $p_x$.

We use the fact that

$$\arg \min_m E[a (x - m) + |x - m|] = \text{quantile}[ \frac{a + 1}{2}]$$

It means that if we fit a neural network such as

$$f(a) = \arg \min_m E[a (x - m) + |x - m|]$$

We will have 

$$f(u) \sim p_x$$

where $u$ is a $U([0,1])$ random variable


---

Recurrent neural network

We train a recurrent neural network $G$ to generate a random sequence whose probability law approximates the probability distribution of a given sequence $x_{0:T-1}$

$$G(u_t, t, h_{t-1}) \sim p(x_t|x_{0:t-1})$$

