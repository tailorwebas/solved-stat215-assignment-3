Download Link: https://assignmentchef.com/product/solved-stat215-assignment-3
<br>






<strong>Problem 1: </strong><em>Variational inference.</em>

Standard VI minimizes KL (<em>q</em>(<em>z</em>) II <em>p</em>(<em>z </em>| <em>x</em>)), the Kullback-Leibler divergence from the variational approx­imation <em>q</em>(<em>z</em>) to the true posterior <em>p</em>(<em>z </em>| <em>x</em>). In this problem we will develop some intuition for this optimization problem. For further reference, see Chapter 10 of <em>Pattern Recognition and Machine Learning </em>by Bishop.

<sub>fl<em>D</em></sub>

<ul>

 <li>Let = {<em>q</em>(<em>z</em>) : <em>q</em>(<em>z</em>) = <em>d</em>=1 JV (<em>z</em><em>d </em>| <em>m</em><em>d</em>, <em>v</em><sup>2</sup><em><sub> d</sub></em>)} denote the set of Gaussian densities on <em>z </em>E R<em><sup>D</sup></em> with</li>

</ul>

diagonal covariance matrices. Solve for

<em>q</em><em>* </em>= argmin KL(<em>q</em>(<em>z</em>) II ~ (<em>z </em>| <em>µ</em>, )) , where is an arbitrary covariance matrix.

Your answer here.

<ul>

 <li>Now solve for <em>q<sup>*</sup></em> E that minimizes the KL in the opposite direction, <em>q</em><em>* </em>= argmin KL(K (<em>z </em>| <em>µ</em>, ) II <em>q</em>(<em>z</em>)). Your answer here.</li>

 <li>Plot the contour lines of your solutions to parts (a) and (b) for the case where</li>

</ul>

[ ~                [                                                          ~

0                                 1 0.9

<em>µ </em>=   ,                   =       .

0                              0.9 1




<strong>Problem 2:</strong><em> Variational autoencoders (VAE’s)</em>

In class we derived VAE’s as generative models<em> p</em>(<em>x</em>,<em> z</em>;<em> θ</em>) of observations<em> x</em> ∈ R<em><sup>P</sup></em> and latent variables<em> z</em> ∈ R<em><sup>D</sup></em>, with parameters<em> θ</em>. We used variational expectation-maximization to learn the parameters<em> θ</em> that maximize a lower bound on the marginal likelihood,

log <em>p</em>(<em>x</em>;<em> θ</em>) ≥ ~<em>N </em>E<em>q</em>(<em>z</em><em>n</em>|<em>x</em><em>n</em>,<em>φ</em>) [log <em>p</em>(<em>x<sub>n</sub></em>,<em>z<sub>n</sub></em>;<em> θ</em>) − log<em>q</em>(<em>z<sub>n</sub></em> |<em> x<sub>n</sub></em>,<em> φ</em>)] 2(<em>θ</em>,<em> φ</em>).

<em>n</em>=1

The difference between VAE’s and regular variational expectation-maximization is that we constrained the variational distribution<em> q</em>(<em>z</em> |<em> x</em>,<em> φ</em>) to be a parametric function of the data; for example, we consid­ered,

<em>q</em>(<em>z</em> | <em>x<sub>n</sub></em>,<em> φ</em>) = jV ( <em>z<sub>n</sub></em> | <em>µ</em>(<em>x</em>;<em> φ</em>), diag([<em>σ</em>(<em>x</em>;<em> φ</em>),…,<em> σ</em>(<em>x<sub>n</sub></em>;<em> φ</em>)])) ,

where<em> µ </em>: R<em><sup>P</sup></em> → R<em>D</em> and<em> σ</em>2 <em>d</em> : R<em><sup>P</sup></em> → R+ are functions parameterized by<em> φ</em> that take in a datapoint<em> x<sub>n</sub></em> and output means and variances of<em> z<sub>n</sub></em>, respectively. In practice, it is common to implement these functions with neural networks. Here we will study VAE’s in some special cases. For further reference, see Kingma and Welling (2019), which is linked on the course website.

<ul>

 <li>Consider the linear Gaussian model factor model, <em>p</em>(<em>x<sub>n</sub></em>,<em>z<sub>n</sub></em>;<em>θ</em>) = J (<em>z<sub>n</sub></em>;0, <em>I</em>)J (<em>x<sub>n</sub></em> | <em>Az<sub>n</sub></em>,<em> V</em>),</li>

</ul>

where<em> A</em> ∈ R<em><sup>P</sup></em><sup>×</sup><em><sup>D</sup></em>,<em> V</em> ∈ R<em><sup>P</sup></em><sup>×</sup><em><sup>P</sup></em> is a diagonal, positive definite matrix, and<em> θ</em> = (<em>A</em>,<em> V</em>). Solve for the true posterior<em> p</em>(<em>z<sub>n</sub></em> |<em> x<sub>n</sub></em>,<em> θ</em>).

Your answer here.

<ul>

 <li>Consider the variational family of Gaussian densities with diagonal covariance, as described above, and assume that<em> µ</em>(<em>x</em>;<em> φ</em>) and log<em> σ</em><sup>2</sup><em><sub> d</sub></em>(<em>x</em>;<em> φ</em>) are linear functions of<em> x</em>. Does this family contain the true posterior? Find the member of this variational family that maximizes Y(<em>θ</em>,<em> φ</em>) for fixed<em> θ</em>. (Hint: use your answer to Problem 1a.)</li>

</ul>

Your answer here.

<ul>

 <li>Now consider a simple nonlinear factor model,</li>

</ul>

<em>p</em>(<em>x<sub>n</sub></em>,<em>z<sub>n</sub></em>;<em> θ</em>) = J (<em>z<sub>n</sub></em> | 0, <em>I</em>) ~<em>P </em>JV(<em>x<sub>np</sub></em> |<em><sub> e</sub></em><em>a</em><sup>T</sup><em>p</em><em>z</em><em>n</em><sub>,</sub><em> v<sub>p</sub></em>),

<em>p</em>=1

parameterized by<em> a<sub>p</sub></em> ∈ R<em>D</em> and<em> v<sub>p</sub></em> ∈ R+. The posterior is no longer Gaussian, since the mean of<em> x</em><em>np</em> is a nonlinear function of the latent variable.<sup>1</sup>

Generate a synthetic dataset by sampling<em> N</em> = 1000 datapoints from a<em> D</em> = 1,<em> P</em> = 2 dimensional model with<em> A </em>= [1.2,<sup> 1</sup><sup>]</sup><sup>T</sup> and<em> v<sub>p</sub></em> = 0.1 for<em> p</em> = 1, 2. Use the reparameterization trick and automatic differentiation to perform stochastic gradient descent on −2(<em>θ</em>,<em>φ</em>).

Make the following plots:

<sup>1</sup>For this particular model, the expectations in 2(<em>θ</em>,<em> φ</em>) can still be computed in closed form using the fact that E[<em>e</em><em>z</em>] =<em> e</em><em>µ</em>+12 <em>σ</em>2 for<em> z</em> ∼ 4 (<em>µ</em>,<em> σ</em><sup>2</sup>).




 A scatter plot of your simulated data (with equal axis limits).

<ul>

 <li>A plot of 2(<em>0</em>, <em>4,</em>) as a function of SGD iteration.</li>

 <li>A plot of the model parameters (<em>A</em>11,<em>A</em>21, <em>v</em>1, <em>v</em>2) as a function of SGD iteration.</li>

 <li>The approximate Gaussian posterior with mean <em>j</em>(<em>x</em>; <em>4,</em>) and variance <em>o</em><sup>2</sup><sub> 1</sub>(<em>x</em>; <em>4,</em>) for <em>x </em>E {(0, 0), (1, 1), (10, 7)} using the learned parameters <em>4,</em>.</li>

 <li>The true posterior at those points. (Since <em>z </em>is one dimensional, you can compute the true posterior with numerical integration.)</li>

</ul>

Comment on your results. Your results here.




<strong>Problem 3:</strong><em> Semi-Markov models</em>

Consider a Markov model as described in class and in, for example, Chapter 13 of<em> Pattern Recogntion and</em> <em>Machine Learning</em> by Bishop,

<em>p</em>(<em>z</em>1:<em>T</em> |<em> π</em>,<em>A</em>) =<em> p</em>(<em>z</em>1 |<em> π</em>) ~<em>T </em><em>p</em>(<em>z<sub>t</sub></em> | <em>z</em><em>t</em>−1, <em>A</em>),

<em>t</em>=2

where<em> z<sub>t</sub></em> ∈ {1,. . . ,<em>K</em>} denotes the “state,” and

<em>p</em>(<em>z</em>1 =<em> i</em>) =<em> π</em><em>i</em>

<em>p</em>(<em>z<sub>t</sub></em> =<em> j</em> | <em>z</em><em>t</em>−1 =<em> i</em>,<em>A</em>) = <em>A</em><em>ij</em>.

We will study the distribution of state durations—the length of time spent in a state before transitioning. Let<em> d</em> ≥ 1 denote the number of time steps before a transition out of state<em> z</em>1. That is,<em> z</em>1 =<em> i</em>,. . . , <em>z</em><em>d </em>=<em> i</em> for some<em> i</em>, but<em> z</em><em>d</em>+1 =5<em> i</em>.

<ul>

 <li>Show that<em> p</em>(<em>d</em> |<em> z</em>1 =<em> i</em>,<em>A</em>) = Geom(<em>d</em> |<em> p</em><em>i</em>), the probability mass function of the geometric distribu­ Solve for the parameter<em> p</em><em>i</em> as a function of the transition matrix<em> A</em>.</li>

</ul>

Your answer here.

<ul>

 <li>We can equivalently represent<em><sub> z</sub></em><sub>1:</sub><em><sub>T</sub></em> as a set of states and durations {(˜<em>z</em><em>n</em>,<em><sub> dn</sub></em><sub>)</sub><sub>}</sub><em><sub>Nn</sub></em><sub>=</sub><sub>1</sub><sub>,</sub> where<sup> ˜</sup><em>z<sub>n</sub></em> ∈ {1,.. . ,<em>K</em>}  {<sup>˜</sup><em>z</em><em>n</em>−1} denotes the index of the<em> n</em>-th visited state and<em> d</em><em>n</em> ∈ N denotes the duration spent in that state before transition. There is a one-to-one mapping between states/durations and the original state sequence:</li>

</ul>




,<sup>˜</sup><em>z</em>2, … ,<sup>˜</sup><em>z</em>2

<u>~ ~~ ~</u>

<em>d</em>2 times

<sup>˜</sup><em>z</em><em>N</em>,.. .,<sup>˜</sup><em>z</em><em>N</em>

<u>~ ~~ </u>~

<em>d</em><em>N</em> times




Show that the probability mass function of the states and durations is of the form

<em>p</em>({(<em>n</em>,<em>d</em><em>n</em>)}=1) = <em>p</em>(1 | <em>π</em>) [<em>f</em>i<em>p</em>(<em>d</em><em>n </em>| <em>z<sub>n</sub></em>,<em>A</em>) <em>p</em>(<em>z</em><em>n</em>+1 | <sup>˜</sup><em>z<sub>n</sub></em>,<em>A</em>)]<em> p</em>(<em>d</em><em>N</em>| <sup>˜</sup><em>z</em><em>N</em>, <em>A</em>), <em>n</em>=1

and derive each conditional probability mass function. Your answer here.

(c)<em> Semi-Markov</em> models replace<em> p</em>(<em>d<sub>n</sub></em> |<sup> ˜</sup><em>z<sub>n</sub></em>) with a more flexible duration distribution. For example, consider the model,

<em>p</em>(<em>d<sub>n</sub></em> | <sup>˜</sup><em>z<sub>n</sub></em>) = NB(<em>d<sub>n</sub></em> |<em> r</em>,<em> θ</em><sup>˜</sup><em>z<sub>n</sub></em>),

where<em> r</em> ∈ N and<em> θ</em><em>k</em> ∈ [0, 1] for<em> k</em> = 1, . . . ,<em> K</em>. Recall from Assignment 1 that the negative binomial distribution with integer<em> r</em> is equivalent to a sum of<em> r</em> geometric random variables. Use this equivalence to write the semi-Markov model with negative binomial durations as a Markov model on an extended set of states<em> s<sub>n</sub></em> ∈ {1,.. . ,<em>Kr</em>}. Specifically, write the transition matrix for<em> p</em>(<em>s<sub>n</sub></em> |<em> s</em><em>n</em>−1) and the mapping from<em> s<sub>n</sub></em> to<em> z<sub>n</sub></em>.

Your answer here.