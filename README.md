---
typora-root-url: ./
---

> 
>
> 📋The README.md for code accompanying the makov decision process neural network for global optimization paper

# Makov Decision Process Neural Network For Global Optimization

This repository is the official implementation of Makov Decision Process Neural Network For Global Optimization. 

> 📋Optional:  The program applies the MDPNN model by default to find the minimum value of the objective function.  If the objective function requires a maximum value, it is necessary to convert the equation into the form of solving the minimum value.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

> 📋This program depends on numpy, math,  random and matplot several Python packages for scientific computing. You must have them installed prior to running MDPN.py.

## Running

By using the model to find the minimum value of the 2 variable rosenbrock equation, set the model parameters as the default, need to run the following command

```train
python MDPNN.py --obj_function rosenbrock
```

> 📋 MDPNN model have some parameters need to be set:
>
> --obj_function  Specifies the name of the objective function, and the equations that can be selected are rosenbrock,griewank,rastrigin,ackley, default is griewank
>
> --Size  Enter a population size for  MDPNN model,  the larger the parameters, the better the optimization, but spend more computing resources, default is 100
>
> --G  Enter iteration times for  MDPNN model, default is 100
>
> --Codel Enter string vector length of each variables, default is 4
>
> --umaxo Enter upper bound of input variables, default is 1000000
>
> --umino  Enter lower bound of input variables, default is -1000000
>
> --n Enter the number of input variables, default is 2
>
> --Po Enter the input probability value controls the proportion of the same character repeated in a population. If greater than the probability, shrink the input variable range and reset the input string vector, default is 0.8

## Results

Our model achieves the following performance on :

​                                             Table 1: Comparison between GA and MDPNN 

| Functione  | algorithm | Searching Cub    | iterations | Global min |
| ---------- | --------- | ---------------- | ---------- | ---------- |
| Rosenbrock | GA        | [-10^6, 10^6]^2  | 100        | 0.11935    |
|            | MDPNN     | [-10^6, 10^6]^2  | 100        | 1.034e-06  |
| Rosenbrock | GA        | [-5.12, 5.12]^10 | 100        | 10.95      |
|            | MDPNN     | [-5.12, 5.12]^10 | 100        | 0.938      |
| Rastrigin  | GA        | [-10^6, 10^6]^2  | 100        | 1.2178     |
|            | MDPNN     | [-10^6, 10^6]^2  | 43         | 0          |
| Griewank   | GA        | [-10^6, 10^6]^2  | 100        | 0.12506    |
|            | MDPNN     | [-10^6, 10^6]^2  | 22         | 0          |
| Griewank   | GA        | [-600, 600]^10   | 100        | 0.55736    |
|            | MDPNN     | [-600, 600]^10   | 100        | 0.0006     |

​                                                                      

​                                                           Table 2: Testing benchmark examples 

| Search indices       | Ackley      | Dixon&Price | Griewank    | Levy        |
| -------------------- | ----------- | ----------- | ----------- | ----------- |
| String Configuration | (5,4,100)   | (5,4,100)   | (5,4,100)   | (5,4,100)   |
| Dimension            | 10          | 10          | 30          | 10          |
| Domain               | [-10,10]^10 | [-10,10]^10 | [-10,10]^30 | [-10,10]^10 |
| Time(sec.)           | 30.41       | 30.64       | 30.41       | 30.94       |
| Iterations           | 100         | 100         | 100         | 100         |
| Prob. control        | 0.9         | 0.8         | 0.8         | 0.8         |
| Theoretical minimum  | 0           | 0           | 0           | 0           |
| Searched minimum     | 0.017       | 0.63        | 0.0011      | 0.584       |

## Contributing

> 📋Makov Decision Process Neural Network For Global Optimization is licensed under the Apache License 2.0