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
>
> 

## Results

Our model achieves the following performance on :

image-20200610194115943

image-20200610194404668


## Contributing

> 📋Makov Decision Process Neural Network For Global Optimization is licensed under the Apache License 2.0