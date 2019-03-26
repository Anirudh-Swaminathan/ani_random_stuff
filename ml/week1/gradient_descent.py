"""
Program to minimize a function using the gradient descent algorithm

It is up to the user to enter a convex function and an initial solution
Default initial solution is 0
"""
from __future__ import print_function

class GradientDescent(object):
    """ An object to implement gradient descent methods """

    def __init__(self):
        """ Constructor """
        self.coeffs = []
        self.ord = 0
        self.ini = 0.0
        self.iters = 1000
        self.alpha = 0.01

        # 42 - The answer to the Ultimate Question of Life, The Universe and Everything
        self.soln = 4.2

    def input(self):
        """
        Method to input the function coefficients
        Function is of the form a + bx + cx^2 + dx^3 + ...
        """
        self.coeffs = map(float, raw_input("Please enter space separated "
                                           +"coefficients from x^0 to x^n\n").split(' '))
        self.ord = len(self.coeffs)
        try:
            self.ini = float(raw_input("Please enter a valid initial solution: "))
        except ValueError as ve:
            print("Error!!", ve.message)
            print("Initializing default value")
        try:
            self.iters = int(raw_input("Enter the number of iterations(<=10000): "))
        except ValueError as ve:
            print("Error!!", ve.message)
            print("Initializing default value")
        if self.iters > 10000:
            self.iters = 1000
        try:
            self.alpha = float(raw_input("Enter the learning rate(0.0001 to 3): "))
        except ValueError as ve:
            print("Error!", ve.message)
            print("Initializing default value")
        if self.alpha < 0.0001 or self.alpha > 3:
            self.alpha = 0.01

    def fx(self, x):
        """
        Method to compute the value of the function at a particular point

        Arguments
        x - The point at which the function is to be computed

        Returns
        f(x) - The numerical value of the function at x
        """
        return sum([co*(x**ind) for ind, co in enumerate(self.coeffs)])

    def grad(self, x):
        """
        Method to find f'(x) at that specific point

        Arguments
        x - The point at which f'(x) is to be calculated

        Returns
        f'(x) - The numerical value of the derivative of f(x) at x
        """
        return sum([co*ind*(x**(ind-1)) for ind, co in enumerate(self.coeffs) if ind > 0])

    def gradient_descent(self):
        """
        Function to find the minima of the given (supposed) convex function
        using gradient descent
        """
        self.soln = self.ini
        for _ in range(self.iters):
            self.soln = self.soln - self.alpha*self.grad(self.soln)
        return self.soln

    def prin(self):
        """ Method to display the result """
        print("\nThe function is")
        for ind, co in enumerate(self.coeffs):
            if ind > 0:
                print(" + "+str(co)+"x^"+str(ind), end=' ')
            else:
                print(""+str(co), end=' ')
        print("\nThe solution parameters were:\nLearning Rate:", self.alpha)
        print("Number of iterations:", self.iters)
        print("Initial Solution:", self.ini)
        print("Initial f(x):", self.fx(self.ini))
        print("Final Solution(minima):", self.soln)
        print("Minimum value of f(x):", self.fx(self.soln))

def main():
    """ The main method """
    gd = GradientDescent()
    gd.input()
    gd.gradient_descent()
    gd.prin()

if __name__ == '__main__':
    main()
