#!/usr/bin/env python
"""This program is to implement a 2 input calculator"""
import itertools


class Calculator(object):
    """
    A class to handle 2 integer inputs only for a calculator
    """

    valid_operators = {'+', '-', '*', '/', '%', '^'}

    def __init__(self):
        """ Constructor """
        self.quest = ""
        self.expr = []
        self.op1 = 0
        self.op2 = 1
        self.operator = '+'
        self.output = 0

    def input(self):
        """ Function to ask for user input """
        self.quest = raw_input("Please enter an expression to be evaluated with only 2 +ve integers"
                               + "\nValid operations\n+ : Add\n- : Subtract\n* : Multiply\n/ :"
                               + " Divide\n% : Remainder\n^ : Exponent\n\n")
        self.quest = self.quest.replace(" ", "")
        self.expr = ["".join(x) for _, x in itertools.groupby(
            self.quest, key=str.isdigit)]

    @classmethod
    def add(cls, op1, op2):
        """ Add 2 integers

        Arguments
        op1 - Operand 1
        op2 - Operand 2

        Returns
        op1+op2
        """
        return op1 + op2

    @classmethod
    def sub(cls, op1, op2):
        """ Subtracts 2nd integer from the first

        Arguments
        op1 - Operand 1
        op2 - Operand 2

        Returns
        op1-op2
        """
        return op1 - op2

    @classmethod
    def mult(cls, op1, op2):
        """ Multiplies 2 integers

        Arguments
        op1 - Operand 1
        op2 - Operand 2

        Returns
        op1*op2
        """
        return op1 * op2

    @classmethod
    def div(cls, op1, op2):
        """ Divides 2nd integer by the first and returns quotient

        Arguments
        op1 - Operand 1
        op2 - Operand 2

        Returns
        op1/op2
        """
        return op1 / op2

    @classmethod
    def rem(cls, op1, op2):
        """ Returns the remainder when 1st positive integer is divided by the 2nd positive integer

        Arguments
        op1 - Operand 1
        op2 - Operand 2

        Returns
        op1%op2
        """
        return op1 % op2

    @classmethod
    def expo(cls, op1, op2):
        """ Returns operand1 raised to operand2

        Arguments
        op1 - Operand 1
        op2 - Operand 2

        Returns
        op1^op2
        """
        return op1 ** op2

    def calculate(self):
        """ The function to evaluate the expression """
        if len(self.expr) != 3:
            print "Invalid input has been provided!!!"
            return False
        try:
            self.op1 = int(self.expr[0])
            self.op2 = int(self.expr[2])
        except ValueError as ex:
            print "The error was", ex.message
            return False
        self.operator = self.expr[1]
        if self.operator not in Calculator.valid_operators:
            print "Invalid operator has been input!!"
            return False
        if self.operator == '+':
            self.output = self.add(self.op1, self.op2)
        elif self.operator == '-':
            self.output = self.sub(self.op1, self.op2)
        elif self.operator == '*':
            self.output = self.mult(self.op1, self.op2)
        elif self.operator == '/':
            self.output = self.div(self.op1, self.op2)
        elif self.operator == '%':
            self.output = self.rem(self.op1, self.op2)
        elif self.operator == '^':
            self.output = self.expo(self.op1, self.op2)
        else:
            print "Invalid operator entered!!"
            return False
        return True

    def display(self):
        """ Method to display the final result """
        print "\n", self.output


def main():
    """ The main method """
    _ca = Calculator()
    _ca.input()
    success = _ca.calculate()
    if success:
        _ca.display()
    else:
        print "Calculator didn't perform as expected!!"

if __name__ == '__main__':
    main()
