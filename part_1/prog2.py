#!usr/bin/env python
""" This program is to print n lines of Pascal's triangle """
from __future__ import print_function

class PascalTri(object):
    """
    A class to accept n as input and print n lines of the Pascal traingle
    To know what a Pascals trianlge is, please follow this link
    https://en.wikipedia.org/wiki/Pascal%27s_triangle
    """

    def __init__(self):
        """ Constructor """
        self.n = 3

    def inpu(self):
        """ Method to input n """
        try:
            self.n = int(raw_input("Enter the number of lines to print: "))
        except ValueError as e:
            print("The error was", e.message)
            self.n = 3
        if self.n < 1:
            self.n = 3

    def prin(self):
        """ Print the Pascal's traingle """
        for line in range(1, self.n + 1):
            for _ in range(self.n - line):
                print("", end=' ')
            c = 1
            for i in range(1, line+1):
                print(c, end=' ')
                c = c*(line-i)/i
            print('')

def main():
    """ The main method """
    _pa = PascalTri()
    _pa.inpu()
    _pa.prin()

if __name__ == '__main__':
    main()
