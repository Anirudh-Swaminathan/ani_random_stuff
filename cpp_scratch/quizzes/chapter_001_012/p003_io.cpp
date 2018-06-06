// Program to define the functions to be used in the main program

#include<iostream>
#include "p003_io.h"

int readNumber()
{
    std::cout << "Please enter an intger: ";
    int numb;
    std::cin >> numb;
    return numb;
}

void writeAnswer(int x)
{
    std::cout << "\nThe sum of the entered integers is: " << x;
}
