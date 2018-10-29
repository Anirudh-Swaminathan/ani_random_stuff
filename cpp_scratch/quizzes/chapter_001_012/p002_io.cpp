// Functions to be used in p002.cpp are declared here

#include<iostream>

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
