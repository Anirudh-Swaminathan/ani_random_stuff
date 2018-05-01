#include <iostream>

// Forward declaration of add() using a function prototype
int add(int a, int b);
// int add(int, int); // also works

int main()
{
    std::cout << "The sum of 3 and 4 is " << add(3, 4) << std::endl;
    return 0;
}

// Defining the body of add
int add(int a, int b)
{
    return a+b;
}

