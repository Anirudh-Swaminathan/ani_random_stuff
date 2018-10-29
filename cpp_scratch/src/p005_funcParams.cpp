#include <iostream>

void doPrint()
{
    std::cout << "In doPrint()" << std::endl;
}

void printValue(int x)
{
    std::cout << x << std::endl;
}

int add(int x, int y)
{
    return x+y;
}

void printValues(int x, int y)
{
    std::cout << x << std::endl;
    std::cout << y << std::endl;
}

int multiply(int z, int w)
{
    return z*w;
}

int main()
{
    printValue(6);
    std::cout << add(4, 5) << std::endl;
    printValues(6, 7);
    std::cout << multiply(4, 5) << std::endl;
    std::cout << add(1+2, 3*4) << std::endl;
    int a = 5;
    std::cout << add(a, a) << std::endl;
    std::cout << add(1, multiply(2, 3)) << std::endl;
    std::cout << add(1, add(2, 3)) << std::endl;
    return 0;
}
