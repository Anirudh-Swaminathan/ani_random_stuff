// Program to learn about fundamental data types, and their initialization methods

#include<iostream>

int main()
{
    // Defining variables
    bool bValue;
    char chValue;
    int iValue;
    float fValue;
    double dValue;

    // void won't work
    // void vValue;
    // Copy initialization
    int nValue = 5;

    // Direct initialization
    int ndValue(5);

    // Uniform initialization
    // Favour this method initialization if using C++ 11 or greater
    int value{5};
    int zValue{};

    std::cout << "nValue: " << nValue << " and ndValue: " << ndValue << std::endl;
    std::cout << "value: " << value << " and zValue: " << zValue << std::endl;
    
    // Assigning values to variables
    int aValue;
    aValue = 5;
    std::cout << "aValue: " << aValue << std::endl;

    // Defining multiple variables
    int a, b;
    int c(7), d(8);
    int e{9}, f{10};
    std::cout << "a:" << a << " b:" << b << " c:" << c << " d:" << d
        << " e:" << e << " f:" << f << std::endl;
    return 0;
}
