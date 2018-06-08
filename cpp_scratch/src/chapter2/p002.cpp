// Program to explore void type
//
#include<iostream>

// void in the parameter list is usually not used
int getValue(void)
{
    int x;
    std::cin >> x;
    return x;
}

// Empty parameter list is implicit void
int getValueEmpty()
{
    int x;
    std::cin >> x;
    return x;
}

// NOTE- Here void as the return type of the function means that the function does not return any value
void writeValue(int x)
{
    std::cout << "The value of x is: " << x << std::endl;
    // No return statment since the return type is void
}

int main()
{
    // This won't work
    // void vValue;
    int a = getValue();
    writeValue(a);
    int b{getValueEmpty()};
    writeValue(b);
    return 0;
}
