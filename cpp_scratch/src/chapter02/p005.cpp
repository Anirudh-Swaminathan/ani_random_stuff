#include<iostream>
#include<cstdint>

/*
 * A function to prove that mixing signed and unsigned types is a bad idea
 * Arguments
 *     x - unsigned int
 */
void someFunc(unsigned int x)
{
    std::cout << "x inside the function is " << x << std::endl;
}

int main()
{
    std::int16_t i(5);
    std::cout << i << std::endl;
    // The following line defines a 8-bit integer, implicitly treated as char
    int8_t in{65};
    std::cout << in << std::endl;
    // The following line issues a warning "overflow in implicit constant conversion" 
    // This is because only 256 characters are possible for 8-bit integers
    // Also to note that even though it was defined by default as signed, it becomes
    // an unsigned 8 bit integer
    int8_t myint = 321;
    std::cout << myint << std::endl;
    int x{-1};
    std::cout << "x outside the function is " << x << std::endl;
    someFunc(x);
    return 0;
}
