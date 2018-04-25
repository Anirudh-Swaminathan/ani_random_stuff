#include <iostream>

// doPrint is the called function
// void is the return type
void doPrint()
{
    std::cout << "In doPrint()" << std::endl;
    // function doesn't return a value, and so no return statement is required
}

// int is the "type" of the value returned by the function
int return5()
{
    // return integer 5 to the caller
    return 5;
}

// void means the function does not return a value to the caller
void returnNothing()
{
    std::cout << "Hi" << std::endl;
    // This function does not return a value so no return statement is needed
}

int main()
{
    std::cout << "Starting main()" << std::endl;

    // Interrupt the main function, call the doPrint() function
    doPrint();

    // The CPU will execute the next line of code once doPrint()
    // function has finished execution
    // This line calls the function return5() and prints the return value
    std::cout << return5() << std::endl;
    std::cout << return5()+2 << std::endl;

    // The main function does nothing with this value and hence this
    // statement is ignored.
    return5();

    // returnNothing() returns no values
    returnNothing();

    // !! The following line will throw a compile error !!
    // std::cout << returnNothing();

    std::cout << "Ending main()" << std::endl;
    return 0;
}
