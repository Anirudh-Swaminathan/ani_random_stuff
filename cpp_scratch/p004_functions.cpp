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

// function to get value from the user and return it to the caller
int getValueFromUser()
{
    std::cout << "Enter an integer: ";
    int a;
    std::cin >> a;
    return a;
}

void printA()
{
    std::cout << "A" << std::endl;
}

void printB()
{
    std::cout << "B" << std::endl;
}

void printAB()
{
    printA();
    printB();
}

int getNumber()
{
    std::cout << "Enter an integer: ";
    int x;
    std::cin >> x;
    std::cout << "You have entered " << x << " within getNumber()." << std::endl;
    // No return statement gives junk value
}

int main()
{
    std::cout << "Starting main()" << std::endl;

    // Interrupt the main function, call the doPrint() function
    doPrint();

    // The CPU will execute the next line of code once doPrint()
    // function has finished execution
    // This line calls the function return5() and prints the return value

    // !!! The name of the function is a pointer to that function !!!
    // !!! Hence, if the function is called without braces, no compiler error occurs !!!
    // !!! It depends on the compiler what value is output !!!
    std::cout << return5() << std::endl;
    std::cout << return5+2 << std::endl;

    // The main function does nothing with this value and hence this
    // statement is ignored.
    return5();

    // returnNothing() returns no values
    returnNothing();

    // !! The following line will throw a compile error !!
    // std::cout << returnNothing();
    int x = getValueFromUser();
    int y = getValueFromUser();
    std::cout << x << " + " << y << " = " << x+y << std::endl;

    printAB();

    // Calling function that prints to console within the cout statement
    // The function call always takes precedence over the (<<) operator
    // Output:- Whatever is output in getNumber() and then only this cout below
    std::cout << "Calling getNumber() " << getNumber() << std::endl;

    std::cout << "Ending main()" << std::endl;
    return 0;
}
