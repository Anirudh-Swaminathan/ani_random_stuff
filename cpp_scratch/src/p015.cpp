/*
 * Breaking down the code into functions
*/
#include <iostream>

int getUserInput()
{
    std::cout << "Please enter an integer: ";
    int value;
    std::cin >> value;
    return value;
}

int getMatheticalOperation()
{
    std::cout << "Please enter which operator you want (1 = +, 2 = -, 3 = *, 4 = /): ";
    int op;
    std::cin >> op;
    return op;
}

int calculateResult(int x, int op, int y)
{
    if (op == 1)
        return x + y;
    if (op == 2)
        return x - y;
    if (op == 3)
        return x * y;
    if (op == 4)
        return x / y;
    
    // Handle the error of an invalid operand
    // Addition is the default operation
    return x + y;
}

void printResult(int result)
{
    // Print the output onto the users console
    std::cout << "Your result is: " << result << std::endl;
}

int main()
{
    // Get the first input from the user
    int input1 = getUserInput();

    // Get the mathematical operand from the user
    int operato = getMatheticalOperation();

    // Get the second input from the user
    int input2 = getUserInput();

    // Calculate the result and store it for printing
    int result = calculateResult(input1, operato, input2);

    // Print the result
    printResult(result);
    return 0;
}
