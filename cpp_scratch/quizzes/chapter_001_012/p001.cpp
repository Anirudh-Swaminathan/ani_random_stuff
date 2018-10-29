// A program to input two integers in cpp, add them together and print
// the output onto the console

#include <iostream>

// prompts the user to input an integer
// Returns
//   numb -  the integer entered by the user
int readNumber()
{
    std::cout << "Please enter an intger: ";
    int numb;
    std::cin >> numb;
    return numb;
}

void writeAnswer(int numb)
{
    std::cout << "\nThe sum of the entered integers is: " << numb;
}

int main()
{
    int a = readNumber();
    int b = readNumber();
    writeAnswer(a + b);
    return 0;
}
