#define FAV_NO 4
#define PRINT_JOE
#define FOO 9

#include <iostream>

int main()
{
    std::cout << "My favourite number is " << FAV_NO << std::endl;

    // Check if PRINT_JOE has been defined
    #ifdef PRINT_JOE
    std::cout << "Defined PRINT_JOE" << std::endl;

    // The statement below produces a run-time error
    // std::cout << "PRINT_JOE is " << PRINT_JOE << std::endl;
    #endif // PRINT_JOE

    #ifdef PRINT_BOB
    std::cout << "Defined PRINT_BOB" << std::endl;
    std::cout << "Some syntax error here "
    #endif // PRINT_BOB

    // Check if PRINT_BOB wasn't defined
    #ifndef PRINT_BOB
    std::cout << "In ifndef of PRINT_BOB " << std::endl;
    #endif // PRINT_BOB

    #ifdef FOO
    std::cout << "FOO is " << FOO << std::endl;
    #endif // FOO

    return 0;
}
