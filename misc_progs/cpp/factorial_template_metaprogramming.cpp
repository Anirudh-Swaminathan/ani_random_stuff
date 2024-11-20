#include <bits/stdc++.h>
using namespace std;

template<long long N>
struct Factorial
{
    static constexpr long long value = N * Factorial<N-1>::value;
};

template<>
struct Factorial<0>
{
    static constexpr long long value = 1;
};

int main()
{
    static constexpr long long ret = Factorial<20>::value;
    cout << "20! = " << ret << endl;
    return 0;
}
