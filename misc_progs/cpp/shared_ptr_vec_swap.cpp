#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>

using namespace std;

int main()
{
    vector<shared_ptr<int>> v1;
    vector<shared_ptr<int>> v2;
    for(int i=0; i<10; ++i)
    {
        v1.push_back(make_shared<int>(i));
    }
    for(int j=10; j<20; ++j)
    {
        v2.push_back(make_shared<int>(j));
    }
    cout << "Contents of v1: ";
    for(const auto& v : v1) cout << *v << "@[" << v << "] ";
    cout << "\n";
    cout << "Contents of v2: ";
    for(const auto& v : v2) cout << *v << "@[" << v << "] ";
    cout << "\n";
    cout << "Calling std::swap(v1, v2)\n";
    swap(v1, v2);
    cout << "Contents of v1: ";
    for(const auto& v : v1) cout << *v << "@[" << v << "] ";
    cout << "\n";
    cout << "Contents of v2: ";
    for(const auto& v : v2) cout << *v << "@[" << v << "] ";
    cout << "\n";
    return 0;
}
