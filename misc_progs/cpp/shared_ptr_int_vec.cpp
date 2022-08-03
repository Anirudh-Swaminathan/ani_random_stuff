#include <iostream>
#include <vector>
#include <memory>
using namespace std;

int main()
{
    int n = 27;
    vector<shared_ptr<int>> vec;
    shared_ptr<int> s = make_shared<int>(n);
    printf("s is %d\n", *s);
    printf("s.use_count() is %ld\n", s.use_count());
    vec.push_back(s);
    printf("After vec.push_back(s), s.use_count() is %ld\n", s.use_count());
    for(const auto& v : vec)
    {
        printf("v is %d\n", *v);
        // legal op since v is const (so the memory address it holds cannot be changed)
        // but the value that the memory address points to can be changed since that is
        // not const
        *v = 4;
        printf("v.use_count() is %ld\n", v.use_count());
    }
    printf("s is %d\n", *s);
    printf("s.use_count() is %ld\n", s.use_count());
    printf("Loop by value\n");
    for(auto v : vec)
    {
        printf("v is %d\n", *v);
        // legal op since v is const (so the memory address it holds cannot be changed)
        // but the value that the memory address points to can be changed since that is
        // not const
        *v = 14;
        printf("v.use_count() is %ld\n", v.use_count());
    }
    printf("s is %d\n", *s);
    printf("s.use_count() is %ld\n", s.use_count());
    // clear vec
    cout << "Clearing vec with vec.clear(). Expecting use_count() to decrease during this\n";
    vec.clear();
    printf("s is %d\n", *s);
    printf("s.use_count() is %ld\n", s.use_count());
    return 0;
}
