#include <iostream>
#include <memory>
 
struct Foo
{
    Foo(int n = 0) noexcept : bar(n)
    {
        std::cout << "Foo::Foo(), bar = " << bar << " @ " << this << '\n';
    }
    ~Foo()
    {
        std::cout << "Foo::~Foo(), bar = " << bar << " @ " << this << '\n';
    }
    template<typename T>
    const T getBar() const noexcept { return static_cast<T>(bar); }
private:
    int bar;
};
 
int main()
{
    void* ptr = new Foo(42);
    std::cout << "Foo bar is " << ((Foo*)ptr)->getBar<float>() << std::endl;
    // comment out next line to trigger a memory leak
    delete ((Foo*)ptr);
}
