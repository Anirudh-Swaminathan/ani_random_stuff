#include<iostream>
using namespace std;

void func(int a)
{
    cout << "In func pass by value, a = " << a << endl;
    cout << "In func pass by value, &a = " << &a << endl;
    a = 50;
    cout << "In func pass by value, a = " << a << endl;
    cout << "In func pass by value, &a = " << &a << endl;
    return;
}

void funcref(int& a)
{
    cout << "In func pass by reference, a = " << a << endl;
    cout << "In func pass by reference, &a = " << &a << endl;
    a = 75;
    cout << "In func pass by reference, a = " << a << endl;
    cout << "In func pass by reference, &a = " << &a << endl;
    return;
}

void func(int *a)
{
    cout << "In func pass by pointer value, *a = " << *a << endl;
    cout << "In func pass by pointer value, a = " << a << endl;
    cout << "In func pass by pointer value, &a = " << &a << endl;
    *a = 100;
    cout << "In func pass by pointer value, *a = " << *a << endl;
    cout << "In func pass by pointer value, a = " << a << endl;
    cout << "In func pass by pointer value, &a = " << &a << endl;
}

void funcref(int *& a)
{
    cout << "In func pass by pointer reference, *a = " << *a << endl;
    cout << "In func pass by pointer reference, a = " << a << endl;
    cout << "In func pass by pointer reference, &a = " << &a << endl;
    *a = 150;
    cout << "In func pass by pointer reference, *a = " << *a << endl;
    cout << "In func pass by pointer reference, a = " << a << endl;
    cout << "In func pass by pointer reference, &a = " << &a << endl;
}

void func(int **a)
{
    cout << "In func pass by double pointer, **a = " << **a << endl;
    cout << "In func pass by double pointer, *a = " << *a << endl;
    cout << "In func pass by double pointer, a = " << a << endl;
    cout << "In func pass by double pointer, &a = " << &a << endl;
    **a = 125;
    cout << "In func pass by double pointer, **a = " << **a << endl;
    cout << "In func pass by double pointer, *a = " << *a << endl;
    cout << "In func pass by double pointer, a = " << a << endl;
    cout << "In func pass by double pointer, &a = " << &a << endl;
}

void func(void **a)
{
    int** q = (int**)a;
    cout << "In func pass by void double pointer, **a = " << **(int**)a << endl;
    cout << "In func pass by void double pointer, *a = " << *a << endl;
    cout << "In func pass by void double pointer, a = " << a << endl;
    cout << "In func pass by void double pointer, &a = " << &a << endl;
    cout << "In func pass by void double pointer, **q = " << **q << endl;
    cout << "In func pass by void double pointer, *q = " << *q << endl;
    cout << "In func pass by void double pointer, q = " << q << endl;
    cout << "In func pass by void double pointer, &q = " << &q << endl;
    **q = 175;
    cout << "In func pass by void double pointer, **a = " << **(int**)a << endl;
    cout << "In func pass by void double pointer, *a = " << *a << endl;
    cout << "In func pass by void double pointer, a = " << a << endl;
    cout << "In func pass by void double pointer, &a = " << &a << endl;
    cout << "In func pass by void double pointer, **q = " << **q << endl;
    cout << "In func pass by void double pointer, *q = " << *q << endl;
    cout << "In func pass by void double pointer, q = " << q << endl;
    cout << "In func pass by void double pointer, &q = " << &q << endl;
}

int main()
{
    int a = 25;
    int* ptr = &a;
    cout << "main initializes a = " << a << endl;
    cout << "In main, memory address of a = " << &a << endl;
    cout << "In main, ptr = " << ptr << endl;
    cout << "In main, memory address of ptr = " << &ptr << endl;
    func(a);
    cout << "In main, a = " << a << endl;
    cout << "In main, memory address of a = " << &a << endl;
    cout << "In main, ptr = " << ptr << endl;
    cout << "In main, memory address of ptr = " << &ptr << endl;
    funcref(a);
    cout << "In main, a = " << a << endl;
    cout << "In main, memory address of a = " << &a << endl;
    cout << "In main, ptr = " << ptr << endl;
    cout << "In main, memory address of ptr = " << &ptr << endl;
    func(ptr);
    cout << "In main, a = " << a << endl;
    cout << "In main, memory address of a = " << &a << endl;
    cout << "In main, ptr = " << ptr << endl;
    cout << "In main, memory address of ptr = " << &ptr << endl;
    funcref(ptr);
    cout << "In main, a = " << a << endl;
    cout << "In main, memory address of a = " << &a << endl;
    cout << "In main, ptr = " << ptr << endl;
    cout << "In main, memory address of ptr = " << &ptr << endl;
    func(&ptr);
    cout << "In main, a = " << a << endl;
    cout << "In main, memory address of a = " << &a << endl;
    cout << "In main, ptr = " << ptr << endl;
    cout << "In main, memory address of ptr = " << &ptr << endl;
    func((void**)&ptr);
    cout << "In main, a = " << a << endl;
    cout << "In main, memory address of a = " << &a << endl;
    cout << "In main, ptr = " << ptr << endl;
    cout << "In main, memory address of ptr = " << &ptr << endl;
    return 0;
}
