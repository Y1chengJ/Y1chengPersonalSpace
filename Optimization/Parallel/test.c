#include <stdio.h>

int main() {
    int a = 42;
    int *p = &a;  // p is a pointer holding the address of a

    int *b = &a;
    printf("Address of a: %p\n", b);  // Print the address of a
    

    printf("Address of a: %p\n", p);  // Print the address of a
    printf("Value of a: %d\n", *p);  // Dereference p to get the value of a

    *p = 100;  // Modify the value of a through the pointer
    printf("Updated value of a: %d\n", a);

    return 0;
}
