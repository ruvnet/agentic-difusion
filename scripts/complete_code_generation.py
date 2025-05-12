#!/usr/bin/env python3
"""
Script to generate a complete implementation of a recursive factorial function.
"""

def factorial(n):
    """
    Calculate the factorial of a number using recursion.
    
    Args:
        n (int): The number to calculate factorial for
        
    Returns:
        int: The factorial of n
    """
    # Base case: factorial of 0 or 1 is 1
    if n <= 1:
        return 1
    
    # Recursive case: n * factorial(n-1)
    return n * factorial(n-1)

def main():
    """
    Test the factorial function with various inputs.
    """
    test_numbers = [0, 1, 5, 10]
    
    print("\n" + "=" * 80)
    print("COMPLETE FACTORIAL IMPLEMENTATION")
    print("=" * 80)
    
    for num in test_numbers:
        result = factorial(num)
        print(f"factorial({num}) = {result}")

if __name__ == "__main__":
    main()