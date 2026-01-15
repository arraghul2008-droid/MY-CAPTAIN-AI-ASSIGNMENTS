def fibonacci(limit):
    # Starting values
    a = 0
    b = 1
    
    print("Fibonacci Sequence:")
    
    for i in range(limit):
        print(a, end=" ") # Show the current number
        
        
        a, b = b, a + b


fibonacci(10)