#non local variable
#nested function 
def display(): #outer function 
    x=50
    def add() : #outer function 
       x=100
       print("inside the function add:",x)
    add()
    print("outer function display:", x)
 display()   